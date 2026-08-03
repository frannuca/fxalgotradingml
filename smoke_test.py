"""Smoke test for the N-independent-per-asset-LSTM architecture (no
CopulaLSTM cross-asset stage - by DEFAULT every asset's LSTM sees ONLY its
own features; `cross_pairs` opts specific pairs into also seeing specific
OTHER pairs' feature blocks, see PredictionModel's docstring), each
followed by a CAUSAL self-attention layer over the time axis, plus the
(mu, sigma) + neutral-band heads - fully deterministic (no NoisyNet head,
no input noise) - synthetic data, no DB, no network. Stubs data.db /
data.fx_downloader before importing models.portfolio_lstm."""
import sys, types, tempfile, os
import numpy as np

# --- stub the DB-touching modules so importing portfolio_lstm works ---
db_stub = types.ModuleType("data.db")
db_stub.get_time_series = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no db in test"))
db_stub.upsert_pairs = lambda *a, **k: None
fx_stub = types.ModuleType("data.fx_downloader")
fx_stub.FXDownloader = object
fx_stub.MAJOR_FX_PAIRS = {}
data_pkg = types.ModuleType("data"); data_pkg.__path__ = []
sys.modules["data"] = data_pkg
sys.modules["data.db"] = db_stub
sys.modules["data.fx_downloader"] = fx_stub

import torch
from models import portfolio_lstm as pl

# Captured before test 7 permanently monkeypatches pl.load_close_prices -
# test 12b needs the REAL function (it's testing load_close_prices itself),
# reached directly rather than via the (by then patched) module attribute.
_real_load_close_prices = pl.load_close_prices

torch.manual_seed(0)
np.random.seed(0)

n_assets, n_channels, lookback, n_samples = 3, 4, 20, 200
pairs = ["A", "B", "C"]

# --- 1. forward returns dense (mu, sigma), sigma > 0, sigma ~ 1 at init ---
model = pl.PredictionModel(n_assets=n_assets, pairs=pairs, n_channels=n_channels, hidden_size=8)
X = torch.randn(n_samples, lookback, n_assets * n_channels)
mu, sigma = model(X)
assert mu.shape == (n_samples, lookback, n_assets) and sigma.shape == (n_samples, lookback, n_assets)
assert (sigma > 0).all(), "sigma must be strictly positive"
assert 0.5 < sigma.mean() < 2.0, f"init sigma should be ~1, got {sigma.mean():.3f}"
print(f"1. forward OK: mu {tuple(mu.shape)}, sigma mean {sigma.mean():.3f}")

# --- 1a. cross_pairs input slicing: default is fully independent (each
# asset's LSTM sees ONLY its own n_channels); explicit cross_pairs widens
# specific pairs' own input to include specific others' full blocks too. ---
assert model.included_indices == [[0], [1], [2]]
assert all(a.input_size == n_channels for a in model.assets)
cross_model = pl.PredictionModel(
    n_assets=n_assets, pairs=pairs, n_channels=n_channels, hidden_size=8,
    cross_pairs={"A": ["B", "C"]},
)
assert cross_model.included_indices == [[0, 1, 2], [1], [2]]
assert cross_model.assets[0].input_size == 3 * n_channels
assert cross_model.assets[1].input_size == n_channels and cross_model.assets[2].input_size == n_channels
try:
    pl.PredictionModel(n_assets=n_assets, pairs=pairs, n_channels=n_channels, cross_pairs={"A": ["ZZZ"]})
    raise AssertionError("unknown pair in cross_pairs should have raised ValueError")
except ValueError:
    pass
print("1a. cross_pairs input slicing OK: default independent, explicit linking widens input, unknown pair rejected")

# --- 1b. causal self-attention: earlier days' predictions must not change
# when a LATER day's own input is perturbed - the attention layer added
# after each AssetLSTM's LSTM is causally masked specifically so every
# day's dense supervision stays "as of that day only" (see AssetLSTM's
# docstring); an unmasked (bidirectional) layer would leak the future. ---
model.eval()
with torch.no_grad():
    mu_a, _ = model(X)
X_perturbed = X.clone()
X_perturbed[:, -1, :] += 5.0  # perturb only the LAST timestep's own input
with torch.no_grad():
    mu_b, _ = model(X_perturbed)
assert torch.allclose(mu_a[:, :-1, :], mu_b[:, :-1, :], atol=1e-5), (
    "earlier-day predictions changed when a LATER day's input was perturbed - causal mask is leaking"
)
assert not torch.allclose(mu_a[:, -1, :], mu_b[:, -1, :]), (
    "perturbing the last day's own input should change its own prediction"
)
model.train()
print("1b. causal self-attention OK: earlier days unaffected by a later day's perturbed input")

# --- 1c. TRUE per-asset training independence: adding a second, UNRELATED
# (pure noise) asset must not change the first asset's own training
# dynamics at all - under the old design (one shared optimizer, one loss
# averaged across every asset) it would have (the signal asset's own
# gradient diluted by 1/n_assets); under per-asset optimizers/losses/
# checkpoints it must not (see train_prediction_model's own docstring).
# dropout=0 here specifically to avoid a RNG-state confound: dropout draws
# from the same global RNG, and interleaving a second asset's draws each
# epoch would shift the first asset's own dropout masks between the solo
# and paired runs, which is a real but IRRELEVANT effect for what this
# checks (parameter/optimizer independence, not RNG bookkeeping). ---
torch.manual_seed(1)
X_solo = torch.randn(n_samples, lookback, n_channels)
z_solo = torch.zeros(n_samples, lookback, 1)
signal_solo = X_solo[:, -1, 0]
z_solo[:, -1, 0] = signal_solo + 0.3 * torch.randn(n_samples)
model_solo = pl.PredictionModel(n_assets=1, pairs=["A"], n_channels=n_channels, hidden_size=8, dropout=0.0)
pl.train_prediction_model(model_solo, X_solo, z_solo, epochs=80, lr=1e-2, bce_weight=1.0)
model_solo.eval()
with torch.no_grad():
    mu_solo, _ = model_solo(X_solo)
hit_solo = float(((mu_solo[:, -1, 0] > 0) == (z_solo[:, -1, 0] > 0)).float().mean())

torch.manual_seed(1)  # SAME seed - asset A's own init is byte-identical to model_solo's until asset B's init draws more randomness
X_pair = torch.cat([X_solo, torch.randn(n_samples, lookback, n_channels)], dim=-1)  # asset A (same signal) + asset B (unrelated pure noise), independent (no cross_pairs)
z_pair = torch.zeros(n_samples, lookback, 2)
z_pair[:, -1, 0] = z_solo[:, -1, 0]
z_pair[:, -1, 1] = torch.randn(n_samples)  # asset B's own label - unrelated to asset A's
model_pair = pl.PredictionModel(n_assets=2, pairs=["A", "B"], n_channels=n_channels, hidden_size=8, dropout=0.0)
pl.train_prediction_model(model_pair, X_pair, z_pair, epochs=80, lr=1e-2, bce_weight=1.0)
model_pair.eval()
with torch.no_grad():
    mu_pair, _ = model_pair(X_pair)
hit_pair_a = float(((mu_pair[:, -1, 0] > 0) == (z_pair[:, -1, 0] > 0)).float().mean())

assert abs(hit_solo - hit_pair_a) < 0.1, (
    f"asset A's own hit rate changed when an unrelated asset B was added to the SAME model "
    f"({hit_solo:.3f} solo vs {hit_pair_a:.3f} alongside B) - training isn't truly independent"
)
print(f"1c. true per-asset independence OK: asset A hit rate {hit_solo:.3f} solo vs {hit_pair_a:.3f} alongside an unrelated asset B")

# --- 2. losses: finite, and BCE beats collapse on a signal-bearing toy ---
z_labels = torch.randn(n_samples, lookback, n_assets)
nll = pl.gaussian_nll(mu, sigma, z_labels)
bce = pl.direction_bce(mu, sigma, z_labels)
assert torch.isfinite(nll) and torch.isfinite(bce)
print(f"2. losses OK: nll {nll:.4f}, bce {bce:.4f}")

# --- 3. training runs and reduces loss; signal is learnable ---
# Plant a real signal: label z = first feature channel of decision day.
signal = X[:, -1, 0::n_channels]  # (n_samples, n_assets)
z_planted = z_labels.clone()
z_planted[:, -1, :] = signal + 0.3 * torch.randn(n_samples, n_assets)
pl.train_prediction_model(model, X, z_planted, epochs=150, lr=1e-2, bce_weight=1.0)
model.eval()
with torch.no_grad():
    mu2, sig2 = model(X)
mu2_decision = mu2[:, -1, :]
hit = float(((mu2_decision > 0) == (z_planted[:, -1, :] > 0)).float().mean())
assert hit > 0.65, f"model failed to learn planted signal (hit {hit:.3f})"
pos_frac = float((mu2_decision > 0).float().mean())
assert 0.2 < pos_frac < 0.8, f"mu collapsed to one sign (pos fraction {pos_frac:.3f})"
print(f"3. training OK: hit rate on planted signal {hit:.3f}, positive-call fraction {pos_frac:.3f}")

# --- 4. neutral band + confusion metrics ---
probs = np.array([[0.30, 0.52], [0.48, 0.90], [0.70, 0.50], [0.51, 0.49]])
labels = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
banded = pl.apply_neutral_band(probs, 0.05)
assert (banded[:, 0] == np.array([0.30, 0.5, 0.70, 0.5])).all()
m = pl.confusion_matrix_metrics(banded, labels, neutral_band=0.05)
# asset 0: decided rows 0 (pred neg, actual neg -> TN) and 2 (pred pos, actual pos -> TP)
assert m["tp"][0] == 1 and m["tn"][0] == 1 and m["fp"][0] == 0 and m["fn"][0] == 0
assert m["abstained"][0] == 2 and abs(m["coverage"][0] - 0.5) < 1e-6  # coverage's denominator has the same defensive +eps as accuracy/precision/etc below
assert abs(m["accuracy"][0] - 1.0) < 1e-6
# asset 1: rows 1 (pred pos 0.90, actual pos -> TP); rows 0,2,3 within band -> abstained
assert m["tp"][1] == 1 and m["abstained"][1] == 3
# band=0 reduces to classic counts over all samples
m0 = pl.confusion_matrix_metrics(probs, labels, neutral_band=0.0)
assert int(m0["tp"][0] + m0["fp"][0] + m0["tn"][0] + m0["fn"][0]) == 4
print("4. neutral band + confusion metrics OK")

# --- 5. decided hit rate ---
hr = pl._decided_hit_rate(banded, labels, 0.05)
assert abs(hr[0] - 1.0) < 1e-6 and abs(hr[1] - 1.0) < 1e-6
print("5. decided hit rate OK")

# --- 6. checkpoint round-trip incl. neutral_band and sigma_hat ---
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "m.pt")
    model.save_model(path, x_mean=np.zeros(n_assets * n_channels, dtype=np.float32),
                     x_std=np.ones(n_assets * n_channels, dtype=np.float32),
                     pairs=pairs, lookback=lookback,
                     features=["log_return", "vol", "skew", "kurt"], cma_windows=[],
                     sigma_hat=np.array([1.1, 0.9, 1.0], dtype=np.float32),
                     neutral_band=0.07, target_vol=0.15)
    loaded = pl.PredictionModel.load_model(path)
    assert loaded.neutral_band == 0.07
    assert loaded.target_vol == 0.15
    assert np.allclose(loaded.sigma_hat, [1.1, 0.9, 1.0])
    assert loaded.included_indices == model.included_indices
    with torch.no_grad():
        lm, ls = loaded(X[:4])
        om, osg = model(X[:4])
    assert torch.allclose(lm, om, atol=1e-6) and torch.allclose(ls, osg, atol=1e-6)
print("6. checkpoint round-trip OK")

# --- 7. bce_weight sweep: every value trained under every seed, best kept per seed then overall ---
import pandas as pd

sweep_pairs = ["A", "B"]
sweep_dates = pd.bdate_range("2020-01-01", periods=300)
sweep_returns = pd.DataFrame(np.random.randn(300, 2) * 0.005, index=sweep_dates, columns=sweep_pairs)
pl.load_close_prices = lambda symbols, years, cutoff_date=None: (1 + sweep_returns[symbols]).cumprod() * 1.1

import argparse
sweep_args = argparse.Namespace(**{
    **pl.DEFAULT_CONFIG,
    "pairs": sweep_pairs, "lookback": 15, "years": 3, "train_frac": 0.8, "test_frac": 0.1,
    "direction_horizon": 5, "rolling_stats_window": 20,
    "epochs": 4, "n_seeds": 2, "device": "cpu", "hidden_size": 4,
    "bce_weight": [0.5, 1.0, 2.0],
})
sweep_result = pl.run_pipeline_multi_seed(sweep_args)
assert sweep_result.hit_rate_val.shape == (len(sweep_pairs),)
print("7. bce_weight sweep OK (2 seeds x 3 values)")

# --- 8. butterworth_bandpass_features: causal band-pass filter, channel expansion ---
bp_dates = pd.bdate_range("2020-01-01", periods=400)
bp_returns = pd.DataFrame(np.random.randn(400, 2) * 0.005, index=bp_dates, columns=["A", "B"])

# 8a. channel-count expansion: "bandpass" contributes one channel PER
# (short, long) window pair, same as "cma".
n_ch = pl.n_channels_per_pair(["log_return", "bandpass"], bandpass_windows=[[10, 50], [20, 100]])
assert n_ch == 1 + 2, f"expected 1 base channel + 2 bandpass channels, got {n_ch}"
print("8a. n_channels_per_pair OK: one channel per bandpass window pair")

# 8b. causality: perturbing FUTURE returns must not change any PAST
# filtered value - the same guarantee cross_moving_averages/trailing_vol/
# rolling_moment_features already hold, and the whole reason lfilter (not
# filtfilt) is used.
bp_before = pl.butterworth_bandpass_features(bp_returns, [[10, 50]], order=3)
bp_returns_perturbed = bp_returns.copy()
bp_returns_perturbed.iloc[300:] += 5.0
bp_after = pl.butterworth_bandpass_features(bp_returns_perturbed, [[10, 50]], order=3)
before_series = bp_before[("A", 10, 50)].iloc[:300]
after_series = bp_after[("A", 10, 50)].iloc[:300]
assert np.allclose(before_series.to_numpy(), after_series.to_numpy(), equal_nan=True), (
    "a future perturbation changed a PAST bandpass filter output - lfilter causality is broken"
)
print("8b. butterworth_bandpass_features OK: causal (past outputs unaffected by a future perturbation)")

# 8c. reacts faster than an equivalent CMA crossover to a step change (the
# actual point of using a proper band-pass design instead of an SMA
# difference): both are pure trend-CHANGE detectors (a band-pass removes
# the DC/long-run level entirely, same as a CMA crossover eventually
# settling back to 0 once the step ages out of both windows), so "reacts
# faster" is measured as fewer days to reach ITS OWN peak magnitude after
# a sudden regime flip - a CMA crossover ramps linearly over ~short_window
# days as the short SMA fills with the new level; a proper Butterworth
# design gets there faster for the same passband.
step_returns = pd.DataFrame(np.zeros((300, 1)), columns=["A"])
step_returns.iloc[150:, 0] = 0.01  # sudden sustained regime shift
bp_step = pl.butterworth_bandpass_features(step_returns, [[10, 50]], order=3)[("A", 10, 50)]
cma_step = pl.cross_moving_averages(step_returns, [[10, 50]])[("A", 10, 50)]
bp_peak_day = int(bp_step.iloc[150:].abs().to_numpy().argmax())
cma_peak_day = int(cma_step.iloc[150:].abs().to_numpy().argmax())
assert bp_peak_day < cma_peak_day, (
    f"bandpass should reach its own peak magnitude faster than an equivalent CMA crossover "
    f"({bp_peak_day}d vs {cma_peak_day}d)"
)
print(f"8c. butterworth_bandpass_features OK: peaks {bp_peak_day}d after a regime shift vs CMA's {cma_peak_day}d")

# 8d. invalid windows are rejected (short >= long, or below the Nyquist floor).
try:
    pl.butterworth_bandpass_features(bp_returns, [[50, 10]])
    raise AssertionError("short_period >= long_period should have raised ValueError")
except ValueError:
    pass
try:
    pl.butterworth_bandpass_features(bp_returns, [[1, 50]])  # short_period=1 is at/above the Nyquist limit for daily data
    raise AssertionError("short_period at the Nyquist limit should have raised ValueError")
except ValueError:
    pass
print("8d. butterworth_bandpass_features OK: invalid windows rejected")

# --- 9. portfolio_pnl: risk parity, causal covariance, target-vol scaling ---
from models import portfolio_pnl as pp

# 9a. equal-risk-contribution: on a correlated 3-asset covariance matrix,
# weights must sum to 1 and every asset's risk contribution (w_i * (Cw)_i)
# must be equal - NOT just inverse-vol (which would ignore correlation).
cov3 = np.array([[0.04, 0.01, 0.02], [0.01, 0.09, 0.015], [0.02, 0.015, 0.16]])
w = pp.risk_parity_weights(cov3)
contrib = w * (cov3 @ w)
assert abs(w.sum() - 1.0) < 1e-6
assert np.allclose(contrib, contrib[0], rtol=1e-3), f"risk contributions not equalized: {contrib}"
assert (w > 0).all(), "risk parity should be long-only"
assert np.allclose(pp.risk_parity_weights(np.array([[0.05]])), [1.0]), "single-asset risk parity must be all-in"
print("9a. risk_parity_weights OK: sums to 1, equalizes risk contribution, single-asset trivial case")

# 9b. rolling_covariance_matrices is causal: perturbing FUTURE returns must
# not change any PAST covariance matrix.
rng = np.random.default_rng(0)
T, n = 200, 3
base = rng.normal(0, 0.01, size=(T, 1))
returns = base * rng.normal(1, 0.3, size=(1, n)) + rng.normal(0, 0.005, size=(T, n))
cov_before = pp.rolling_covariance_matrices(returns, window=60)
returns_perturbed = returns.copy()
returns_perturbed[100:] += 5.0
cov_after = pp.rolling_covariance_matrices(returns_perturbed, window=60)
assert np.allclose(cov_before[:100], cov_after[:100], equal_nan=True), "future perturbation leaked into a past covariance matrix"
print("9b. rolling_covariance_matrices OK: past matrices unaffected by a future perturbation")

# 9c. compute_portfolio: realized vol of the modulated book should land in
# the neighborhood of target_vol (it's an ex-ante scale, not exact - actual
# realized vol will drift, but shouldn't be wildly off), and the strategy
# actually shorts when probabilities lean negative.
probs = 0.5 + 0.15 * np.sign(rng.normal(size=(T, n)))
target_vol = 0.10
out = pp.compute_portfolio(probs, returns, direction_horizon=5, target_vol=target_vol, cov_window=60)
valid = ~np.isnan(out["positions_modulated"]).any(axis=1)
assert valid.sum() > T // 2, "too many NaN days in a 200-day series with a 60-day cov window"
realized_vol = float(np.std(out["pnl_modulated"][valid])) * np.sqrt(pp.TRADING_DAYS_PER_YEAR)
assert 0.02 < realized_vol < 0.5, f"realized vol {realized_vol:.3f} is nowhere near target_vol {target_vol}"
some_negative = (out["positions_modulated"][valid] < 0).any()
assert some_negative, "modulated positions should go short when a pair's probability signal is negative"
assert (out["positions_baseline"][valid] >= 0).all(), "unmodulated risk-parity baseline must stay long-only"
print(f"9c. compute_portfolio OK: realized annualized vol {realized_vol:.3f} near target {target_vol}, shorts when signal is negative")

# 9d. latest_position: today's probability should move the position in the
# right direction relative to the SAME history (the direction_horizon-day
# smoothing dilutes a single day's signal - see compute_portfolio's
# docstring - so this compares bullish vs. bearish today rather than
# asserting an absolute sign).
bullish_today = pp.latest_position(
    probs, returns, np.array([0.9, 0.9, 0.9], dtype=np.float32), direction_horizon=5, target_vol=target_vol, cov_window=60,
)
bearish_today = pp.latest_position(
    probs, returns, np.array([0.1, 0.1, 0.1], dtype=np.float32), direction_horizon=5, target_vol=target_vol, cov_window=60,
)
assert bullish_today["position_modulated"].shape == (n,)
assert (bullish_today["position_modulated"] > bearish_today["position_modulated"]).all(), (
    "a more bullish today's probability should size a larger (or less negative) position, same history"
)
print("9d. latest_position OK: today's probability moves the booked position in the right direction")

# --- 10. train_prediction_model's optional portfolio-Sharpe phase (sharpe_weight) ---

def _make_sharpe_data(n_assets_, seed, t=250):
    torch.manual_seed(seed)
    x = torch.randn(t, lookback, n_assets_ * n_channels)
    z = torch.randn(t, lookback, n_assets_)
    sig = x[:, -1, 0::n_channels]
    z[:, -1, :] = sig + 0.3 * torch.randn(t, n_assets_)
    ret = (0.001 * sig + 0.002 * torch.randn(t, n_assets_)).numpy().astype(np.float32)
    rp, cov_ = pp.precompute_risk_parity(ret, cov_window=60)
    return x, z, torch.tensor(ret), torch.tensor(rp, dtype=torch.float32), torch.tensor(cov_, dtype=torch.float32)

# 10a. sharpe_weight=0 (default) is a byte-identical no-op: calling
# train_prediction_model WITH the new kwargs (all disabled) vs WITHOUT
# them at all must produce identical trained weights - the regression
# guarantee every other test in this file already relies on implicitly.
X10, z10, ret10, rp10, cov10 = _make_sharpe_data(2, seed=7)
torch.manual_seed(3)
model_old_call = pl.PredictionModel(n_assets=2, pairs=["A", "B"], n_channels=n_channels, hidden_size=8, dropout=0.0)
pl.train_prediction_model(model_old_call, X10, z10, epochs=10, lr=1e-2, bce_weight=1.0)
torch.manual_seed(3)
model_new_call = pl.PredictionModel(n_assets=2, pairs=["A", "B"], n_channels=n_channels, hidden_size=8, dropout=0.0)
pl.train_prediction_model(
    model_new_call, X10, z10, epochs=10, lr=1e-2, bce_weight=1.0,
    sharpe_weight=0.0, sharpe_window=20, direction_horizon=5, target_vol=0.10,
    next_returns_train=ret10, rp_weights_train=rp10, cov_train=cov10,
)
with torch.no_grad():
    mu_old, _ = model_old_call(X10)
    mu_new, _ = model_new_call(X10)
assert torch.allclose(mu_old, mu_new, atol=1e-7), "sharpe_weight=0 must be a byte-identical no-op"
print("10a. sharpe_weight=0 OK: byte-identical no-op vs. the pre-existing call signature")

# 10b. sharpe_weight > 0 genuinely couples assets (the whole point of the
# joint vol-scaling design) - asset A's OWN training now depends on an
# unrelated asset B being present, the OPPOSITE of test 1c's guarantee
# (which only holds for the default sharpe_weight=0 path). Same solo-vs-
# paired harness as 1c, but with sharpe_weight=1.0.
torch.manual_seed(11)
Xa, za, reta, rpa, cova = _make_sharpe_data(1, seed=11)
model_solo = pl.PredictionModel(n_assets=1, pairs=["A"], n_channels=n_channels, hidden_size=8, dropout=0.0)
pl.train_prediction_model(
    model_solo, Xa, za, epochs=15, lr=1e-2, bce_weight=1.0,
    sharpe_weight=1.0, sharpe_window=20, direction_horizon=5, target_vol=0.10,
    next_returns_train=reta, rp_weights_train=rpa, cov_train=cova,
)
with torch.no_grad():
    mu_solo10, _ = model_solo(Xa)
hit_solo10 = float(((mu_solo10[:, -1, 0] > 0) == (za[:, -1, 0] > 0)).float().mean())

torch.manual_seed(11)
Xa2, za2, _, _, _ = _make_sharpe_data(1, seed=11)  # asset A alone, same seed -> identical init/data up to this point
torch.manual_seed(22)
x_b_noise = torch.randn(za.shape[0], lookback, n_channels)
X_pair10 = torch.cat([Xa2, x_b_noise], dim=-1)
z_pair10 = torch.zeros(za.shape[0], lookback, 2)
z_pair10[:, -1, 0] = za2[:, -1, 0]
z_pair10[:, -1, 1] = torch.randn(za.shape[0])
ret_pair10 = torch.cat([reta, 0.01 * torch.randn(za.shape[0], 1)], dim=-1).numpy().astype(np.float32)
rp_pair10, cov_pair10 = pp.precompute_risk_parity(ret_pair10, cov_window=60)

torch.manual_seed(11)
model_pair10 = pl.PredictionModel(n_assets=2, pairs=["A", "B"], n_channels=n_channels, hidden_size=8, dropout=0.0)
pl.train_prediction_model(
    model_pair10, X_pair10, z_pair10, epochs=15, lr=1e-2, bce_weight=1.0,
    sharpe_weight=1.0, sharpe_window=20, direction_horizon=5, target_vol=0.10,
    next_returns_train=torch.tensor(ret_pair10), rp_weights_train=torch.tensor(rp_pair10, dtype=torch.float32),
    cov_train=torch.tensor(cov_pair10, dtype=torch.float32),
)
with torch.no_grad():
    mu_pair10, _ = model_pair10(X_pair10)
hit_pair_a10 = float(((mu_pair10[:, -1, 0] > 0) == (z_pair10[:, -1, 0] > 0)).float().mean())

assert abs(hit_solo10 - hit_pair_a10) > 1e-6, (
    f"sharpe_weight > 0 should couple assets (solo {hit_solo10:.3f} vs paired {hit_pair_a10:.3f}) - "
    f"got identical results, the joint vol-scaling phase isn't actually reaching asset A's gradient"
)
print(f"10b. sharpe_weight>0 OK: asset A's training genuinely coupled to unrelated asset B ({hit_solo10:.3f} solo vs {hit_pair_a10:.3f} paired)")

# 10c. training with sharpe_weight > 0 actually improves the realized
# training Sharpe on a planted profitable signal - checks the objective's
# SIGN is right (maximizing, not accidentally minimizing) and that
# gradients genuinely flow end-to-end from the Sharpe loss to the model's
# own weights.
X10c, z10c, ret10c, rp10c, cov10c = _make_sharpe_data(2, seed=42, t=300)
model_sharpe = pl.PredictionModel(n_assets=2, pairs=["A", "B"], n_channels=n_channels, hidden_size=8, dropout=0.0)
with torch.no_grad():
    sharpe_before = -pl._portfolio_sharpe_loss(model_sharpe, X10c, ret10c, rp10c, cov10c, 5, 20, 0.10).item()
pl.train_prediction_model(
    model_sharpe, X10c, z10c, epochs=40, lr=1e-2, bce_weight=1.0,
    sharpe_weight=2.0, sharpe_window=20, direction_horizon=5, target_vol=0.10,
    next_returns_train=ret10c, rp_weights_train=rp10c, cov_train=cov10c,
)
with torch.no_grad():
    sharpe_after = -pl._portfolio_sharpe_loss(model_sharpe, X10c, ret10c, rp10c, cov10c, 5, 20, 0.10).item()
assert sharpe_after > sharpe_before, (
    f"training sharpe should improve with sharpe_weight > 0 (before {sharpe_before:.3f}, after {sharpe_after:.3f})"
)
print(f"10c. sharpe_weight>0 OK: training sharpe improved {sharpe_before:.3f} -> {sharpe_after:.3f}")

# 10d. NaN in the precomputed rp_weights/cov (early warm-up days, see
# rolling_covariance_matrices) must not poison the loss/gradients with NaN
# (see _portfolio_sharpe_loss's own docstring on why nan_to_num happens
# BEFORE arithmetic, not after: 0 * NaN is still NaN in IEEE754).
X10d, z10d, ret10d, rp10d, cov10d = _make_sharpe_data(2, seed=5, t=80)  # short series -> real NaN warm-up region
assert torch.isnan(rp10d[:19]).any(), "test setup assumption: short series should have a real NaN warm-up region"
model_nan = pl.PredictionModel(n_assets=2, pairs=["A", "B"], n_channels=n_channels, hidden_size=8, dropout=0.0)
loss10d = pl._portfolio_sharpe_loss(model_nan, X10d, ret10d, rp10d, cov10d, 5, 20, 0.10)
assert torch.isfinite(loss10d), f"NaN warm-up rows leaked into the Sharpe loss: {loss10d}"
loss10d.backward()
for p in model_nan.parameters():
    assert p.grad is None or torch.isfinite(p.grad).all(), "NaN warm-up rows leaked into gradients"
print("10d. _portfolio_sharpe_loss OK: NaN warm-up rows don't poison the loss or gradients")

# --- 11. _assert_finite_grad: the immediate-failure safety net ---
# Real-world context: a run with num_layers=2, hidden_size=128,
# device="mps" produced a FINITE loss but a NaN gradient on its very
# first backward() - a confirmed PyTorch MPS multi-layer-LSTM
# backward-kernel bug (the identical forward/backward pass on
# device="cpu" is fine - see train_prediction_model's own docstring/
# _assert_finite_grad). That bug is environment-specific and can't be
# reproduced portably here, so this test instead verifies the SAFETY NET
# itself: a manufactured NaN gradient must be caught immediately with a
# clear error, not silently passed to optimizer.step().
tiny_model = pl.PredictionModel(n_assets=1, pairs=["A"], n_channels=n_channels, hidden_size=4)
some_param = next(tiny_model.assets[0].parameters())
some_param.grad = torch.full_like(some_param, float("nan"))
try:
    pl._assert_finite_grad(tiny_model.assets[0].parameters(), "test context")
    raise AssertionError("_assert_finite_grad should have raised on a NaN gradient")
except RuntimeError as exc:
    assert "test context" in str(exc), f"error message should include the context: {exc}"
some_param.grad = torch.zeros_like(some_param)  # finite grad must NOT raise
pl._assert_finite_grad(tiny_model.assets[0].parameters(), "test context")
print("11. _assert_finite_grad OK: raises immediately on a NaN gradient, silent on a finite one")

# --- 12. cutoff_date: never fetch/return a day after it ---
from datetime import date as _date

# 12a. _resolve_cutoff_date: None/far-future both collapse to today; a
# genuine past date passes through unchanged.
assert pl._resolve_cutoff_date(None) == _date.today()
assert pl._resolve_cutoff_date("9999-01-01") == _date.today()
past_cutoff = _date.today() - pd.Timedelta(days=30)
assert pl._resolve_cutoff_date(past_cutoff.isoformat()) == past_cutoff
assert pl._resolve_cutoff_date(past_cutoff) == past_cutoff
print("12a. _resolve_cutoff_date OK: None/future -> today, past date passes through unchanged")

# 12b. load_close_prices must never ask db.py for (or return) a day past
# cutoff_date - patch get_time_series to record its own `end` argument and
# to return data that DELIBERATELY extends past the requested cutoff (as
# if Postgres already had fresher rows from e.g. /api/quotes/refresh), and
# confirm both the query bound AND the returned frame respect the cutoff.
cutoff_dates = pd.bdate_range("2020-01-01", periods=250)
cutoff_wide = pd.DataFrame(1.1 + np.random.randn(250, 2) * 0.01, index=cutoff_dates, columns=["A", "B"])
captured_end = {}

def _fake_get_time_series(symbols, start, end, source="yahoo", field="close"):
    # Mirrors db.py's own real SQL `WHERE as_of_date BETWEEN start AND
    # end`, deliberately fed a frame that extends PAST the requested
    # cutoff (as if Postgres already had fresher rows from e.g.
    # /api/quotes/refresh) - so this only passes if load_close_prices
    # actually asks for (and therefore only ever receives) rows up to
    # cutoff_date, not "whatever happens to be in the table".
    captured_end["end"] = end
    mask = (cutoff_wide.index >= pd.Timestamp(start)) & (cutoff_wide.index <= pd.Timestamp(end))
    return cutoff_wide.loc[mask, list(symbols)]

_real_get_time_series = pl.get_time_series
pl.get_time_series = _fake_get_time_series
try:
    cutoff = _date(2020, 6, 1)
    prices_cut = _real_load_close_prices(["A", "B"], years=3, cutoff_date=cutoff.isoformat())
    assert captured_end["end"] == cutoff, f"expected query end={cutoff}, got {captured_end['end']}"
    assert prices_cut.index.max().date() <= cutoff, "load_close_prices returned a row after cutoff_date"
    assert prices_cut.index.max().date() == cutoff_dates[cutoff_dates <= pd.Timestamp(cutoff)].max().date()
finally:
    pl.get_time_series = _real_get_time_series
print("12b. load_close_prices OK: cutoff_date bounds both the db.py query and the returned frame")

print("\nALL SMOKE TESTS PASSED")
