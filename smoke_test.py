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
# get_connection is only imported at api/server.py's own MODULE level (its
# list_models() endpoint body is the only actual caller, never exercised
# below) - stubbed just so `import api.server` itself succeeds (test 20).
db_stub.get_connection = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no db in test"))
fx_stub = types.ModuleType("data.fx_downloader")
fx_stub.FXDownloader = object
fx_stub.MAJOR_FX_PAIRS = {}
# prediction_model_name() (models/portfolio_lstm.py) lazily imports
# build_model_name UNCONDITIONALLY (even when save_db=False - it's only
# used to derive a deterministic LOCAL job name, see _run_training_job's
# own model_base_name) - stubbed with a simple deterministic name-builder,
# never actually hitting a DB (save_model_blob/load_model_blob deliberately
# left unstubbed - not needed below, since test 20 stops short of the real
# save_to_db call).
registry_stub = types.ModuleType("data.model_registry")
registry_stub.build_model_name = lambda kind, **kwargs: f"{kind}_" + "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
data_pkg = types.ModuleType("data"); data_pkg.__path__ = []
sys.modules["data"] = data_pkg
sys.modules["data.db"] = db_stub
sys.modules["data.fx_downloader"] = fx_stub
sys.modules["data.model_registry"] = registry_stub

import torch
from models import portfolio_lstm as pl
from models import risk_engine as re

# GLOBAL SAFETY NET, installed before any test runs: models/prediction_model.pt
# is a REAL, git-tracked file in this repo - PredictionModel.save_model()'s
# own default `path` param points straight at it (see its docstring), and
# several tests below (esp. 20/21, which exercise the real
# api/server.py._run_training_job end to end) call it without an explicit
# path. Redirecting that ONE literal default string to a throwaway tmpdir
# for this entire process means ANY call to save_model() anywhere below that
# forgets to pass a path - a new test added later, a code path inside
# _run_training_job neither of the per-test patches happens to wrap - can
# never overwrite the tracked file, rather than relying on every individual
# test to bracket itself correctly (tests 20/21 additionally redirect to
# their OWN named tmp path on top of this, since they need to read the
# saved file back afterward - this is just the last line of defense).
_SMOKE_TEST_SAVE_DIR = tempfile.mkdtemp()
_unsafe_default_save_model = pl.PredictionModel.save_model
def _safe_default_save_model(self, path="models/prediction_model.pt", **kwargs):
    if path == "models/prediction_model.pt":
        path = os.path.join(_SMOKE_TEST_SAVE_DIR, "unbracketed_save.pt")
    return _unsafe_default_save_model(self, path, **kwargs)
pl.PredictionModel.save_model = _safe_default_save_model

# save_ensemble_model is a STANDALONE function (not a PredictionModel
# method - an ensemble isn't one nn.Module), so it calls torch.save(...)
# directly and is NOT covered by the patch above - it needs its own,
# same-shaped redirect (caught the hard way once already: a test that
# called api/server.py's real use_kfold_cv path wrote straight to the
# tracked file before this existed).
_unsafe_save_ensemble_model = pl.save_ensemble_model
def _safe_save_ensemble_model(path="models/prediction_model.pt", *args, **kwargs):
    if path == "models/prediction_model.pt":
        path = os.path.join(_SMOKE_TEST_SAVE_DIR, "unbracketed_save.pt")
    return _unsafe_save_ensemble_model(path, *args, **kwargs)
pl.save_ensemble_model = _safe_save_ensemble_model

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

# 9e. compute_portfolio's signal_min/signal_max: default (-1, 1) must be
# BYTE-IDENTICAL to the original fixed (p - 0.5) * 2 map (no args at all);
# a narrowed per-asset range (e.g. long-only [0, 1] for one asset) must
# actually change what gets traded - never go short on that asset - while
# leaving an unbounded asset's own behavior untouched.
out_default_explicit = pp.compute_portfolio(
    probs, returns, direction_horizon=5, target_vol=target_vol, cov_window=60, signal_min=-1.0, signal_max=1.0,
)
assert np.allclose(out_default_explicit["positions_modulated"], out["positions_modulated"], equal_nan=True), (
    "explicit signal_min=-1/signal_max=1 must reproduce the no-args default exactly"
)
signal_min_9e = np.array([0.0] + [-1.0] * (n - 1), dtype=np.float64)  # asset 0 long-only, rest unbounded
signal_max_9e = np.array([1.0] * n, dtype=np.float64)
out_longonly = pp.compute_portfolio(
    probs, returns, direction_horizon=5, target_vol=target_vol, cov_window=60,
    signal_min=signal_min_9e, signal_max=signal_max_9e,
)
valid_9e = ~np.isnan(out_longonly["positions_modulated"]).any(axis=1)
assert (out_longonly["positions_modulated"][valid_9e, 0] >= 0).all(), "asset 0 was configured long-only (min=0) but went short"
# (Other assets' own positions DO shift too, even though their own signal
# map is untouched - _scale_to_target_vol scales the WHOLE portfolio by one
# shared factor derived from every asset's current weight, same joint
# coupling _portfolio_sharpe_loss_from_predictions's own docstring
# describes for the training-time counterpart - not a bug.)
print("9e. compute_portfolio OK: default signal_min/max reproduces the original map; a per-asset long-only range holds")

# 9f. compute_portfolio's neutral_band: 0.0 (default) is a no-op; a band
# wide enough to cover every probability here (`probs` is built as
# 0.5 +/- 0.15, so |p - 0.5| = 0.15 for every entry) forces every day's
# signal to exactly 1.0 - riding the unmodulated risk-parity weight -
# which must be IDENTICAL to passing signal_min=signal_max=1.0 with no
# band at all (that map also produces signal=1.0 unconditionally).
out_zero_band9f = pp.compute_portfolio(
    probs, returns, direction_horizon=5, target_vol=target_vol, cov_window=60, neutral_band=0.0,
)
assert np.allclose(out_zero_band9f["positions_modulated"], out["positions_modulated"], equal_nan=True), (
    "neutral_band=0.0 must be a byte-identical no-op"
)
out_full_band9f = pp.compute_portfolio(
    probs, returns, direction_horizon=5, target_vol=target_vol, cov_window=60, neutral_band=0.2,
)
out_all_ones9f = pp.compute_portfolio(
    probs, returns, direction_horizon=5, target_vol=target_vol, cov_window=60,
    signal_min=np.ones(n), signal_max=np.ones(n),
)
assert np.allclose(out_full_band9f["positions_modulated"], out_all_ones9f["positions_modulated"], equal_nan=True), (
    "a band covering every probability should force every day's signal to 1.0, matching an all-ones signal map"
)
assert not np.allclose(out_full_band9f["positions_modulated"], out["positions_modulated"], equal_nan=True), (
    "a full neutral band must actually change the modulated positions vs no band"
)
print("9f. compute_portfolio OK: neutral_band=0.0 is a no-op; a full band forces every day onto the unmodulated risk-parity weight")

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

# --- 13. checkpoint_metric ("val_loss"/"hit_rate"/"sharpe") + continue_training ---

# 13a. invalid checkpoint_metric is rejected loudly, not silently ignored.
try:
    pl.train_prediction_model(
        pl.PredictionModel(n_assets=2, pairs=["A", "B"], n_channels=n_channels, hidden_size=4, dropout=0.0),
        X10, z10, epochs=1, lr=1e-2, checkpoint_metric="bogus",
    )
    raise AssertionError("checkpoint_metric='bogus' should have raised")
except ValueError as exc:
    assert "checkpoint_metric" in str(exc)
print("13a. checkpoint_metric OK: invalid value rejected with a clear error")

# 13b. "hit_rate"/"sharpe" run cleanly end-to-end, INCLUDING validation-time
# Sharpe being computed even though sharpe_weight=0 (checkpoint_metric
# alone should be enough to trigger it - see track_sharpe_val), and
# actually change which epoch's weights get restored (not a no-op).
X13, z13, _, _, _ = _make_sharpe_data(2, seed=99, t=200)
Xv13, zv13, retv13, rpv13, covv13 = _make_sharpe_data(2, seed=100, t=60)
for metric in ("hit_rate", "sharpe"):
    torch.manual_seed(1)
    model13 = pl.PredictionModel(n_assets=2, pairs=["A", "B"], n_channels=n_channels, hidden_size=4, dropout=0.0)
    with torch.no_grad():
        mu_before, _ = model13(X13)
    pl.train_prediction_model(
        model13, X13, z13, epochs=8, lr=1e-2, bce_weight=1.0,
        X_val=Xv13, z_labels_val=zv13,
        next_returns_val=retv13, rp_weights_val=rpv13, cov_val=covv13,
        checkpoint_metric=metric,
    )
    with torch.no_grad():
        mu_after, _ = model13(X13)
    assert not torch.allclose(mu_before, mu_after), f"checkpoint_metric={metric!r} should have actually trained/restored something"
print("13b. checkpoint_metric OK: 'hit_rate'/'sharpe' run end-to-end (val Sharpe computed even with sharpe_weight=0)")

# 13c. continue_training: a genuine warm start, not a fresh random init -
# epochs=0 must reproduce the base model's own probabilities EXACTLY
# (nothing in the loop runs, so best_state stays empty and the warm-
# started weights are returned untouched).
continue_pairs = ["A", "B"]
continue_dates = pd.bdate_range("2020-01-01", periods=300)
continue_returns = pd.DataFrame(np.random.randn(300, 2) * 0.005, index=continue_dates, columns=continue_pairs)
pl.load_close_prices = lambda symbols, years, cutoff_date=None: (1 + continue_returns[symbols]).cumprod() * 1.1

base_args = argparse.Namespace(**{
    **pl.DEFAULT_CONFIG,
    "pairs": continue_pairs, "lookback": 15, "years": 3, "train_frac": 0.8, "test_frac": 0.1,
    "direction_horizon": 5, "rolling_stats_window": 20,
    "epochs": 5, "n_seeds": 1, "device": "cpu", "hidden_size": 4, "num_layers": 1,
})
base_result = pl.run_pipeline_multi_seed(base_args)

# continue_training is only ever called (in production - see
# api/server.py) with a model reloaded via load_prediction_model_auto,
# which reconstructs it through PredictionModel._from_checkpoint - THAT is
# what actually populates .x_mean/.x_std/.features/.cma_windows/etc as
# real attributes (a freshly in-memory-trained PredictionModel, straight
# out of run_pipeline_multi_seed, has none of them - they're checkpoint-
# only metadata). Save-then-reload here so this test exercises the exact
# same object shape continue_training will actually receive.
with tempfile.TemporaryDirectory() as ct_dir:
    base_path = os.path.join(ct_dir, "base.pt")
    base_result.model.save_model(
        base_path, x_mean=base_result.x_mean, x_std=base_result.x_std, pairs=base_result.pairs,
        lookback=base_result.lookback, features=base_args.features, cma_windows=base_args.cma_windows,
        sigma_hat=base_result.sigma_hat, neutral_band=base_result.neutral_band, target_vol=base_args.target_vol,
    )
    loaded_base_model = pl.PredictionModel.load_model(base_path)

    continue_args_noop = argparse.Namespace(**{**vars(base_args), "epochs": 0})
    result_noop = pl.continue_training(continue_args_noop, loaded_base_model)
    assert np.allclose(result_noop.probabilities_train, base_result.probabilities_train, atol=1e-6), (
        "continue_training with epochs=0 should reproduce the base model's own probabilities exactly - "
        "got different values, so it isn't actually warm-starting from the base model's weights"
    )
    print("13c. continue_training OK: epochs=0 exactly reproduces the base model (genuine warm start, not a fresh init)")

    # 13d. architecture is recovered from base_model, NEVER from args - even
    # a deliberately WRONG hidden_size/pairs in args must be ignored.
    continue_args_wrong_arch = argparse.Namespace(**{
        **vars(base_args), "epochs": 1, "hidden_size": 999, "pairs": ["Z"],
    })
    result_arch = pl.continue_training(continue_args_wrong_arch, loaded_base_model)
    assert result_arch.model.hidden_size == base_result.model.hidden_size == 4
    assert result_arch.pairs == base_result.pairs == continue_pairs
    print("13d. continue_training OK: architecture recovered from the base model, ignoring mismatched args")

# --- 14. on_best_checkpoint callback + summarize_checkpoint: "save best model so far" ---

# 14a. on_best_checkpoint fires with the RIGHT references (model/data/args
# match what actually trained; epoch increases; best_state is populated
# for every asset after at least one validated epoch); best_score is a
# finite number matching the running best displayed elsewhere.
captured_calls = []
def _on_best14(model, data, args, epoch, best_state, best_score):
    captured_calls.append({
        "model": model, "data": data, "args": args, "epoch": epoch, "best_state": best_state, "best_score": best_score,
    })

best_args14 = argparse.Namespace(**{
    **pl.DEFAULT_CONFIG,
    "pairs": continue_pairs, "lookback": 15, "years": 3, "train_frac": 0.8, "test_frac": 0.1,
    "direction_horizon": 5, "rolling_stats_window": 20,
    "epochs": 6, "n_seeds": 1, "device": "cpu", "hidden_size": 4, "num_layers": 1,
})
pl.run_pipeline_multi_seed(best_args14, on_best_checkpoint=_on_best14)
assert len(captured_calls) >= 1, "on_best_checkpoint should have fired at least once (every validated epoch)"
last = captured_calls[-1]
assert all(s is not None for s in last["best_state"]), "every asset should have a validated checkpoint by the end"
assert last["epoch"] == best_args14.epochs, f"last call should be from the final epoch ({best_args14.epochs}), got {last['epoch']}"
assert all(c["best_score"] is not None and np.isfinite(c["best_score"]) for c in captured_calls), (
    "best_score should be a finite number on every validated-epoch call, not just the sparse logging cadence"
)
print(f"14a. on_best_checkpoint OK: fired {len(captured_calls)} times, last at epoch {last['epoch']}, every asset validated, best_score always finite")

# 14b. summarize_checkpoint returns None if any asset lacks a validated checkpoint.
assert pl.summarize_checkpoint(last["model"], last["data"], last["args"], [None, {"dummy": 1}]) is None
print("14b. summarize_checkpoint OK: returns None when any asset has no validated checkpoint yet")

# 14c. Full snapshot from a REAL best_state: correct shape, finite loss,
# hit_rate in [0, 1], and a Sharpe (float or None, never NaN) for each split.
snapshot = pl.summarize_checkpoint(last["model"], last["data"], last["args"], last["best_state"])
assert snapshot is not None
snapshot_model, snapshot_result, summary = snapshot
assert isinstance(snapshot_model, pl.PredictionModel)
for split in ("train", "val", "test"):
    assert split in summary
    assert np.isfinite(summary[split]["loss"]), f"{split} loss should be finite, got {summary[split]['loss']}"
    assert 0.0 <= summary[split]["hit_rate"] <= 1.0
    assert summary[split]["sharpe"] is None or np.isfinite(summary[split]["sharpe"])
print("14c. summarize_checkpoint OK: valid snapshot + train/val/test summary (finite loss, hit_rate in [0,1], Sharpe finite-or-None)")

# --- 15. _non_overlapping_sharpe_torch: non-overlapping, backward-walking chunks, remainder DROPPED ---

# 15a. exact chunk boundaries: T=50, window=20 -> chunk 0 = pnl[30:50]
# (the LAST 20 days), chunk 1 = pnl[10:30] (the 20 days before that); the
# oldest 10 days (pnl[0:10]) are never included in ANY chunk.
torch.manual_seed(0)
pnl15 = torch.randn(50) * 0.01 + 0.001
sharpes15 = pl._non_overlapping_sharpe_torch(pnl15, window=20)
assert sharpes15.shape == (2,), f"expected 2 full 20-day chunks from 50 days, got {tuple(sharpes15.shape)}"
for i, (lo, hi) in enumerate([(30, 50), (10, 30)]):
    chunk = pnl15[lo:hi]
    expected = chunk.mean() / (chunk.std(unbiased=True) + 1e-8)
    assert torch.allclose(sharpes15[i], expected, atol=1e-6), (
        f"chunk {i} should be pnl[{lo}:{hi}] (walking backward from the last day) - got {sharpes15[i]:.6f}, "
        f"expected {expected:.6f}"
    )
print("15a. _non_overlapping_sharpe_torch OK: chunks walk backward from the last day, exact boundaries match a manual reference")

# 15b. a leftover remainder shorter than `window` is DROPPED entirely -
# not computed as its own (shorter) chunk, and not merged into a neighbor.
pnl15b = torch.randn(45) * 0.01  # 45 = 2*20 + 5 leftover
sharpes15b = pl._non_overlapping_sharpe_torch(pnl15b, window=20)
assert sharpes15b.shape == (2,), f"the 5-day remainder should be dropped, not counted as a 3rd chunk - got {tuple(sharpes15b.shape)}"
print("15b. _non_overlapping_sharpe_torch OK: a shorter leftover chunk is dropped entirely, never computed")

# 15c. fewer than one full window -> EMPTY result (not a shrunk window).
pnl15c = torch.randn(10) * 0.01  # < window=20
sharpes15c = pl._non_overlapping_sharpe_torch(pnl15c, window=20)
assert sharpes15c.numel() == 0, f"expected an empty result for T < window, got shape {tuple(sharpes15c.shape)}"
print("15c. _non_overlapping_sharpe_torch OK: T < window returns an empty result rather than shrinking the window")

# 15d. wired end-to-end: a split shorter than sharpe_window yields a
# NEUTRAL zero Sharpe loss (not NaN) from _portfolio_sharpe_loss, and
# training with sharpe_weight > 0 on such a short split still runs
# cleanly (the zero contributes nothing to the combined loss, rather than
# poisoning it - see _portfolio_sharpe_loss_from_predictions's own
# docstring on why this must never be NaN).
X15d, z15d, ret15d, rp15d, cov15d = _make_sharpe_data(2, seed=3, t=15)  # 15 < default sharpe_window=20
model15d = pl.PredictionModel(n_assets=2, pairs=["A", "B"], n_channels=n_channels, hidden_size=4, dropout=0.0)
loss15d = pl._portfolio_sharpe_loss(model15d, X15d, ret15d, rp15d, cov15d, direction_horizon=5, sharpe_window=20, target_vol=0.10)
assert torch.isfinite(loss15d) and float(loss15d) == 0.0, f"expected a neutral zero loss for T < sharpe_window, got {loss15d}"

model15d2 = pl.PredictionModel(n_assets=2, pairs=["A", "B"], n_channels=n_channels, hidden_size=4, dropout=0.0)
pl.train_prediction_model(
    model15d2, X15d, z15d, epochs=3, lr=1e-2, bce_weight=1.0,
    sharpe_weight=1.0, sharpe_window=20, direction_horizon=5, target_vol=0.10,
    next_returns_train=ret15d, rp_weights_train=rp15d, cov_train=cov15d,
)
with torch.no_grad():
    mu15d2, _ = model15d2(X15d)
assert torch.isfinite(mu15d2).all(), "training with a too-short split for sharpe_window should stay finite (neutral, not crashing)"
print("15d. _portfolio_sharpe_loss OK: a split shorter than sharpe_window yields a neutral zero loss and trains without crashing")

# --- 16. Per-asset signal_range (resolve_signal_bounds + checkpoint persistence) ---

# 16a. resolve_signal_bounds: an unlisted pair defaults to (-1, 1); a listed
# one uses its own configured bounds; order matches the `pairs` argument.
smin16, smax16 = pl.resolve_signal_bounds(["A", "B", "C"], {"B": [0.0, 1.0]})
assert np.allclose(smin16, [-1.0, 0.0, -1.0]) and np.allclose(smax16, [1.0, 1.0, 1.0]), (
    f"expected B alone narrowed to [0, 1], got min={smin16}, max={smax16}"
)
smin16_empty, smax16_empty = pl.resolve_signal_bounds(["A", "B"], None)
assert np.allclose(smin16_empty, [-1.0, -1.0]) and np.allclose(smax16_empty, [1.0, 1.0]), (
    "signal_range=None must resolve to (-1, 1) for every asset"
)
print("16a. resolve_signal_bounds OK: per-pair override applied, unlisted pairs default to (-1, 1)")

# 16b. _portfolio_sharpe_loss_from_predictions: explicit (-1, 1) tensors
# must reproduce the None-default loss exactly (same map, different code
# path), and narrowing one asset's range must change the loss (the signal
# actually gets clipped into the new range, not silently ignored). Reuses
# X10/ret10/rp10/cov10 (test 10's t=250 fixture) rather than the t=15
# fixtures above - those have NO valid (non-NaN) risk-parity/covariance
# rows at all (t=15 < rolling_covariance_matrices' min_periods=20), which
# would zero out the signal's effect entirely regardless of signal_min/max.
model16b = pl.PredictionModel(n_assets=2, pairs=["A", "B"], n_channels=n_channels, hidden_size=4, dropout=0.0)
mu16, sigma16 = model16b(X10)
mu16_dec, sigma16_dec = mu16[:, -1, :].detach(), sigma16[:, -1, :].detach()
loss_default16 = pl._portfolio_sharpe_loss_from_predictions(
    mu16_dec, sigma16_dec, ret10, rp10, cov10, direction_horizon=5, sharpe_window=20, target_vol=0.10,
)
symmetric_min16 = torch.full((2,), -1.0)
symmetric_max16 = torch.ones(2)
loss_explicit16 = pl._portfolio_sharpe_loss_from_predictions(
    mu16_dec, sigma16_dec, ret10, rp10, cov10, direction_horizon=5, sharpe_window=20, target_vol=0.10,
    signal_min=symmetric_min16, signal_max=symmetric_max16,
)
assert torch.allclose(loss_default16, loss_explicit16), "explicit (-1, 1) tensors must reproduce the None-default Sharpe loss"
narrowed_min16 = torch.tensor([0.0, -1.0])
loss_narrowed16 = pl._portfolio_sharpe_loss_from_predictions(
    mu16_dec, sigma16_dec, ret10, rp10, cov10, direction_horizon=5, sharpe_window=20, target_vol=0.10,
    signal_min=narrowed_min16, signal_max=symmetric_max16,
)
assert not torch.allclose(loss_default16, loss_narrowed16), "narrowing asset 0's range should change the joint Sharpe loss"
print("16b. _portfolio_sharpe_loss_from_predictions OK: default matches explicit (-1, 1); a narrowed range changes the loss")

# 16c. signal_range persists through save_model/load_model, defaulting to
# {} (i.e. every asset (-1, 1)) for a checkpoint that never set it.
with tempfile.TemporaryDirectory() as tmp16:
    path16 = os.path.join(tmp16, "signal_range.pt")
    model16 = pl.PredictionModel(n_assets=2, pairs=["A", "B"], n_channels=n_channels, hidden_size=4, dropout=0.0)
    model16.save_model(
        path16, x_mean=np.zeros(n_channels * 2, dtype=np.float32), x_std=np.ones(n_channels * 2, dtype=np.float32),
        pairs=["A", "B"], lookback=5, signal_range={"A": [0.0, 1.0]},
    )
    loaded16 = pl.load_prediction_model(path16)
    assert loaded16.signal_range == {"A": [0.0, 1.0]}, f"expected the saved signal_range to round-trip, got {loaded16.signal_range}"

    path16b = os.path.join(tmp16, "no_signal_range.pt")
    model16.save_model(
        path16b, x_mean=np.zeros(n_channels * 2, dtype=np.float32), x_std=np.ones(n_channels * 2, dtype=np.float32),
        pairs=["A", "B"], lookback=5,
    )
    loaded16b = pl.load_prediction_model(path16b)
    assert loaded16b.signal_range == {}, f"expected an unset signal_range to load back as {{}}, got {loaded16b.signal_range}"
print("16c. PredictionModel.signal_range OK: round-trips through save_model/load_model, defaults to {} when unset")

# --- 17. neutral_band inside the Sharpe training objective ---

# 17a. neutral_band=0.0 (default) is a byte-identical no-op - regression
# guarantee for every earlier Sharpe test in this file, which never pass
# neutral_band at all.
model17 = pl.PredictionModel(n_assets=2, pairs=["A", "B"], n_channels=n_channels, hidden_size=4, dropout=0.0)
mu17, sigma17 = model17(X10)
mu17_dec, sigma17_dec = mu17[:, -1, :].detach(), sigma17[:, -1, :].detach()
loss_no_band17 = pl._portfolio_sharpe_loss_from_predictions(
    mu17_dec, sigma17_dec, ret10, rp10, cov10, direction_horizon=5, sharpe_window=20, target_vol=0.10,
)
loss_zero_band17 = pl._portfolio_sharpe_loss_from_predictions(
    mu17_dec, sigma17_dec, ret10, rp10, cov10, direction_horizon=5, sharpe_window=20, target_vol=0.10,
    neutral_band=0.0,
)
assert torch.equal(loss_no_band17, loss_zero_band17), "neutral_band=0.0 must be a byte-identical no-op"
print("17a. _portfolio_sharpe_loss_from_predictions OK: neutral_band=0.0 is a byte-identical no-op")

# 17b. a band wide enough to cover every possible probability (probit's
# output is strictly inside (0, 1), so |p - 0.5| < 0.5 always) forces
# EVERY day's signal to exactly 1.0 - the book rides the unmodulated
# risk-parity weight the whole series - which must be IDENTICAL to simply
# passing signal_min=signal_max=1.0 with NO band at all (that map also
# produces signal=1.0 unconditionally, regardless of p, so both scenarios
# feed the exact same raw per-day signal series through the rest of the
# pipeline).
ones17 = torch.ones(2)
loss_all_ones17 = pl._portfolio_sharpe_loss_from_predictions(
    mu17_dec, sigma17_dec, ret10, rp10, cov10, direction_horizon=5, sharpe_window=20, target_vol=0.10,
    signal_min=ones17, signal_max=ones17,
)
loss_full_band17 = pl._portfolio_sharpe_loss_from_predictions(
    mu17_dec, sigma17_dec, ret10, rp10, cov10, direction_horizon=5, sharpe_window=20, target_vol=0.10,
    neutral_band=0.5,
)
assert torch.allclose(loss_all_ones17, loss_full_band17, atol=1e-6), (
    f"a band covering every probability should force every day's signal to 1.0, matching an all-ones signal map - "
    f"got {loss_full_band17} vs {loss_all_ones17}"
)
assert not torch.allclose(loss_no_band17, loss_full_band17), "a full neutral band must actually change the loss vs no band"
print("17b. _portfolio_sharpe_loss_from_predictions OK: a full band forces every day onto the unmodulated risk-parity weight")

# 17c. wired end-to-end: train_prediction_model with sharpe_weight>0 AND a
# neutral_band actually threads it into both the train and val Sharpe
# terms (checked by comparing a trained model's train_sharpe against one
# trained identically but with neutral_band=0 - not asserting a direction,
# just that the band is not silently ignored) and doesn't crash.
X17d, z17d, ret17d, rp17d, cov17d = _make_sharpe_data(2, seed=17, t=250)
torch.manual_seed(21)
model17d_band = pl.PredictionModel(n_assets=2, pairs=["A", "B"], n_channels=n_channels, hidden_size=8, dropout=0.0)
pl.train_prediction_model(
    model17d_band, X17d, z17d, epochs=5, lr=1e-2, bce_weight=1.0,
    sharpe_weight=1.0, sharpe_window=20, direction_horizon=5, target_vol=0.10,
    next_returns_train=ret17d, rp_weights_train=rp17d, cov_train=cov17d, neutral_band=0.3,
)
torch.manual_seed(21)
model17d_noband = pl.PredictionModel(n_assets=2, pairs=["A", "B"], n_channels=n_channels, hidden_size=8, dropout=0.0)
pl.train_prediction_model(
    model17d_noband, X17d, z17d, epochs=5, lr=1e-2, bce_weight=1.0,
    sharpe_weight=1.0, sharpe_window=20, direction_horizon=5, target_vol=0.10,
    next_returns_train=ret17d, rp_weights_train=rp17d, cov_train=cov17d, neutral_band=0.0,
)
weights_differ17d = any(
    not torch.equal(p1, p2)
    for p1, p2 in zip(model17d_band.state_dict().values(), model17d_noband.state_dict().values())
)
assert weights_differ17d, "training with vs. without neutral_band (same seed, same data) should produce different weights"
with torch.no_grad():
    mu17d, _ = model17d_band(X17d)
assert torch.isfinite(mu17d).all(), "training with sharpe_weight>0 and a neutral_band should stay finite"
print("17c. train_prediction_model OK: neutral_band is threaded into the Sharpe objective and changes what's learned")

# --- 18. direction_horizon/rolling_stats_window persistence + recovery ---

# 18a. Round-trip through save_model/load_model, defaulting to 5/20
# (DEFAULT_CONFIG's own values) for a checkpoint saved before these
# existed as persisted fields.
with tempfile.TemporaryDirectory() as tmp18:
    path18 = os.path.join(tmp18, "horizon.pt")
    model18 = pl.PredictionModel(n_assets=2, pairs=["A", "B"], n_channels=n_channels, hidden_size=4, dropout=0.0)
    model18.save_model(
        path18, x_mean=np.zeros(n_channels * 2, dtype=np.float32), x_std=np.ones(n_channels * 2, dtype=np.float32),
        pairs=["A", "B"], lookback=5, direction_horizon=7, rolling_stats_window=15,
    )
    loaded18 = pl.load_prediction_model(path18)
    assert loaded18.direction_horizon == 7 and loaded18.rolling_stats_window == 15, (
        f"expected the saved direction_horizon/rolling_stats_window to round-trip, got "
        f"{loaded18.direction_horizon}/{loaded18.rolling_stats_window}"
    )

    path18b = os.path.join(tmp18, "no_horizon.pt")
    model18.save_model(
        path18b, x_mean=np.zeros(n_channels * 2, dtype=np.float32), x_std=np.ones(n_channels * 2, dtype=np.float32),
        pairs=["A", "B"], lookback=5,
    )
    loaded18b = pl.load_prediction_model(path18b)
    assert loaded18b.direction_horizon == 5 and loaded18b.rolling_stats_window == 20, (
        f"expected an unset direction_horizon/rolling_stats_window to default to 5/20, got "
        f"{loaded18b.direction_horizon}/{loaded18b.rolling_stats_window}"
    )
print("18a. PredictionModel.direction_horizon/rolling_stats_window OK: round-trip through save_model/load_model, default to 5/20 when unset")

# 18b. load_pipeline recovers direction_horizon/rolling_stats_window from
# the LOADED MODEL's own checkpoint, not from whatever `args` happens to
# specify - a mismatch would silently compare the model's forecast against
# the wrong realized outcome (direction_horizon) or feed it feature values
# it never trained on (rolling_stats_window).
with tempfile.TemporaryDirectory() as tmp18c:
    path18c = os.path.join(tmp18c, "load_pipeline_horizon.pt")
    torch.manual_seed(55)
    model18c = pl.PredictionModel(n_assets=2, pairs=sweep_pairs, n_channels=n_channels, hidden_size=4, dropout=0.0)
    x_mean_18c = np.zeros(n_channels * 2, dtype=np.float32)
    x_std_18c = np.ones(n_channels * 2, dtype=np.float32)
    model18c.save_model(
        path18c, x_mean=x_mean_18c, x_std=x_std_18c,
        pairs=sweep_pairs, lookback=15, direction_horizon=7, rolling_stats_window=15,
        # Explicit - must match model18c's own n_channels=4 construction
        # above (this file's shared synthetic-architecture constant); left
        # unset, save_model would fall back to DEFAULT_FEATURES, which
        # doesn't necessarily produce 4 channels/pair.
        features=["log_return", "vol", "skew", "kurt"],
    )
    # `load_args18c` deliberately specifies the WRONG (mismatched)
    # direction_horizon/rolling_stats_window - load_pipeline must ignore
    # them entirely and use the model's own persisted 7/15 instead.
    load_args18c = argparse.Namespace(**{
        **pl.DEFAULT_CONFIG, "pairs": sweep_pairs, "lookback": None, "years": 3,
        # Explicit (not DEFAULT_CONFIG's own default) - must match model18c's
        # own n_channels=4 (this file's shared synthetic-architecture
        # constant, set at the top - see its own comment), independent of
        # whatever models.portfolio_lstm.DEFAULT_FEATURES currently is.
        "features": ["log_return", "vol", "skew", "kurt"],
        "direction_horizon": 1, "rolling_stats_window": 2,
        "train_frac": 0.8, "test_frac": 0.1, "device": "cpu", "load_model": path18c,
    })
    result18c = pl.load_pipeline(load_args18c)
    # Cross-check against _prepare_data called directly with the SAME
    # (correct) overrides - if load_pipeline is genuinely using the
    # model's own persisted values, the resulting split size must match
    # exactly; if it were still reading args' mismatched 1/2, it would
    # match `data18c_wrong` below instead.
    data18c_expected = pl._prepare_data(
        load_args18c, x_mean=x_mean_18c, x_std=x_std_18c, pairs=sweep_pairs, lookback=15,
        direction_horizon=7, rolling_stats_window=15,
    )
    data18c_wrong = pl._prepare_data(
        load_args18c, x_mean=x_mean_18c, x_std=x_std_18c, pairs=sweep_pairs, lookback=15,
        direction_horizon=1, rolling_stats_window=2,
    )
    assert len(data18c_expected.dates_train) != len(data18c_wrong.dates_train), (
        "test setup issue: direction_horizon 7/15 vs 1/2 produced the SAME split size - the test itself would be "
        "meaningless if these coincided"
    )
    assert len(result18c.dates_train) == len(data18c_expected.dates_train), (
        f"load_pipeline did not use the model's own persisted direction_horizon/rolling_stats_window - got "
        f"{len(result18c.dates_train)} train dates, expected {len(data18c_expected.dates_train)} "
        f"(args' mismatched values would have given {len(data18c_wrong.dates_train)})"
    )
print("18b. load_pipeline OK: recovers direction_horizon/rolling_stats_window from the model's own checkpoint, not from args")

# --- 19. models/risk_engine.py: a second-stage risk-attenuation overlay ---
re.load_close_prices = pl.load_close_prices  # same continue_returns-backed fake as test 18's own patch

# 19a. n_risk_channels_per_asset: weight + skew + kurt always, + one
# channel per (short, long) window pair in risk_cma_windows/risk_bandpass_windows.
assert re.n_risk_channels_per_asset([], []) == 3
assert re.n_risk_channels_per_asset([[10, 50]], [[10, 50]]) == 5
print("19a. n_risk_channels_per_asset OK")

# 19b. build_risk_stats_dataframe: correct shape, no NaN left over (ffill +
# fillna(0.0) warm-up handling, same as build_feature_dataframe's own).
stats19 = re.build_risk_stats_dataframe(continue_returns, continue_pairs, 20, [[10, 50]], [], 3)
assert stats19.shape == (len(continue_returns), 2 * 3), stats19.shape
assert not stats19.isna().any().any()
print("19b. build_risk_stats_dataframe OK, shape", stats19.shape)

# 19c. RiskEngine forward shape + attenuation_from_raw bounds.
engine19 = re.RiskEngine(n_assets=2, risk_channels_per_asset=3, hidden_size=4, num_layers=1, dropout=0.0, n_attn_heads=2)
x19 = torch.randn(5, 10, 2 * 3)
raw19 = engine19(x19)
assert raw19.shape == (5, 10, 2)
att19 = re.attenuation_from_raw(raw19, 0.0, 1.0)
assert att19.min() >= 0.0 and att19.max() <= 1.0
att19b = re.attenuation_from_raw(raw19, -0.1, 1.0)
assert att19b.min() >= -0.1 and att19b.max() <= 1.0
print("19c. RiskEngine.forward/attenuation_from_raw OK: shapes and bounds hold for both (0, 1) and (-0.1, 1) ranges")

# 19d. _non_overlapping_sortino_torch: shape + a regression guard for the
# sqrt-at-zero gradient blowup this function originally had (a chunk with
# NO down days has downside deviation exactly 0 pre-fix, and d/du sqrt(u)
# at u=0 is infinite - see this function's own docstring on the `+ eps`
# INSIDE the sqrt).
pnl19 = torch.tensor([0.01, 0.02, -0.01, 0.03, -0.02, 0.01, 0.02, -0.01, 0.01, 0.0], requires_grad=True)
sortinos19 = re._non_overlapping_sortino_torch(pnl19, window=5)
assert sortinos19.shape == (2,)
sortinos19.sum().backward()
assert torch.isfinite(pnl19.grad).all(), "non-overlapping Sortino produced a non-finite gradient"
pnl19_allpos = torch.tensor([0.01] * 10, requires_grad=True)  # a chunk with ZERO down days - the exact failure mode
re._non_overlapping_sortino_torch(pnl19_allpos, window=5).sum().backward()
assert torch.isfinite(pnl19_allpos.grad).all(), "a zero-downside chunk produced a non-finite (sqrt-at-zero) gradient"
print("19d. _non_overlapping_sortino_torch OK: finite gradients even for a chunk with zero down days")

# 19e. train_risk_engine end to end: runs without crashing, restores its
# own best-validation checkpoint, reports finite train/val Sortino.
torch.manual_seed(21)
base_args19 = argparse.Namespace(**{
    **pl.DEFAULT_CONFIG, "pairs": continue_pairs, "lookback": 15, "years": 3, "train_frac": 0.8, "test_frac": 0.1,
    "epochs": 3, "n_seeds": 1, "device": "cpu", "hidden_size": 4,
})
base_result19 = pl.run_pipeline_multi_seed(base_args19)
with tempfile.TemporaryDirectory() as tmp19:
    base_path19 = os.path.join(tmp19, "base19.pt")
    base_result19.model.save_model(
        base_path19, x_mean=base_result19.x_mean, x_std=base_result19.x_std, pairs=base_result19.pairs,
        lookback=base_result19.lookback, features=base_args19.features, cma_windows=[],
        sigma_hat=base_result19.sigma_hat, neutral_band=0.0, target_vol=pl.DEFAULT_TARGET_VOL,
        direction_horizon=5, rolling_stats_window=20,
    )
    base_model19 = pl.load_prediction_model(base_path19)

    risk_args19 = argparse.Namespace(**{
        **pl.DEFAULT_CONFIG, "pairs": continue_pairs, "years": 3, "train_frac": 0.8, "test_frac": 0.1, "device": "cpu",
        "risk_lookback": 10, "risk_cma_windows": [[10, 50]], "risk_bandpass_windows": [], "risk_bandpass_order": 3,
        "min_risk_att": 0.0, "max_risk_att": 1.0, "risk_hidden_size": 4, "risk_num_layers": 1, "risk_dropout": 0.0,
        "risk_n_attn_heads": 2, "risk_epochs": 3, "risk_lr": 1e-2, "risk_weight_decay": 0.0,
        "risk_sortino_window": 8, "full_exposure_penalty": 0.05,
    })
    risk_engine19, summary19 = re.train_risk_engine(base_model19, risk_args19)
    assert summary19["train_sortino"] is None or np.isfinite(summary19["train_sortino"])
    assert summary19["val_sortino"] is None or np.isfinite(summary19["val_sortino"])
    assert risk_engine19.pairs == continue_pairs
    assert risk_engine19.risk_lookback == 10
    print(f"19e. train_risk_engine OK: train_sortino={summary19['train_sortino']}, val_sortino={summary19['val_sortino']}")

    # 19f. checkpoint round-trip: risk_engine_checkpoint_dict/risk_engine_from_checkpoint.
    ckpt19 = re.risk_engine_checkpoint_dict(
        risk_engine19, continue_pairs, risk_engine19.risk_lookback, risk_engine19.risk_cma_windows,
        risk_engine19.risk_bandpass_windows, risk_engine19.risk_bandpass_order, risk_engine19.rolling_stats_window,
        risk_engine19.min_risk_att, risk_engine19.max_risk_att,
    )
    loaded_engine19 = re.risk_engine_from_checkpoint(ckpt19)
    assert loaded_engine19.pairs == continue_pairs
    assert loaded_engine19.risk_lookback == 10
    assert loaded_engine19.rolling_stats_window == 20
    with torch.no_grad():
        x19f = torch.randn(3, 10, risk_engine19.n_assets * risk_engine19.risk_channels_per_asset)
        assert torch.allclose(risk_engine19(x19f), loaded_engine19(x19f))
    print("19f. risk_engine_checkpoint_dict/risk_engine_from_checkpoint OK: round-trips weights and every hyperparameter")

    # 19g. PredictionModel bundles the risk engine into its OWN checkpoint -
    # save_model/load_model round-trip; a model with none loads with
    # model.risk_engine is None (backward compatible with every checkpoint
    # saved before this existed).
    bundled_path19 = os.path.join(tmp19, "bundled19.pt")
    base_model19.save_model(
        bundled_path19, x_mean=base_model19.x_mean, x_std=base_model19.x_std, pairs=base_model19.pairs,
        lookback=base_model19.lookback, features=base_model19.features, cma_windows=base_model19.cma_windows,
        sigma_hat=base_model19.sigma_hat, neutral_band=base_model19.neutral_band, target_vol=base_model19.target_vol,
        direction_horizon=base_model19.direction_horizon, rolling_stats_window=base_model19.rolling_stats_window,
        risk_engine=risk_engine19,
    )
    bundled_model19 = pl.load_prediction_model(bundled_path19)
    assert bundled_model19.risk_engine is not None
    assert bundled_model19.risk_engine.risk_lookback == 10
    with torch.no_grad():
        assert torch.allclose(bundled_model19.risk_engine(x19f), risk_engine19(x19f))
    # base_model19 (saved WITHOUT risk_engine=... above) has none - save_model/
    # save_to_db deliberately never auto-inherit self.risk_engine (see their
    # own docstrings).
    reloaded_base19 = pl.load_prediction_model(base_path19)
    assert reloaded_base19.risk_engine is None
    print("19g. PredictionModel risk_engine bundling OK: round-trips via save_model/load_model, absent stays None")

    # 19g2. continue_training on a risk-engine-bundled model must not
    # crash, and the resaved checkpoint must load cleanly afterward -
    # regression test for a bug where `base_model.risk_engine`, once set
    # by _from_checkpoint (a real registered nn.Module submodule, since
    # RiskEngine extends nn.Module), made base_model.state_dict() silently
    # include "risk_engine.*" keys - both inside continue_training's own
    # internal warm-start copy (model.load_state_dict(base_model.state_dict())
    # onto a FRESH model with no risk_engine submodule yet - an immediate
    # crash) and inside _checkpoint_dict's own self.state_dict() at resave
    # time (no crash there, but silently produced a checkpoint who's OWN
    # state_dict carried stale risk_engine weights forward while its
    # top-level "risk_engine" key stayed None - permanently unloadable
    # afterward, since a freshly-constructed model has no risk_engine
    # submodule at the point _from_checkpoint calls load_state_dict).
    continue_on_risk_args19 = argparse.Namespace(**{
        **pl.DEFAULT_CONFIG, "years": 3, "cutoff_date": None, "train_frac": 0.8, "test_frac": 0.1,
        "epochs": 2, "lr": 1e-3, "weight_decay": 0.0, "bce_weight": 1.0, "sharpe_weight": 0.0,
        "sharpe_window": 10, "direction_horizon": 5, "checkpoint_metric": "val_loss",
        "target_vol": pl.DEFAULT_TARGET_VOL, "neutral_band": 0.0, "signal_range": {}, "device": "cpu",
    })
    continued_result19 = pl.continue_training(continue_on_risk_args19, bundled_model19)
    resaved_path19 = os.path.join(tmp19, "continued_from_bundled19.pt")
    continued_result19.model.save_model(
        resaved_path19, x_mean=continued_result19.x_mean, x_std=continued_result19.x_std,
        pairs=continued_result19.pairs, lookback=continued_result19.lookback,
        features=bundled_model19.features, cma_windows=bundled_model19.cma_windows,
        sigma_hat=continued_result19.sigma_hat, neutral_band=continued_result19.neutral_band,
        target_vol=continue_on_risk_args19.target_vol,
        direction_horizon=continue_on_risk_args19.direction_horizon,
        rolling_stats_window=bundled_model19.rolling_stats_window,
    )
    raw_checkpoint19 = torch.load(resaved_path19, map_location="cpu", weights_only=True)
    polluted_keys19 = [k for k in raw_checkpoint19["state_dict"] if k.startswith("risk_engine.")]
    assert not polluted_keys19, f"THE BUG IS BACK: state_dict still carries stale risk_engine keys: {polluted_keys19}"
    # The ultimate regression check - THIS exact call is what used to fail
    # for a user continuing training a second time on such a model.
    reloaded_continued19 = pl.load_prediction_model(resaved_path19)
    assert reloaded_continued19.risk_engine is None  # never passed forward - by design, see save_model's docstring
    print("19g2. continue_training on a risk-engine-bundled model OK: no crash, resaved checkpoint reloads cleanly")

    # 19h. evaluate_risk_engine: dense (T, n_assets) attenuation, warm-up
    # rows default to max_risk_att (no attenuation), everything within bounds.
    positions19 = np.random.default_rng(7).normal(size=(200, 2)).astype(np.float32) * 0.05
    dates19 = continue_dates[:200]
    attenuation19 = re.evaluate_risk_engine(loaded_engine19, positions19, continue_returns, dates19)
    assert attenuation19.shape == (200, 2)
    assert (attenuation19[: loaded_engine19.risk_lookback - 1] == loaded_engine19.max_risk_att).all()
    assert attenuation19.min() >= loaded_engine19.min_risk_att - 1e-6
    assert attenuation19.max() <= loaded_engine19.max_risk_att + 1e-6
    print("19h. evaluate_risk_engine OK: warm-up rows default to max_risk_att, output stays within bounds")

# 19i. compute_portfolio's optional `attenuation` param: None (default)
# leaves the dict shape completely unchanged (no risk_attenuated keys at
# all, not even NaN-filled); a given attenuation multiplies
# positions_modulated EXACTLY (applied AFTER target-vol scaling, per this
# module's own docstring).
from models import portfolio_pnl as pp19
rng19 = np.random.default_rng(11)
probs19 = 0.5 + 0.2 * np.sign(rng19.normal(size=(200, 2)))
returns19 = rng19.normal(scale=0.005, size=(200, 2))
out19_no_att = pp19.compute_portfolio(probs19, returns19, direction_horizon=5, target_vol=0.1)
assert "positions_risk_attenuated" not in out19_no_att
att19_const = np.full((200, 2), 0.4, dtype=np.float32)
out19_att = pp19.compute_portfolio(probs19, returns19, direction_horizon=5, target_vol=0.1, attenuation=att19_const)
valid19 = ~np.isnan(out19_att["positions_modulated"]).any(axis=1)
ratio19 = out19_att["positions_risk_attenuated"][valid19] / out19_att["positions_modulated"][valid19]
assert np.allclose(ratio19, 0.4, atol=1e-6)
print("19i. compute_portfolio OK: attenuation=None leaves the dict shape unchanged; a given attenuation scales positions_modulated exactly")

# --- 20. api/server.py's "save best model so far" regression: the
# DISPLAYED best-so-far score must always match whatever would actually
# get saved - even across MULTIPLE seed restarts where a LATER restart
# starts out worse than an EARLIER one's own peak (see
# api/server.py's _on_best_checkpoint docstring for the exact bug this
# guards against: it used to overwrite _BEST_CHECKPOINT_STATE
# unconditionally on every validated epoch of EVERY restart, so a worse
# LATER restart could silently replace a better EARLIER one).
#
# Imports api.server here (nowhere else in this file does) - safe despite
# this file's own "no DB, no network" design: FastAPI/pydantic themselves
# touch neither, and the only DB-touching call (save_to_db, inside
# save_best_checkpoint) is deliberately NOT exercised below - this test
# stops at summarize_checkpoint (also DB-free), which is everything
# save_best_checkpoint does except the actual upload.
import api.server as srv
srv.load_close_prices = pl.load_close_prices  # reuse whatever fake is currently patched onto pl

job_id20 = "smoke-test-job-20"
srv._JOBS[job_id20] = {
    "status": "pending", "result": None, "error": None, "logs": [],
    "progress": {"seed_index": 1, "n_seeds": 3, "lambda_index": 1, "n_lambdas": 1, "epoch": 0, "total_epochs": 6, "percent": 0.0},
    "interim": None, "interim_history": [], "stop_requested": False,
}
config20 = {
    **pl.DEFAULT_CONFIG,
    "pairs": continue_pairs, "lookback": 15, "years": 3, "train_frac": 0.8, "test_frac": 0.1,
    "features": ["log_return", "vol"], "cma_windows": [], "bandpass_windows": [],
    "epochs": 6, "n_seeds": 3, "device": "cpu", "hidden_size": 4,
    "checkpoint_metric": "sharpe", "sharpe_weight": 0.0, "sharpe_window": 10,
    "save_db": False, "model_description": "",
}
# _run_training_job's own final save_model() call always writes to its
# DEFAULT local path ("models/prediction_model.pt" - the SAME real,
# git-tracked file every actual training run also writes to) since it
# never passes a `path` - redirect that ONE call to a tmpdir for the
# duration of this test so it can't clobber a real file on disk, then
# restore the real method immediately after.
_real_save_model20 = pl.PredictionModel.save_model
_tmpdir20 = tempfile.mkdtemp()
def _redirected_save_model20(self, path="models/prediction_model.pt", **kwargs):
    return _real_save_model20(self, os.path.join(_tmpdir20, "job20.pt"), **kwargs)
pl.PredictionModel.save_model = _redirected_save_model20
try:
    srv._run_training_job(job_id20, config20)
finally:
    pl.PredictionModel.save_model = _real_save_model20
job20 = srv._JOBS[job_id20]
assert job20["status"] == "done", job20.get("error")
stored20 = srv._BEST_CHECKPOINT_STATE.get(job_id20)
assert stored20 is not None, "expected at least one on_best_checkpoint call across 3 restarts x 6 epochs"
assert abs(stored20["score"] - job20["best_score_overall"]) < 1e-9, (
    f"THE BUG IS BACK: the DISPLAYED best-so-far ({job20['best_score_overall']}) no longer matches what's "
    f"actually stored to be saved ({stored20['score']}) - see api/server.py's _on_best_checkpoint"
)

# summarize_checkpoint is everything save_best_checkpoint (the real
# endpoint) does short of the actual DB upload - confirms the CORRECTLY-
# selected best_state builds a valid, finite snapshot.
snapshot20 = pl.summarize_checkpoint(stored20["model"], stored20["data"], stored20["args"], stored20["best_state"])
assert snapshot20 is not None, "every asset should have a validated checkpoint after 3 restarts x 6 epochs"
_, _, summary20 = snapshot20
assert np.isfinite(summary20["val"]["loss"])
print(
    f"20. api/server.py save-best OK: displayed best-so-far ({job20['best_score_overall']:.4f}) matches what "
    f"actually gets saved across {job20['progress']['n_seeds']} seed restarts"
)

# 21. Merged two-phase job (train_risk_engine=True): _run_training_job
# trains the prediction model (phase 1), then continues STRAIGHT into
# train_risk_engine on the just-trained, in-memory model (phase 2) - no
# separate save/reload round trip, no separate job. Confirms: the job
# reaches "done" with both phases run, progress/interim correctly switch
# to phase "risk_engine" partway through, the result payload carries a
# risk_engine summary, and the ONE saved checkpoint has a RiskEngine
# actually attached (bundled - see PredictionModel._checkpoint_dict's own
# risk_engine key).
job_id21 = "smoke-test-job-21"
srv._JOBS[job_id21] = {
    "status": "pending", "result": None, "error": None, "logs": [],
    "progress": {
        "seed_index": 1, "n_seeds": 1, "lambda_index": 1, "n_lambdas": 1, "epoch": 0, "total_epochs": 3,
        "percent": 0.0, "phase": "prediction", "n_phases": 2, "phase_index": 1,
    },
    "interim": None, "interim_history": [], "stop_requested": False,
}
config21 = {
    **pl.DEFAULT_CONFIG,
    "pairs": continue_pairs, "lookback": 15, "years": 3, "train_frac": 0.8, "test_frac": 0.1,
    "features": ["log_return", "vol"], "cma_windows": [], "bandpass_windows": [],
    "epochs": 3, "n_seeds": 1, "device": "cpu", "hidden_size": 4,
    "checkpoint_metric": "val_loss", "sharpe_weight": 0.0,
    "save_db": False, "model_description": "",
    "train_risk_engine": True,
    "risk_lookback": 10, "risk_rolling_stats_window": 20, "risk_cma_windows": [[10, 50]],
    "risk_bandpass_windows": [], "risk_bandpass_order": 3,
    "min_risk_att": 0.0, "max_risk_att": 1.0, "risk_hidden_size": 4, "risk_num_layers": 1, "risk_dropout": 0.0,
    "risk_n_attn_heads": 2, "risk_epochs": 3, "risk_lr": 1e-2, "risk_weight_decay": 0.0,
    "risk_sortino_window": 8, "full_exposure_penalty": 0.05,
}
_real_save_model21 = pl.PredictionModel.save_model
_tmpdir21 = tempfile.mkdtemp()
def _redirected_save_model21(self, path="models/prediction_model.pt", **kwargs):
    return _real_save_model21(self, os.path.join(_tmpdir21, "job21.pt"), **kwargs)
pl.PredictionModel.save_model = _redirected_save_model21
try:
    srv._run_training_job(job_id21, config21)
finally:
    pl.PredictionModel.save_model = _real_save_model21
job21 = srv._JOBS[job_id21]
assert job21["status"] == "done", job21.get("error")
assert job21["progress"]["phase"] == "risk_engine", job21["progress"]
assert job21["progress"]["phase_index"] == 2
assert job21["progress"]["percent"] == 100.0
assert any(h.get("phase") == "risk_engine" for h in job21["interim_history"]), (
    "expected at least one phase-2 interim entry (train/val Sortino) in interim_history"
)
risk_summary21 = job21["result"]["risk_engine"]
assert risk_summary21 is not None
assert risk_summary21["train_sortino"] is None or np.isfinite(risk_summary21["train_sortino"])
assert risk_summary21["val_sortino"] is None or np.isfinite(risk_summary21["val_sortino"])
saved21 = pl.load_prediction_model(os.path.join(_tmpdir21, "job21.pt"))
assert saved21.risk_engine is not None, "the SAVED checkpoint must have the risk engine bundled in (same job, one save)"
assert saved21.risk_engine.risk_lookback == 10
print(
    f"21. api/server.py two-phase merge OK: phase 1 -> phase 2 in one job, risk engine bundled into the saved "
    f"checkpoint (train_sortino={risk_summary21['train_sortino']}, val_sortino={risk_summary21['val_sortino']})"
)

# 22a. generate_purged_kfold_splits: exhaustive checks on the pure
# splitting primitive - no overlap between any train/val pair, purge zone
# on BOTH sides of every fold's validation block genuinely excluded from
# training, and the documented error cases actually raise.
splits22a = pl.generate_purged_kfold_splits(n_foldable=100, n_folds=5, purge_gap=10)
assert len(splits22a) == 5
all_val22a = []
for k, (train_idx, val_idx) in enumerate(splits22a):
    assert len(set(train_idx.tolist()) & set(val_idx.tolist())) == 0, f"fold {k}: train/val overlap"
    val_lo, val_hi = int(val_idx.min()), int(val_idx.max())
    purge_zone = set(range(max(val_lo - 10, 0), min(val_hi + 10 + 1, 100)))
    leaked = purge_zone.intersection(train_idx.tolist()) - set(val_idx.tolist())
    assert not leaked, f"fold {k}: train indices inside the purge zone: {leaked}"
    all_val22a.extend(val_idx.tolist())
# Every one of the 100 indices is SOMEONE's validation index exactly once
# (the folds partition [0, n_foldable) completely, even though train pools
# overlap across folds).
assert sorted(all_val22a) == list(range(100))
try:
    pl.generate_purged_kfold_splits(n_foldable=3, n_folds=5, purge_gap=1)
    raise AssertionError("expected ValueError: not enough sequences for 5 folds")
except ValueError:
    pass
try:
    pl.generate_purged_kfold_splits(n_foldable=10, n_folds=1, purge_gap=1)
    raise AssertionError("expected ValueError: n_folds must be >= 2")
except ValueError:
    pass
print("22a. generate_purged_kfold_splits OK: no train/val overlap, purge zones respected, validation blocks partition the full range")

# 22b. run_kfold_pipeline end to end (models/portfolio_lstm.py level): 3
# folds train successfully on the same synthetic 300-day fixture used
# above, each fold's own PredictionResult scores finite, EVERY fold's
# model + its own risk engine (train_risk_engine=True exercises the
# per-fold risk-engine path too) is kept as an ensemble member (no more
# "best fold" - see run_kfold_pipeline's own docstring on why), and the
# leakage-aware historical curve covers the WHOLE foldable range exactly
# once with no gaps.
torch.manual_seed(22)
kfold_args22 = argparse.Namespace(**{
    **pl.DEFAULT_CONFIG, "pairs": continue_pairs, "lookback": 15, "years": 3, "train_frac": 0.8, "test_frac": 0.1,
    "features": ["log_return", "vol"], "cma_windows": [], "bandpass_windows": [],
    "epochs": 2, "n_seeds": 1, "device": "cpu", "hidden_size": 4,
    "checkpoint_metric": "val_loss", "sharpe_weight": 0.0, "load_model": None,
    "train_risk_engine": True, "risk_lookback": 10, "risk_epochs": 2, "risk_cma_windows": [],
    "risk_bandpass_windows": [], "min_risk_att": 0.0, "max_risk_att": 1.0,
})
kfold_result22 = pl.run_kfold_pipeline(kfold_args22, n_folds=3)
assert kfold_result22.n_folds == 3
assert len(kfold_result22.fold_scores) == 3
assert all(np.isfinite(s) for s in kfold_result22.fold_scores)
assert abs(kfold_result22.mean_score - float(np.mean(kfold_result22.fold_scores))) < 1e-9
assert kfold_result22.purge_gap == kfold_args22.lookback + kfold_args22.direction_horizon
assert len(kfold_result22.members) == 3
assert all(m.pairs == continue_pairs for m in kfold_result22.members)
assert len(kfold_result22.member_risk_engines) == 3
assert all(re is not None for re in kfold_result22.member_risk_engines), "train_risk_engine=True should give every fold its own RiskEngine"
# The historical curve concatenates each fold's own validation block IN
# FOLD ORDER (see _historical_ensemble_curve's own docstring on why that's
# already chronological) - confirm the resulting dates are genuinely
# strictly increasing end to end, not just "non-empty per block".
hist_dates22 = kfold_result22.historical_dates
assert len(hist_dates22) > 0
assert (hist_dates22[1:] > hist_dates22[:-1]).all(), "historical curve dates must be strictly chronological"
assert np.isfinite(kfold_result22.historical_probabilities).all()
assert ((kfold_result22.historical_probabilities >= 0) & (kfold_result22.historical_probabilities <= 1)).all()
# mu/sigma/z_labels (the "forecast vs actual" distribution chart's own
# inputs - see _distribution_payload) must be tracked too, same length as
# everything else - a real regression once left these empty for every
# ensemble job (see api/server.py's own distribution payload).
assert len(kfold_result22.historical_mu) == len(hist_dates22)
assert len(kfold_result22.historical_sigma) == len(hist_dates22)
assert len(kfold_result22.historical_z_labels) == len(hist_dates22)
assert np.isfinite(kfold_result22.historical_mu).all()
assert np.isfinite(kfold_result22.historical_sigma).all() and (kfold_result22.historical_sigma > 0).all()
assert len(kfold_result22.test_dates) == len(kfold_result22.test_probabilities)
assert len(kfold_result22.test_mu) == len(kfold_result22.test_dates)
assert len(kfold_result22.test_sigma) == len(kfold_result22.test_dates)
assert len(kfold_result22.test_z_labels) == len(kfold_result22.test_dates)
print(
    f"22b. run_kfold_pipeline OK: 3 folds (+ per-fold risk engines), val_loss scores "
    f"{[round(s, 4) for s in kfold_result22.fold_scores]}, historical curve spans {len(kfold_result22.historical_dates)} days"
)

# 22c. api/server.py end to end with use_kfold_cv=True: the job reaches
# "done", progress/interim carry fold_index/n_folds throughout, and the
# SAVED checkpoint is a genuine EnsemblePredictionModel bundling every
# fold's own model + risk engine - not any single "winning" one.
job_id22 = "smoke-test-job-22"
srv._JOBS[job_id22] = {
    "status": "pending", "result": None, "error": None, "logs": [],
    "progress": {
        "seed_index": 1, "n_seeds": 1, "lambda_index": 1, "n_lambdas": 1, "epoch": 0, "total_epochs": 2,
        "percent": 0.0, "phase": "prediction", "n_phases": 1, "phase_index": 1,
        "fold_index": 1, "n_folds": 3, "global_percent": 0.0,
    },
    "interim": None, "interim_history": [], "stop_requested": False,
}
config22 = {
    **pl.DEFAULT_CONFIG,
    "pairs": continue_pairs, "lookback": 15, "years": 3, "train_frac": 0.8, "test_frac": 0.1,
    "features": ["log_return", "vol"], "cma_windows": [], "bandpass_windows": [],
    "epochs": 2, "n_seeds": 1, "device": "cpu", "hidden_size": 4,
    "checkpoint_metric": "val_loss", "sharpe_weight": 0.0,
    "save_db": False, "model_description": "",
    "use_kfold_cv": True, "n_folds": 3,
    "train_risk_engine": True, "risk_lookback": 10, "risk_epochs": 2, "risk_cma_windows": [],
    "risk_bandpass_windows": [], "min_risk_att": 0.0, "max_risk_att": 1.0,
}
# save_ensemble_model always writes to its own EXPLICIT path
# ("models/prediction_model.pt" by default, same tracked-file risk as
# every other save call in this file) - _run_training_job's ensemble
# branch calls it with that exact literal default, so the SAME global
# safety net (see this file's own top-of-file _safe_default_save_model)
# already redirects it; no extra per-test patch needed here.
srv._run_training_job(job_id22, config22)
job22 = srv._JOBS[job_id22]
assert job22["status"] == "done", job22.get("error")
assert job22["progress"]["n_folds"] == 3
assert job22["progress"]["fold_index"] == 3, "should have advanced through to the LAST fold by job completion"
assert job22["progress"]["global_percent"] == 100.0
assert any(h.get("n_folds") == 3 for h in job22["interim_history"]), (
    "expected at least one interim entry tagged with n_folds=3 (see _on_fold_start/current_fold_index)"
)
kfold_summary22 = job22["result"]["kfold"]
assert kfold_summary22 is not None
assert kfold_summary22["n_folds"] == 3
assert len(kfold_summary22["fold_scores"]) == 3
assert kfold_summary22["n_members"] == 3
assert kfold_summary22["has_risk_engine"] is True
assert "best_fold_index" not in kfold_summary22, "THE OLD BUG IS BACK: ensembling should never pick a single 'best' fold"
# "train" is intentionally empty (no single train/val split for an
# ensemble - see api/server.py's own ensemble branch); "val" carries the
# leakage-aware historical curve, "test" the shared held-out split - both
# populated for every configured pair.
assert job22["result"]["hit_rate"]["train"] == {p: 0.5 for p in continue_pairs}
assert set(job22["result"]["hit_rate"]["val"].keys()) == set(continue_pairs)
assert set(job22["result"]["hit_rate"]["test"].keys()) == set(continue_pairs)
assert len(job22["result"]["probabilities"]["train"]["dates"]) == 0
assert len(job22["result"]["probabilities"]["val"]["dates"]) > 0
assert len(job22["result"]["probabilities"]["test"]["dates"]) > 0
# THE BUG THIS GUARDS AGAINST: the "forecast vs actual" distribution chart
# was left EMPTY for every ensemble job (val/test included) - only
# probabilities were ever tracked through _historical_ensemble_curve, not
# mu/sigma/z_labels (see models/portfolio_lstm.py's own fix).
for pair in continue_pairs:
    assert len(job22["result"]["distribution"]["val"][pair]["actual"]) > 0, "distribution chart is empty for 'val' - the bug is back"
    assert len(job22["result"]["distribution"]["test"][pair]["actual"]) > 0, "distribution chart is empty for 'test' - the bug is back"
# The SAVED checkpoint is the actual ensemble, not any single fold's model.
saved22 = pl.load_prediction_model(os.path.join(_SMOKE_TEST_SAVE_DIR, "unbracketed_save.pt"))
assert isinstance(saved22, pl.EnsemblePredictionModel)
assert saved22.n_members == 3
assert saved22.risk_engine is not None, "every fold got its own risk engine (train_risk_engine=True) - the ensemble should report having one"
print(
    f"22c. api/server.py use_kfold_cv=True OK: 3 folds end to end, progress/interim fold-tagged throughout, "
    f"saved checkpoint is a genuine {saved22.n_members}-member ensemble with risk engines attached"
)

# 23a/23b. _train_restart_with_retry: a single restart's non-finite-
# gradient failure (the known MPS multi-layer-LSTM bug - see
# _assert_finite_grad) no longer aborts the WHOLE multi-seed/K-fold run -
# it retries THAT restart, falling back mps->cpu on the very first
# failure (retrying on the SAME device would just reproduce a
# deterministic bug identically), and only re-raises once
# MAX_TRAIN_ATTEMPTS is exhausted. Exercised without real MPS hardware by
# monkeypatching pl._train_and_evaluate itself - a real _PreparedData
# fixture (reused from the continue_pairs/continue_returns fixture above,
# via _build_full_sequences/_prepare_data) supplies real CPU tensors so
# _prepared_data_to_device's own .to(device) calls have something valid to
# operate on, with .device overridden to simulate "mps".
args23 = argparse.Namespace(**{
    **pl.DEFAULT_CONFIG, "pairs": continue_pairs, "lookback": 15, "years": 3, "train_frac": 0.8, "test_frac": 0.1,
    "features": ["log_return", "vol"], "cma_windows": [], "bandpass_windows": [], "device": "cpu",
})
data23 = pl._prepare_data(args23)
data23.device = torch.device("mps")  # simulate - real tensors stay on cpu underneath, only the LABEL says mps

_real_train_and_evaluate = pl._train_and_evaluate


def _make_fake_train_and_evaluate(fail_times, message="Non-finite gradient during test"):
    calls = []

    def _fake(data, args, on_best_checkpoint=None):
        calls.append(data.device)
        if len(calls) <= fail_times:
            raise RuntimeError(message)
        return "SUCCESS_SENTINEL"

    return _fake, calls


# 23a: fails twice, succeeds on the 3rd attempt - falls back to cpu after
# the FIRST failure and stays there (retrying "mps" again would be
# pointless for a deterministic bug).
pl._train_and_evaluate, calls23a = _make_fake_train_and_evaluate(fail_times=2)
try:
    result23a = pl._train_restart_with_retry(data23, argparse.Namespace(**vars(args23)), None, "test-restart-23a", seed=0)
finally:
    pl._train_and_evaluate = _real_train_and_evaluate
assert result23a == "SUCCESS_SENTINEL"
assert len(calls23a) == 3
assert calls23a[0] == torch.device("mps")
assert calls23a[1] == torch.device("cpu")
assert calls23a[2] == torch.device("cpu")
print("23a. _train_restart_with_retry OK: retries in place, falls back mps->cpu on first failure, succeeds without aborting the whole run")

# 23b: fails every time (even after the cpu fallback) - must give up
# after EXACTLY MAX_TRAIN_ATTEMPTS attempts, with a clear error, not hang
# or retry forever.
pl._train_and_evaluate, calls23b = _make_fake_train_and_evaluate(fail_times=999)
try:
    try:
        pl._train_restart_with_retry(data23, argparse.Namespace(**vars(args23)), None, "test-restart-23b", seed=0)
        raise AssertionError("expected RuntimeError after exhausting all retry attempts")
    except RuntimeError as exc:
        assert "test-restart-23b" in str(exc) and str(pl.MAX_TRAIN_ATTEMPTS) in str(exc)
finally:
    pl._train_and_evaluate = _real_train_and_evaluate
assert len(calls23b) == pl.MAX_TRAIN_ATTEMPTS
print(f"23b. _train_restart_with_retry OK: gives up after exactly {pl.MAX_TRAIN_ATTEMPTS} attempts (not fewer, not forever)")

# 24a. continue_training_ensemble (models/portfolio_lstm.py level):
# continues EVERY member of an already-trained ensemble (reusing saved22
# from test 22c, which already has 3 members + their own risk engines)
# independently, each getting a FRESH risk engine (train_risk_engine=True) -
# the OLD ones must never be silently carried forward (see
# continue_training's own discipline, mirrored here per member).
old_risk_engines24 = [m.risk_engine for m in saved22.members]
assert all(re is not None for re in old_risk_engines24)
continue_ensemble_args24 = argparse.Namespace(**{
    **pl.DEFAULT_CONFIG, "pairs": continue_pairs, "years": 3, "train_frac": 0.8, "test_frac": 0.1,
    "epochs": 2, "device": "cpu",
    "checkpoint_metric": "val_loss", "sharpe_weight": 0.0, "load_model": None,
    "train_risk_engine": True, "risk_lookback": 10, "risk_epochs": 2, "risk_cma_windows": [],
    "risk_bandpass_windows": [], "min_risk_att": 0.0, "max_risk_att": 1.0,
})
member_results24, member_risk_engines24 = pl.continue_training_ensemble(continue_ensemble_args24, saved22)
assert len(member_results24) == saved22.n_members == 3
assert all(r.pairs == continue_pairs for r in member_results24)
assert len(member_risk_engines24) == 3
assert all(re is not None for re in member_risk_engines24)
assert all(new is not old for new, old in zip(member_risk_engines24, old_risk_engines24)), (
    "a continued member's risk engine must be a FRESH one, never the old one carried forward"
)
ensemble_temp24 = pl.EnsemblePredictionModel([r.model for r in member_results24])
averaged24 = pl._average_ensemble_results(member_results24, ensemble_temp24)
assert averaged24.pairs == continue_pairs
assert len(averaged24.dates_train) > 0 and len(averaged24.dates_val) > 0
print("24a. continue_training_ensemble OK: continued 3 members independently, each with a genuinely fresh risk engine")

# 24b. api/server.py end to end: continue_from pointed at a saved ENSEMBLE
# checkpoint continues every member, saves a NEW ensemble (never
# overwriting the base), and reports genuine train/val/test splits -
# unlike a fresh K-fold job (which has none, only historical/test - see
# the "ensemble" vs "kfold" result fields, deliberately distinct shapes).
base_ensemble_path24 = os.path.join(_SMOKE_TEST_SAVE_DIR, "unbracketed_save.pt")
job_id24 = "smoke-test-job-24"
srv._JOBS[job_id24] = {
    "status": "pending", "result": None, "error": None, "logs": [],
    "progress": {
        "seed_index": 1, "n_seeds": 1, "lambda_index": 1, "n_lambdas": 1, "epoch": 0, "total_epochs": 2,
        "percent": 0.0, "phase": "prediction", "n_phases": 1, "phase_index": 1,
        "fold_index": 1, "n_folds": 1, "global_percent": 0.0,
    },
    "interim": None, "interim_history": [], "stop_requested": False,
}
config24 = {
    **pl.DEFAULT_CONFIG,
    "pairs": [], "years": 3, "train_frac": 0.8, "test_frac": 0.1,
    "epochs": 2, "device": "cpu",
    "checkpoint_metric": "val_loss", "sharpe_weight": 0.0,
    "save_db": False, "model_description": "",
    "continue_from": base_ensemble_path24,
    "train_risk_engine": True, "risk_lookback": 10, "risk_epochs": 2, "risk_cma_windows": [],
    "risk_bandpass_windows": [], "min_risk_att": 0.0, "max_risk_att": 1.0,
}
srv._run_training_job(job_id24, config24)
job24 = srv._JOBS[job_id24]
assert job24["status"] == "done", job24.get("error")
assert job24["result"]["kfold"] is None
ensemble_summary24 = job24["result"]["ensemble"]
assert ensemble_summary24 is not None
assert ensemble_summary24["n_members"] == 3
assert ensemble_summary24["has_risk_engine"] is True
# Unlike a fresh K-fold job, a CONTINUED ensemble has genuine train/val/test
# splits (each member's own continue_training produces one).
assert len(job24["result"]["probabilities"]["train"]["dates"]) > 0
assert len(job24["result"]["probabilities"]["val"]["dates"]) > 0
saved24 = pl.load_prediction_model(base_ensemble_path24)
assert isinstance(saved24, pl.EnsemblePredictionModel)
assert saved24.n_members == 3
print("24b. api/server.py continue_from=<ensemble> OK: continued all 3 members, saved a new ensemble, genuine train/val/test reported")

# 25. run_kfold_pipeline on device="mps" (skipped where MPS isn't
# available, e.g. most CI - this only runs on Apple Silicon dev machines,
# but that's exactly where the bug it guards against was actually hit).
# THE BUG THIS GUARDS AGAINST: _historical_ensemble_curve's own mu/sigma
# tracking (added for the "forecast vs actual" distribution chart fix)
# multiplied a tensor still on `device` (mps) by `sigma_hat_t.cpu()` -
# `.cpu()` applied to only ONE operand mid-expression instead of to the
# final result - raising "Expected all tensors to be on the same device"
# the moment a K-fold job actually ran on MPS. A CPU-only smoke run can
# never catch this class of bug (CPU-vs-CPU is trivially consistent) - it
# needs a REAL non-cpu device to reproduce, hence the explicit availability
# check rather than unconditionally running this on every machine.
if torch.backends.mps.is_available():
    torch.manual_seed(25)
    mps_args25 = argparse.Namespace(**{
        **pl.DEFAULT_CONFIG, "pairs": continue_pairs, "lookback": 15, "years": 3, "train_frac": 0.8, "test_frac": 0.1,
        "features": ["log_return", "vol"], "cma_windows": [], "bandpass_windows": [],
        "epochs": 2, "n_seeds": 1, "device": "mps", "hidden_size": 4,
        "checkpoint_metric": "val_loss", "sharpe_weight": 0.0, "load_model": None,
    })
    kfold_result25 = pl.run_kfold_pipeline(mps_args25, n_folds=3)
    assert len(kfold_result25.historical_dates) > 0
    assert np.isfinite(kfold_result25.historical_probabilities).all()
    assert np.isfinite(kfold_result25.historical_mu).all()
    assert np.isfinite(kfold_result25.historical_sigma).all()
    assert np.isfinite(kfold_result25.test_mu).all()
    assert np.isfinite(kfold_result25.test_sigma).all()
    print("25. run_kfold_pipeline on device='mps' OK: no cpu/mps device-mismatch, historical curve mu/sigma finite")
else:
    print("25. run_kfold_pipeline on device='mps' SKIPPED (MPS not available on this machine)")

print("\nALL SMOKE TESTS PASSED")
