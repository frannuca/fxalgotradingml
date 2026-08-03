# Sharpe-training NaN repro configs

11 small, 1-epoch configs isolating one variable at a time, built to track
down a reported NaN in the portfolio-Sharpe training phase (`sharpe_weight
> 0` - see `models/portfolio_lstm.py`'s `_portfolio_sharpe_loss`) when
training on all 7 major FX pairs with Butterworth bandpass windows
`[[5, 25], [50, 100]]`. `epochs: 1` throughout - the report was that NaN
appears on the very first iteration, so there's nothing to gain from
training longer.

## Running one

```bash
python main.py configs/debug_sharpe_nan/07_sharpe_on_7pairs_bp_both_REPORTED.json
```

Watch the "train sharpe" field in the epoch 1 log line.

## The configs

| # | Config | Isolates |
|---|--------|----------|
| 01 | `01_sanity_sharpe_off` | Baseline - sharpe disabled entirely, 2 pairs, no bandpass. Must never NaN. |
| 02 | `02_sharpe_on_2pairs_nobandpass` | `sharpe_weight > 0` alone, no bandpass - does the Sharpe phase itself ever NaN without bandpass features? |
| 03 | `03_sharpe_on_7pairs_nobandpass` | All 7 pairs, no bandpass - does pair COUNT alone (bigger covariance matrix, bigger joint einsum) trigger it? |
| 04 | `04_sharpe_on_2pairs_bp_5_25` | Bandpass `[5, 25]` only |
| 05 | `05_sharpe_on_2pairs_bp_50_100` | Bandpass `[50, 100]` only |
| 06 | `06_sharpe_on_2pairs_bp_both` | Both windows combined, still only 2 pairs - isolates COMBINING windows from pair count |
| 07 | `07_sharpe_on_7pairs_bp_both_REPORTED` | The exact reported combination: 7 pairs + both bandpass windows |
| 08 | `08_sharpe_on_7pairs_bp_both_window5` | Same as 07, `sharpe_window: 5` |
| 09 | `09_sharpe_on_7pairs_bp_both_window60` | Same as 07, `sharpe_window: 60` |
| 10 | `10_sharpe_on_2pairs_bp_edge_3_10` | `short_period: 3` - right at the edge of what's usable for daily data (Nyquist limit is 2 days) |
| 11 | `11_sharpe_on_7pairs_bp_both_crosspairs` | Same as 07, every pair also cross-linked to every other pair |

All configs use `device: cpu` (deterministic, rules out an MPS-specific
numerical quirk as a variable) and `save_db: false`.

## Resolution

None of the 11 configs above reproduced it - the Sharpe-training code
itself turned out to be innocent. The actual trigger, found by directly
comparing `configs/test_eurusd_gbpusd_usdjpy.json` (which DID reproduce
it) against these: **`num_layers: 2` combined with `hidden_size: 128` on
`device: "mps"`**. Confirmed root cause:

```
loss_i.backward()   # loss itself is finite (~1.46)
# -> gradient is already NaN, on the FIRST backward() call, before any
#    training has happened - the identical forward/backward pass on
#    device="cpu" gives a normal finite gradient (~3.5)
```

This is a PyTorch MPS multi-layer-LSTM backward-kernel bug, not a bug in
this codebase's math (bisected: `hidden_size=128` alone is fine,
`num_layers=2` alone is fine, `n_attn_heads=8` alone is fine - only the
`hidden_size=128` + `num_layers=2` combination together produces a NaN
gradient on MPS). Nothing here can fix PyTorch's own Metal kernel, so two
things were added instead (see `models/portfolio_lstm.py`):

1. `_assert_finite_grad`, called right after every `backward()` call in
   `train_prediction_model` (both the per-asset NLL+BCE phase and the
   optional joint Sharpe phase) - raises immediately with a clear message
   instead of silently training on (and checkpoint-selecting from) a
   corrupted model for however many epochs remain.
2. A proactive `logger.warning` in `_train_and_evaluate` whenever
   `device="mps"` and `num_layers > 1` are combined, pointing at this
   exact issue before training even starts.

Workaround: use `device: "cpu"` or `num_layers: 1` for large
(`hidden_size >= 128`), multi-layer configs on Apple Silicon.
