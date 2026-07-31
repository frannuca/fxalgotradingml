# Debug configs

22 small, fast-running configs, each isolating **one** architectural
decision (or a closely-related pair) from the codebase, meant to be run
with a debugger attached so you can step through the exact code path each
one exercises. Every config trains in a couple of seconds (`epochs: 5`,
`hidden_size: 8`, 3 years of history) - the point isn't a good Sharpe
ratio, it's a training loop short enough to single-step through more than
once without losing patience.

## Running one

```bash
python main.py configs/debug/01_baseline.json
```

Or, in VS Code / PyCharm, set `main.py` as the debug target with
`configs/debug/<name>.json` as its one argument, drop a breakpoint at the
file:function pointer below, and step in.

**Heads up - two side effects of every run, not just these debug ones:**
- `main.py` always writes `models/portfolio_lstm.pt` (and
  `models/risk_lstm.pt` if `risk_overlay` is on) unconditionally - there's
  no config key to change that path. If you have a real trained model
  there you care about, back it up before running any of these.
- Every config below sets `"save_db": false`, so none of them touch
  `quant.model_registry` - only the local `.pt` files above are written.
- Each config gets its own `output`/`position_output`/etc. paths under
  `configs/debug/output/`, so runs never clobber each other's plots (or
  your real `results/*.png`).

## The configs, in order, with where to look

| # | Config | What it isolates | Breakpoint here |
|---|--------|-------------------|------------------|
| 01 | `01_baseline.json` | The default pipeline: softmax weights, concat encoder, last-hidden-state pooling, sample covariance, rolling-window Sharpe objective. Run this first - everything else is a one-line diff from it. | `models/portfolio_lstm.py`: `_prepare_data`, `PortfolioLSTM.forward` (the `encoder_type == "concat"` branch), `train_portfolio_model`'s epoch loop |
| 02 | `02_weight_scheme_tanh_norm.json` | Long/short book: `tanh(logits)` then L1-normalized, instead of softmax | `PortfolioLSTM._weights_from_logits`, the `"tanh_norm"` branch |
| 03 | `03_use_prev_weight.json` | Recurrent-policy feedback - sample *i*'s head input includes sample *i-1*'s own decided weight | `PortfolioLSTM.forward_sequence` - note the `if not self.use_prev_weight: return self.forward(X)` fast path is skipped here, forcing the sequential Python loop |
| 04 | `04_noisy_head.json` | NoisyLinear: the head's weight/bias are themselves stochastic (`mu + sigma * epsilon`), resampled every `forward()` call while training | `NoisyLinear.forward` and `_reset_noise` |
| 05 | `05_noisy_head_plus_prev_weight.json` | The combination that used to raise a real autograd error (`...modified by an in-place operation`) - see the session history for the bug. Step through and watch `freeze_noise`/`resample_noise` keep the SAME noise draw fixed across the whole sequential loop | `PortfolioLSTM.forward_sequence`, the `freeze = self.noisy_head and self.training` block, and `NoisyLinear.freeze_noise`/`resample_noise` |
| 06 | `06_cash_asset.json` | `CASH` as a data-layer pseudo-pair (constant daily return), not a model-layer special case | `_prepare_data`, the `if has_cash:` block that appends `"CASH"` to `pairs`/`returns` BEFORE windowing |
| 07 | `07_carry_and_vol_features.json` | Multi-channel input: FX carry (FRED rate differential) + vol-normalized momentum at 5/20-day horizons, asset-major channel layout (`n_channels > 1`) | `build_feature_dataframe`, `load_carry`, `vol_normalized_returns`; then `_prepare_data`'s `X[:, :, 0::n_channels]` raw-return recovery for vol-targeting |
| 08 | `08_per_asset_encoder_attention.json` | Shared per-asset LSTM (same weights reused for every asset) + cross-asset self-attention | `PortfolioLSTM.forward`, the `encoder_type == "per_asset"` branch, `self.asset_attn(h, h, h, ...)` |
| 09 | `09_per_asset_encoder_mean.json` | Same per-asset encoder, but assets combined via a plain mean-pool instead of attention | Same branch as #08, the `else: context = h.mean(dim=1, ...)` line |
| 10 | `10_attention_pooling.json` | Learned attention pooling over every timestep's hidden state, instead of `h_n[-1]` (concat encoder) | `TemporalAttentionPool.forward` |
| 11 | `11_covariance_ewma.json` | Volatility targeting's covariance estimate: exponentially-weighted (RiskMetrics-style) instead of the plain sample covariance | `estimate_covariance`, the `"ewma"` branch |
| 12 | `12_covariance_ledoit_wolf.json` | Ledoit-Wolf shrinkage toward a scaled-identity target, with the analytically optimal (not hand-tuned) shrinkage intensity | `estimate_covariance`, the `"ledoit_wolf"` branch - watch `pi_hat`/`gamma_hat`/`delta` get computed |
| 13 | `13_objective_kelly.json` | Training loss: negative expected log-wealth growth (Kelly criterion) instead of the rolling-window Sharpe ratio | `compute_training_loss` dispatcher -> `kelly_loss` |
| 14 | `14_objective_cvar.json` | Training loss: mean return net of an annualized CVaR tail-risk penalty | `compute_training_loss` -> `mean_cvar_loss` -> `cvar` |
| 15 | `15_multi_seed_best.json` | 3 independent random restarts; the one with the best validation Sharpe wins | `run_pipeline_multi_seed`'s restart loop and the `"best"` selection (`max(enumerate(results), key=...)`) |
| 16 | `16_multi_seed_ensemble.json` | Same 3 restarts, averaged into one `EnsemblePortfolioLSTM` instead of picking a single winner | `EnsemblePortfolioLSTM.forward` - note the re-normalization after averaging |
| 17 | `17_test_split.json` | Chronological 3-way split: the most recent 15% is carved off FIRST as a test set that never influences training or checkpoint selection | `_prepare_data`, the `test_frac`/`n_test`/`n_remaining` block |
| 18 | `18_transaction_costs.json` | Turnover-based transaction costs subtracted INSIDE the training loss (not just a post-hoc reporting adjustment) | `apply_transaction_costs_torch`, called from `train_portfolio_model`'s epoch loop |
| 19 | `19_risk_overlay_basic.json` | RiskLSTM overlay with its defaults - per-asset rolling std/skew/kurtosis of the weighted PnL path | `add_risk_overlay`, `make_risk_sequences`, `RiskLSTM.forward` (in `models/risk_lstm.py`) |
| 20 | `20_risk_overlay_cross_sectional.json` | RiskLSTM also sees 3 portfolio-wide channels: average pairwise correlation, correlation dispersion, top-eigenvalue share | `cross_sectional_features` - note the `torch.linalg.eigvalsh` call (CPU-only - MPS doesn't implement it) |
| 21 | `21_risk_overlay_attention_pooling.json` | Both networks pool over time via learned attention instead of `h_n[-1]` | `RiskLSTM.forward`'s `self.temporal_pool(lstm_out)` branch |
| 22 | `22_kitchen_sink.json` | Everything above, turned on at once - a full-integration smoke test | Set breakpoints wherever you're least sure two features compose correctly |

## A suggested debugging path

1. Run **01** first and watch one full `train_portfolio_model` epoch: step
   into `forward_sequence` (it takes the fast vectorized path here, no
   loop), `scale_weights_to_target_vol`, `compute_training_loss`.
2. Run **03**, then **05** back to back, and diff what's different inside
   `forward_sequence` - this pair tells the whole "why does the sequential
   loop need detach + noise-freezing" story in one sitting.
3. Run **08**/**09** back to back and compare the `combined` tensor's
   shape and values right before the shared head - same head, different
   cross-asset context.
4. Run **19**, then **20**, and compare `features.shape[-1]` in
   `RiskLSTM.forward` (`4*n_assets` vs `4*n_assets + 3`).
5. Finish with **22** to confirm everything you just stepped through
   individually still composes when all switched on together.
