-- Model registry: stores trained PortfolioLSTM/RiskLSTM checkpoints (and
-- their ensemble variants) in Postgres, so they can be loaded later by
-- name instead of from a local .pt file.
--
-- `name` is derived deterministically from the characteristics of the
-- arguments the model was trained with (pairs, weight scheme, lookback,
-- hidden size, target vol, etc. - see data/model_registry.py's
-- build_model_name()), so the same training configuration always maps to
-- the same name and re-saving under it is a natural overwrite/versioned
-- update rather than an accidental collision with something unrelated.

CREATE SCHEMA IF NOT EXISTS quant;

CREATE TABLE IF NOT EXISTS quant.model_registry (
    name         text PRIMARY KEY,
    model_type   text NOT NULL,               -- e.g. 'portfolio', 'portfolio_ensemble', 'risk', 'risk_ensemble'
    description  text NOT NULL DEFAULT '',
    blob         bytea NOT NULL,               -- the serialized torch checkpoint (torch.save output)
    created_at   timestamptz NOT NULL DEFAULT now(),
    updated_at   timestamptz NOT NULL DEFAULT now()
);

COMMENT ON TABLE quant.model_registry IS
    'Trained model checkpoints (PortfolioLSTM/RiskLSTM, single or ensemble), keyed by a name derived from training-argument characteristics.';
COMMENT ON COLUMN quant.model_registry.name IS
    'Deterministic name built from model type + characteristic training arguments (see data/model_registry.py build_model_name()).';
COMMENT ON COLUMN quant.model_registry.model_type IS
    'One of: portfolio, portfolio_ensemble, risk, risk_ensemble.';
COMMENT ON COLUMN quant.model_registry.blob IS
    'Raw bytes of the torch.save() checkpoint (same format as the local .pt file), loadable via torch.load(io.BytesIO(blob)).';
