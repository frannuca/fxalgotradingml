"""Postgres persistence for trained model checkpoints (quant.model_registry).

Reuses data/db.py's connection and the same environment variables. See
data/sql/create_model_registry.sql for the table this reads/writes.

Models are named deterministically from characteristics of the arguments
they were trained with (via build_model_name()), so the same training
configuration always resolves to the same name: saving under it is a
natural update, and --load-{portfolio,risk} can accept that name directly
instead of a local file path.
"""

from __future__ import annotations

import logging

from data.db import get_connection

logger = logging.getLogger(__name__)


def build_model_name(model_type: str, **characteristics: object) -> str:
    """Build a deterministic model name from `model_type` plus whatever
    characteristic arguments distinguish this training run - e.g. for a
    portfolio model: pairs, weight_scheme, lookback, hidden_size,
    target_vol; for a risk model, also risk_hidden_size, risk_rolling_window,
    max_attenuation.

    Sorted by key so argument order never changes the resulting name, and
    list/tuple values (e.g. `pairs`) are joined so the name stays a single
    readable token - e.g.
        portfolio_hidden_size=32_lookback=30_pairs=EURUSD-GBPUSD_target_vol=0.2_weight_scheme=softmax
    """
    parts = [model_type]
    for key in sorted(characteristics):
        value = characteristics[key]
        if isinstance(value, (list, tuple)):
            value = "-".join(str(v) for v in value)
        parts.append(f"{key}={value}")
    return "_".join(parts)


def save_model_blob(name: str, blob: bytes, model_type: str, description: str = "") -> None:
    """Upsert a serialized model checkpoint into quant.model_registry.
    `blob` is the raw bytes of a torch.save() checkpoint (identical format
    to the local .pt file - see models/portfolio_lstm.py's/models/risk_lstm.py's
    save_model() methods).
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO quant.model_registry (name, model_type, description, blob, updated_at)
            VALUES (%s, %s, %s, %s, now())
            ON CONFLICT (name) DO UPDATE SET
                model_type = EXCLUDED.model_type,
                description = EXCLUDED.description,
                blob = EXCLUDED.blob,
                updated_at = now()
            """,
            (name, model_type, description, blob),
        )
        conn.commit()
    logger.info("Saved model %r (%s, %d bytes) to quant.model_registry", name, model_type, len(blob))


def load_model_blob(name: str) -> bytes:
    """Fetch a checkpoint's raw bytes from quant.model_registry by name.
    Raises KeyError if no model is registered under that name.
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT blob FROM quant.model_registry WHERE name = %s", (name,))
        row = cur.fetchone()
    if row is None:
        raise KeyError(f"No model named {name!r} in quant.model_registry")
    return bytes(row[0])


def model_exists(name: str) -> bool:
    """Check whether a model is registered under `name`, without fetching its blob."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM quant.model_registry WHERE name = %s", (name,))
        return cur.fetchone() is not None
