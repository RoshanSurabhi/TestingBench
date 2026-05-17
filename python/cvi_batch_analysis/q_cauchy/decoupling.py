"""
Facades for call-price *delta* residuals and realized-delta decoupling.

Implementations live in :mod:`cvi_batch_analysis.cvi_batch_plot` (heavy Plotly / CSV).
Import through this module to keep analysis code paths stable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def build_call_price_delta_payload(
    batch_dir: Path,
    *,
    day: str,
    expiry_index: int,
    horizons_min: list[float],
    strike_z_offsets: list[float],
    target_source: str = "both",
) -> dict[str, Any]:
    from cvi_batch_analysis.cvi_batch_plot.call_price_renders import (
        build_call_price_prediction_payload,
    )

    return build_call_price_prediction_payload(
        batch_dir,
        day=day,
        expiry_index=expiry_index,
        horizons_min=horizons_min,
        strike_z_offsets=strike_z_offsets,
        target_source=target_source,
    )


def build_decoupling_payload(*args: Any, **kwargs: Any) -> Any:
    from cvi_batch_analysis.cvi_batch_plot.snapshots_decouple import (
        build_decoupling_payload as _impl,
    )

    return _impl(*args, **kwargs)
