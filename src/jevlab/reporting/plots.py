"""Plot helpers for experiment reports (Plotly HTML snippets; no recomputation)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:  # pragma: no cover
    go = None  # type: ignore[assignment]
    make_subplots = None  # type: ignore[assignment]


def _fig_to_html(fig: Any, *, include_js: bool = False) -> str:
    if fig is None:
        return "<p><em>Plot unavailable.</em></p>"
    return fig.to_html(full_html=False, include_plotlyjs="cdn" if include_js else False)


def equity_drawdown_figure(
    equity: Sequence[Mapping[str, Any]] | None,
    *,
    title: str = "Equity and drawdown",
) -> str:
    """Build equity + drawdown subplot HTML from precomputed equity rows.

    Expected keys per row: ``date``, ``equity_usd`` (optional ``drawdown_dec``).
    """
    if not equity or go is None or make_subplots is None:
        return "<p><em>Equity/drawdown data not available in manifest.</em></p>"
    dates = [str(r.get("date", "")) for r in equity]
    eq = [float(r.get("equity_usd", float("nan"))) for r in equity]
    if "drawdown_dec" in equity[0]:
        dd = [float(r.get("drawdown_dec", 0.0)) for r in equity]
    else:
        arr = np.asarray(eq, dtype=float)
        peak = np.maximum.accumulate(np.where(np.isfinite(arr), arr, -np.inf))
        dd = list(1.0 - arr / np.where(peak > 0, peak, np.nan))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    fig.add_trace(go.Scatter(x=dates, y=eq, name="equity_usd", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=dd, name="drawdown_dec", mode="lines"), row=2, col=1)
    fig.update_layout(title=title, height=480, margin=dict(l=40, r=20, t=40, b=40))
    return _fig_to_html(fig)


def placebo_histogram_figure(
    samples: Sequence[float] | None,
    observed: float | None,
    *,
    title: str = "Placebo distribution",
    xlabel: str = "delta_log_loss_gain",
) -> str:
    """Histogram of placebo draws with observed value marked."""
    if samples is None or observed is None or go is None:
        return "<p><em>Placebo histogram data not available in manifest.</em></p>"
    arr = [float(x) for x in samples if x is not None and np.isfinite(float(x))]
    if not arr:
        return "<p><em>Placebo samples empty.</em></p>"
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=arr, name="placebo", nbinsx=min(40, max(10, len(arr) // 5))))
    fig.add_vline(
        x=float(observed),
        line_dash="dash",
        annotation_text=f"observed={float(observed):.4f}",
    )
    fig.update_layout(
        title=title, xaxis_title=xlabel, height=360, margin=dict(l=40, r=20, t=40, b=40)
    )
    return _fig_to_html(fig)


def reliability_figure(
    bins: Mapping[str, Any] | None,
    *,
    title: str = "Reliability diagram",
) -> str:
    """Calibration / reliability curve from bin means."""
    if not bins or go is None:
        return "<p><em>Calibration/reliability data not available in manifest.</em></p>"
    mean_pred = bins.get("mean_predicted") or bins.get("bin_mean_pred") or []
    frac_pos = bins.get("frac_positive") or bins.get("bin_frac_pos") or []
    if not mean_pred or not frac_pos:
        return "<p><em>Calibration bins incomplete.</em></p>"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="ideal",
            line=dict(dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[float(x) for x in mean_pred],
            y=[float(y) for y in frac_pos],
            mode="lines+markers",
            name="empirical",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="mean predicted",
        yaxis_title="fraction positive",
        height=360,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return _fig_to_html(fig)


def trade_distribution_figure(
    trade_pnls: Sequence[float] | None,
    *,
    title: str = "Trade P&L distribution",
) -> str:
    if trade_pnls is None or go is None:
        return "<p><em>Trade distribution data not available in manifest.</em></p>"
    arr = [float(x) for x in trade_pnls if np.isfinite(float(x))]
    if not arr:
        return "<p><em>No trades.</em></p>"
    fig = go.Figure(data=[go.Histogram(x=arr, nbinsx=min(40, max(8, len(arr) // 4)))])
    fig.update_layout(title=title, xaxis_title="trade_pnl_usd", height=320)
    return _fig_to_html(fig)
