"""SVI implied-volatility surface views."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _raw_iv_trace(frame, *, name, colorscale):
    return go.Scatter3d(
        x=frame["strike"],
        y=frame["Days Till Expiration"],
        z=frame["impliedVolatility"],
        mode="markers",
        marker=dict(size=3, color=frame["impliedVolatility"], colorscale=colorscale, opacity=0.8),
        name=name,
        hovertemplate="Strike: %{x}<br>Days: %{y}<br>IV: %{z:.3f}<extra></extra>",
    )


def _svi_surface_trace(strike_grid, time_grid, iv_surface, *, name, colorscale, showscale=True):
    strike_mesh, dte_mesh = np.meshgrid(strike_grid, time_grid * 365)
    return go.Surface(
        x=strike_mesh,
        y=dte_mesh,
        z=iv_surface,
        colorscale=colorscale,
        opacity=0.7,
        showscale=showscale,
        colorbar=dict(title="IV"),
        name=name,
    )


def _spot_plane_trace(strike_plane, days_mesh, iv_mesh):
    return go.Surface(
        x=strike_plane,
        y=days_mesh,
        z=iv_mesh,
        showscale=False,
        opacity=0.25,
        colorscale=[[0, "gray"], [1, "gray"]],
        hoverinfo="skip",
        name="Spot Price Plane",
    )


def _realized_vol_surface_trace(strike_grid, dte_grid, vol_grid):
    return go.Surface(
        x=strike_grid,
        y=dte_grid,
        z=vol_grid,
        colorscale=[[0, "orange"], [1, "orange"]],
        opacity=0.5,
        showscale=False,
        name="Realized Vol Surface",
        hoverinfo="skip",
    )


def plot_svi_surface_view(
    call_df,
    put_df,
    *,
    call_strike_grid,
    call_t_grid,
    call_iv_svi_surface,
    put_strike_grid,
    put_t_grid,
    put_iv_svi_surface,
    realized_vol_surface_df=None,
    spot_price,
    ticker_label="Asset",
):
    """Compose call and put raw-IV/SVI surfaces with spot and realized-volatility overlays."""
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Call Options IV & SVI Surface", "Put Options IV & SVI Surface"),
    )
    fig.add_trace(_raw_iv_trace(call_df, name="Call Raw IV", colorscale="Viridis"), row=1, col=1)
    fig.add_trace(
        _svi_surface_trace(
            call_strike_grid,
            call_t_grid,
            call_iv_svi_surface,
            name="Call SVI Surface",
            colorscale="Viridis",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(_raw_iv_trace(put_df, name="Put Raw IV", colorscale="Cividis"), row=1, col=2)
    fig.add_trace(
        _svi_surface_trace(
            put_strike_grid,
            put_t_grid,
            put_iv_svi_surface,
            name="Put SVI Surface",
            colorscale="Cividis",
        ),
        row=1,
        col=2,
    )

    strike_min = min(call_df["strike"].min(), put_df["strike"].min())
    strike_max = max(call_df["strike"].max(), put_df["strike"].max())
    days_min = min(call_df["Days Till Expiration"].min(), put_df["Days Till Expiration"].min())
    days_max = max(call_df["Days Till Expiration"].max(), put_df["Days Till Expiration"].max())
    iv_min = min(call_df["impliedVolatility"].min(), put_df["impliedVolatility"].min())
    iv_max = max(call_df["impliedVolatility"].max(), put_df["impliedVolatility"].max())

    days_grid = np.linspace(days_min, days_max, 30)
    iv_grid = np.linspace(iv_min, iv_max, 30)
    days_mesh, iv_mesh = np.meshgrid(days_grid, iv_grid)
    strike_plane = np.full_like(days_mesh, spot_price)
    fig.add_trace(_spot_plane_trace(strike_plane, days_mesh, iv_mesh), row=1, col=1)
    fig.add_trace(_spot_plane_trace(strike_plane, days_mesh, iv_mesh), row=1, col=2)

    if realized_vol_surface_df is not None and not realized_vol_surface_df.empty:
        strike_range = np.linspace(strike_min, strike_max, 50)
        dte_rv_grid, strike_rv_grid = np.meshgrid(realized_vol_surface_df["Days Till Expiration"], strike_range)
        vol_rv_grid = np.tile(realized_vol_surface_df["Realized Vol"].values, (len(strike_range), 1))
        fig.add_trace(_realized_vol_surface_trace(strike_rv_grid, dte_rv_grid, vol_rv_grid), row=1, col=1)
        fig.add_trace(_realized_vol_surface_trace(strike_rv_grid, dte_rv_grid, vol_rv_grid), row=1, col=2)

    scene_layout = dict(
        xaxis=dict(title="Strike", range=[strike_min, strike_max]),
        yaxis=dict(title="Days to Expiration", range=[days_min, days_max]),
        zaxis=dict(title="Implied Volatility", range=[iv_min, iv_max]),
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
    )
    fig.update_layout(
        height=800,
        title_text=f"{ticker_label} Call and Put Options: Raw IV & SVI Surfaces with Spot Price Plane",
        scene=scene_layout,
        scene2=scene_layout,
    )
    return fig
