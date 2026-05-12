"""Options DTE ladder views."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def plot_options_expiration_ladder(positions_df: pd.DataFrame) -> go.Figure | None:
    if positions_df.empty:
        print("No option positions available for the DTE ladder.")
        return None

    dte_view = positions_df.copy()
    dte_view["expiration"] = pd.to_datetime(dte_view["expiration"], errors="coerce")
    dte_view["days_to_expiration"] = pd.to_numeric(dte_view["days_to_expiration"], errors="coerce")
    dte_view = dte_view.dropna(subset=["underlying", "expiration", "days_to_expiration", "net_quantity"])

    if dte_view.empty:
        print("No valid option rows available for the DTE ladder.")
        return None

    dte_view["days_to_expiration"] = dte_view["days_to_expiration"].astype(int)
    dte_view["gross_contracts"] = dte_view["net_quantity"].abs()
    dte_view["expiration_label"] = (
        dte_view["expiration"].dt.strftime("%Y-%m-%d")
        + " | "
        + dte_view["days_to_expiration"].astype(str)
        + " DTE"
    )
    expiration_order_frame = (
        dte_view[["expiration_label", "expiration", "days_to_expiration"]]
        .drop_duplicates()
        .sort_values(["expiration", "days_to_expiration"])
    )
    expiration_order = expiration_order_frame["expiration_label"].tolist()
    underlying_order = dte_view.groupby("underlying")["gross_contracts"].sum().sort_values(ascending=False).index.tolist()
    ladder_frame = (
        dte_view.groupby(["expiration_label", "underlying"], as_index=False)
        .agg(
            gross_contracts=("gross_contracts", "sum"),
            days_to_expiration=("days_to_expiration", "first"),
            expiration=("expiration", "first"),
        )
    )
    ladder_frame["expiration_label"] = pd.Categorical(ladder_frame["expiration_label"], categories=expiration_order, ordered=True)
    ladder_frame = ladder_frame.sort_values(["expiration_label", "underlying"])

    full_dte_range = list(range(0, int(dte_view["days_to_expiration"].max()) + 1))
    dense_dtick = 1 if len(full_dte_range) <= 31 else 5 if len(full_dte_range) <= 120 else 10 if len(full_dte_range) <= 260 else 25
    dte_axis_frame = dte_view.groupby(["days_to_expiration", "underlying"], as_index=False)["gross_contracts"].sum()
    dte_expiration_lookup = (
        dte_view[["days_to_expiration", "expiration"]]
        .drop_duplicates()
        .sort_values(["days_to_expiration", "expiration"])
        .assign(expiration=lambda frame: frame["expiration"].dt.strftime("%Y-%m-%d"))
        .groupby("days_to_expiration")["expiration"]
        .agg(", ".join)
        .reindex(full_dte_range, fill_value="No listed expiration")
    )
    dense_expiration_labels = dte_expiration_lookup.tolist()
    sparse_hovertemplate = "Underlying: %{fullData.name}<br>Expiration: %{x}<br>Gross Contracts: %{y:.2f}<extra></extra>"
    dense_hovertemplate = "Underlying: %{fullData.name}<br>DTE: %{x}<br>Expiration(s): %{customdata}<br>Gross Contracts: %{y:.2f}<extra></extra>"

    print("[DTE Ladder] Expiration Ladder Summary:")
    print(f"- Option rows: {len(dte_view):,}")
    print(f"- Unique underlyings: {dte_view['underlying'].nunique():,}")
    print(f"- Expiration dates tracked: {dte_view['expiration'].nunique():,}")
    print(f"- Nearest expiration: {dte_view['days_to_expiration'].min()} DTE")
    print(f"- Furthest expiration: {dte_view['days_to_expiration'].max()} DTE")
    print(f"- Gross contracts tracked: {dte_view['gross_contracts'].sum():,.2f}")

    sparse_x_data = []
    sparse_y_data = []
    sparse_customdata = []
    sparse_hovertemplates = []
    dense_x_data = []
    dense_y_data = []
    dense_customdata = []
    dense_hovertemplates = []
    fig = go.Figure()

    for underlying in underlying_order:
        sparse_slice = ladder_frame[ladder_frame["underlying"] == underlying].sort_values("days_to_expiration")
        if sparse_slice.empty:
            continue

        dense_slice = (
            dte_axis_frame[dte_axis_frame["underlying"] == underlying]
            .set_index("days_to_expiration")
            .reindex(full_dte_range, fill_value=0.0)
            .rename_axis("days_to_expiration")
            .reset_index()
        )
        sparse_x = sparse_slice["expiration_label"].astype(str).tolist()
        sparse_y = sparse_slice["gross_contracts"].tolist()
        dense_x = dense_slice["days_to_expiration"].tolist()
        dense_y = dense_slice["gross_contracts"].tolist()
        sparse_x_data.append(sparse_x)
        sparse_y_data.append(sparse_y)
        sparse_customdata.append([None] * len(sparse_x))
        sparse_hovertemplates.append(sparse_hovertemplate)
        dense_x_data.append(dense_x)
        dense_y_data.append(dense_y)
        dense_customdata.append(dense_expiration_labels)
        dense_hovertemplates.append(dense_hovertemplate)
        fig.add_trace(go.Bar(x=sparse_x, y=sparse_y, name=underlying, customdata=[None] * len(sparse_x), hovertemplate=sparse_hovertemplate))

    dense_shapes = [
        dict(
            type="line",
            x0=dte_marker,
            x1=dte_marker,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color=color, width=2, dash="dash"),
        )
        for dte_marker, color in zip([21, 50, 200], ["#00CC96", "#FECB52", "#EF553B"])
        if dte_marker <= full_dte_range[-1]
    ]
    fig.update_layout(
        title="Options Expiration Ladder",
        template="plotly_dark",
        height=760,
        barmode="stack",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0, font=dict(size=9)),
        margin=dict(l=50, r=50, t=135, b=120),
        updatemenus=[
            dict(
                buttons=[
                    dict(
                        label="By Expiration",
                        method="update",
                        args=[
                            {
                                "x": sparse_x_data,
                                "y": sparse_y_data,
                                "customdata": sparse_customdata,
                                "hovertemplate": sparse_hovertemplates,
                            },
                            {
                                "title": {"text": "Options Expiration Ladder"},
                                "shapes": [],
                                "xaxis": dict(
                                    title="Expiration | DTE",
                                    tickangle=-35,
                                    type="category",
                                    categoryorder="array",
                                    categoryarray=expiration_order,
                                ),
                            },
                        ],
                    ),
                    dict(
                        label="Full DTE Axis",
                        method="update",
                        args=[
                            {
                                "x": dense_x_data,
                                "y": dense_y_data,
                                "customdata": dense_customdata,
                                "hovertemplate": dense_hovertemplates,
                            },
                            {
                                "title": {"text": "Options Expiration Ladder | Full DTE Axis"},
                                "shapes": dense_shapes,
                                "xaxis": dict(
                                    title="Days to Expiration",
                                    tickangle=0,
                                    type="linear",
                                    range=[-0.5, full_dte_range[-1] + 0.5],
                                    dtick=dense_dtick,
                                ),
                            },
                        ],
                    ),
                ],
                direction="down",
                showactive=True,
                x=0.5,
                xanchor="center",
                y=1.14,
                yanchor="top",
            )
        ],
    )
    fig.update_xaxes(title_text="Expiration | DTE", tickangle=-35, type="category", categoryorder="array", categoryarray=expiration_order)
    fig.update_yaxes(title_text="Gross Contracts")
    return fig
