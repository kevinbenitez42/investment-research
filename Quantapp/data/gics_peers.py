"""Reusable GICS peer-frame helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


GICS_LEVELS = ("Sector", "Industry Group", "Industry", "Sub-Industry")
CAPITALIZATION_ORDER = {"Large Cap": 0, "Mid Cap": 1, "Small Cap": 2}
PEER_DISPLAY_COLUMNS = (
    "Rank",
    "Symbol",
    "Normalized Symbol",
    "Capitalization",
    "Sector",
    "Industry Group",
    "Industry",
    "Sub-Industry",
)


@dataclass(frozen=True)
class GICSPeerFrames:
    """GICS peer hierarchy and display tables for one target ticker."""

    target_symbol: str
    target_row: pd.Series
    target_capitalization: str | None
    capitalization_filter: str
    capitalization_list: list[str] | None
    frames: dict[str, pd.DataFrame]
    tables: dict[str, pd.DataFrame]
    summary: pd.DataFrame

    def frame(self, level: str) -> pd.DataFrame:
        """Return the raw peer frame for a GICS level."""
        return self.frames[level]

    def table(self, level: str) -> pd.DataFrame:
        """Return the display table for a GICS level."""
        return self.tables[level]

    def symbols(self, level: str) -> list[str]:
        """Return normalized symbols for a GICS level."""
        return self.frames[level]["Normalized Symbol"].tolist()


def normalize_peer_symbol(symbol) -> str:
    """Normalize ticker symbols for yfinance/FMP-style peer comparisons."""
    return str(symbol).strip().upper().replace(".", "-")


def resolve_capitalization_filter(
    capitalizations,
    target_capitalization,
) -> tuple[list[str] | None, str]:
    """Resolve a capitalization filter into a list plus display label."""
    if isinstance(capitalizations, str) and capitalizations.strip().lower() == "same_as_target":
        capitalization_list = [target_capitalization] if pd.notna(target_capitalization) else None
    elif capitalizations is None:
        capitalization_list = None
    elif isinstance(capitalizations, str):
        capitalization_list = [capitalizations]
    else:
        capitalization_list = [
            capitalization
            for capitalization in list(capitalizations)
            if str(capitalization).strip()
        ]

    capitalization_filter_label = (
        "Large / Mid / Small Cap"
        if capitalization_list is None
        else ", ".join(map(str, capitalization_list))
    )
    return capitalization_list, capitalization_filter_label


def sort_gics_peers(peers: pd.DataFrame) -> pd.DataFrame:
    """Sort GICS peers by capitalization bucket, then symbol."""
    if peers.empty:
        return peers.copy().reset_index(drop=True)

    ranked_peers = peers.copy()
    ranked_peers["Capitalization Rank"] = (
        ranked_peers["Capitalization"].map(CAPITALIZATION_ORDER).fillna(99)
    )
    ranked_peers = ranked_peers.sort_values(["Capitalization Rank", "Symbol"])
    return ranked_peers.drop(columns=["Capitalization Rank"]).reset_index(drop=True)


def build_gics_peer_table(peer_frame: pd.DataFrame, *, symbol_label: str = "Symbol") -> pd.DataFrame:
    """Build a notebook-friendly peer table from a GICS peer frame."""
    table = peer_frame.copy().reset_index(drop=True)
    table.insert(0, "Rank", range(1, len(table) + 1))
    table = table[list(PEER_DISPLAY_COLUMNS)]
    return table.rename(columns={"Normalized Symbol": symbol_label})


def build_capitalization_count_table(peer_table: pd.DataFrame) -> pd.DataFrame:
    """Count peers by Large/Mid/Small capitalization bucket."""
    return (
        peer_table["Capitalization"]
        .value_counts()
        .reindex(["Large Cap", "Mid Cap", "Small Cap"])
        .dropna()
        .astype(int)
        .rename("Peer Count")
        .reset_index()
        .rename(columns={"index": "Capitalization"})
    )


def build_gics_peer_summary(peer_frames: dict[str, pd.DataFrame], target_row: pd.Series) -> pd.DataFrame:
    """Summarize peer counts with GICS levels as rows and capitalization buckets as columns."""
    summary_rows = []
    for level in GICS_LEVELS:
        frame = peer_frames[level]
        counts = frame["Capitalization"].value_counts()
        summary_rows.append(
            {
                "GICS Level": level,
                "Large Cap": int(counts.get("Large Cap", 0)),
                "Mid Cap": int(counts.get("Mid Cap", 0)),
                "Small Cap": int(counts.get("Small Cap", 0)),
                "Total": int(len(frame)),
            }
        )
    summary = pd.DataFrame(summary_rows).set_index("GICS Level")
    summary.columns.name = "Capitalization"
    return summary


def _normalize_companies(companies: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"Symbol", "Capitalization", *GICS_LEVELS}
    missing_columns = sorted(required_columns.difference(companies.columns))
    if missing_columns:
        raise ValueError(f"companies is missing required columns: {missing_columns}")

    company_universe = companies.copy()
    company_universe["Normalized Symbol"] = company_universe["Symbol"].map(normalize_peer_symbol)
    return company_universe


def build_gics_peer_frames(
    ticker,
    *,
    companies: pd.DataFrame,
    capitalizations=None,
    symbol_label: str = "Symbol",
    exclude_target: bool = True,
) -> GICSPeerFrames:
    """Build cumulative Sector -> Sub-Industry peer frames with capitalization applied at every level."""
    target_symbol = normalize_peer_symbol(ticker)
    company_universe = _normalize_companies(companies)

    target_rows = company_universe.loc[company_universe["Normalized Symbol"].eq(target_symbol)]
    if target_rows.empty:
        raise ValueError(
            f"{target_symbol} was not found in the S&P 1500 GICS company universe. "
            "For ETFs or off-index securities, use manual peers or switch to a company ticker."
        )

    target_row = target_rows.iloc[0]
    target_capitalization = target_row.get("Capitalization")
    capitalization_list, capitalization_filter_label = resolve_capitalization_filter(
        capitalizations,
        target_capitalization,
    )

    candidate_universe = company_universe
    if capitalization_list is not None:
        candidate_universe = candidate_universe[
            candidate_universe["Capitalization"].isin(capitalization_list)
        ]

    peer_frames = {}
    current_universe = candidate_universe
    for level in GICS_LEVELS:
        level_value = target_row[level]
        current_universe = current_universe.loc[current_universe[level].eq(level_value)].copy()
        peers_at_level = current_universe
        if exclude_target:
            peers_at_level = peers_at_level.loc[
                ~peers_at_level["Normalized Symbol"].eq(target_symbol)
            ].copy()
        peer_frames[level] = sort_gics_peers(peers_at_level)

    peer_tables = {
        level: build_gics_peer_table(frame, symbol_label=symbol_label)
        for level, frame in peer_frames.items()
    }
    peer_summary = build_gics_peer_summary(peer_frames, target_row)

    return GICSPeerFrames(
        target_symbol=target_symbol,
        target_row=target_row,
        target_capitalization=target_capitalization if pd.notna(target_capitalization) else None,
        capitalization_filter=capitalization_filter_label,
        capitalization_list=capitalization_list,
        frames=peer_frames,
        tables=peer_tables,
        summary=peer_summary,
    )


def choose_gics_peer_level(
    peer_frames: GICSPeerFrames | dict[str, pd.DataFrame],
    *,
    preferred_level: str,
    minimum_peer_count: int = 4,
) -> tuple[str, pd.DataFrame]:
    """Choose the narrowest requested GICS level with enough peers, falling back upward."""
    if preferred_level not in GICS_LEVELS:
        raise ValueError(f"preferred_level must be one of {list(GICS_LEVELS)}.")

    frames = peer_frames.frames if isinstance(peer_frames, GICSPeerFrames) else peer_frames
    preferred_index = GICS_LEVELS.index(preferred_level)
    for level in reversed(GICS_LEVELS[: preferred_index + 1]):
        if len(frames[level]) >= minimum_peer_count or level == "Sector":
            return level, frames[level]

    return preferred_level, frames[preferred_level]


def select_gics_peer_rows(
    peers: pd.DataFrame,
    *,
    peer_count: int,
    target_capitalization=None,
) -> pd.DataFrame:
    """Select a peer sample, rotating across capitalization buckets when possible."""
    if peer_count <= 0 or peers.empty:
        return peers.head(0).copy()

    present_capitalizations = list(dict.fromkeys(peers["Capitalization"].dropna().tolist()))
    bucket_order = []
    if pd.notna(target_capitalization) and target_capitalization in present_capitalizations:
        bucket_order.append(target_capitalization)
    bucket_order.extend(
        capitalization
        for capitalization in sorted(
            present_capitalizations,
            key=lambda value: CAPITALIZATION_ORDER.get(value, 99),
        )
        if capitalization not in bucket_order
    )

    bucket_indices = {
        capitalization: peers.loc[peers["Capitalization"].eq(capitalization)].index.tolist()
        for capitalization in bucket_order
    }
    selected_indices = []
    while len(selected_indices) < peer_count:
        picked_this_round = False
        for capitalization in bucket_order:
            indices = bucket_indices.get(capitalization, [])
            if not indices:
                continue
            selected_indices.append(indices.pop(0))
            picked_this_round = True
            if len(selected_indices) >= peer_count:
                break
        if not picked_this_round:
            break

    if len(selected_indices) < peer_count:
        remaining_indices = [index for index in peers.index if index not in selected_indices]
        selected_indices.extend(remaining_indices[: peer_count - len(selected_indices)])

    return peers.loc[selected_indices].reset_index(drop=True)
