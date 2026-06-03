"""
score.py - Congressional trade signal scoring module.

Importable standalone module. All scores use only data available
as of filing_date (no lookahead bias).
"""

from __future__ import annotations

import re
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# GICS sector -> oversight committee keyword fragments (lowercase)
# ---------------------------------------------------------------------------
SECTOR_COMMITTEES: dict[str, list[str]] = {
    "Health Care": [
        "health, education, labor",
        "senate help",
        "senate finance",
        "energy and commerce",
        "labor, health and human",
        "appropriations",
    ],
    "Defense": [
        "armed services",
        "appropriations",
    ],
    "Industrials": [
        "armed services",
        "transportation and infrastructure",
        "commerce, science",
    ],
    "Information Technology": [
        "commerce, science",
        "science, space",
        "energy and commerce",
        "judiciary",
    ],
    "Communication Services": [
        "commerce, science",
        "energy and commerce",
        "judiciary",
    ],
    "Financials": [
        "banking, housing",
        "financial services",
        "senate finance",
    ],
    "Energy": [
        "energy and natural resources",
        "environment and public works",
        "energy and commerce",
        "natural resources",
    ],
    "Utilities": [
        "energy and natural resources",
        "energy and commerce",
    ],
    "Consumer Discretionary": [
        "commerce, science",
        "energy and commerce",
    ],
    "Consumer Staples": [
        "agriculture",
        "commerce, science",
    ],
    "Materials": [
        "environment and public works",
        "natural resources",
        "agriculture",
    ],
    "Real Estate": [
        "banking, housing",
        "financial services",
    ],
}

_AMOUNT_RE = re.compile(r"\$?([\d,]+)")


def _parse_lower_bound(amount_str: str) -> int:
    """Extract the lower-bound dollar value from a disclosure amount string."""
    if not amount_str or (isinstance(amount_str, float) and np.isnan(amount_str)):
        return 0
    m = _AMOUNT_RE.search(str(amount_str))
    if not m:
        return 0
    try:
        return int(m.group(1).replace(",", ""))
    except ValueError:
        return 0


def _committee_relevance(committees: list[str], sector: Optional[str]) -> int:
    """Returns 0, 2, or 5 based on how directly the committees relate to the sector."""
    if not sector or not committees:
        return 0
    keywords = SECTOR_COMMITTEES.get(sector, [])
    if not keywords:
        return 0
    joined = " ".join(c.lower() for c in committees)
    for kw in keywords:
        if kw in joined:
            return 5
    # Tangential: first keyword word only
    for kw in keywords:
        if kw.split(",")[0].split()[0] in joined:
            return 2
    return 0


def score_congressional_signal(
    amount_str: str,
    legislator_committees: list[str],
    sector: Optional[str],
    cluster_count: int,
) -> int:
    """Component 1: Congressional Signal Quality (0-25)."""
    low = _parse_lower_bound(amount_str)

    if low >= 500_000:
        size_pts = 25
    elif low >= 100_000:
        size_pts = 20
    elif low >= 50_000:
        size_pts = 14
    elif low >= 15_000:
        size_pts = 8
    else:
        size_pts = 3

    committee_pts = _committee_relevance(legislator_committees, sector)

    if cluster_count >= 3:
        cluster_bonus = 5
    elif cluster_count >= 2:
        cluster_bonus = 3
    else:
        cluster_bonus = 0

    return min(25, size_pts + committee_pts + cluster_bonus)


def score_related_persons(
    ticker: str,
    legislator: str,
    filing_date: pd.Timestamp,
    all_trades: pd.DataFrame,
) -> int:
    """Component 2: Related Persons Activity (0-15)."""
    window = pd.Timedelta(days=30)
    mask = (
        (all_trades["ticker"] == ticker)
        & (all_trades["legislator"] == legislator)
        & (all_trades["filing_date"] >= filing_date - window)
        & (all_trades["filing_date"] <= filing_date + window)
    )
    related = all_trades[mask]
    if related.empty:
        return 0

    owners = related["owner"].str.lower().fillna("self")
    has_related = owners.str.contains("spouse|child|dependent|joint", na=False).any()
    has_self = owners.str.contains("self|himself|herself", na=False).any()

    if has_related and not has_self:
        return 12  # Distancing pattern
    if has_related and has_self:
        return 8   # Both buy
    return 0


def score_fundamentals(fundamentals: Optional[dict]) -> int:
    """Component 3: Fundamentals (0-25). Returns 12 (neutral) if data unavailable."""
    if fundamentals is None:
        return 12

    pts = 0

    rev = fundamentals.get("revenue_growth")
    if rev is not None:
        if rev > 0.15:
            pts += 10
        elif rev > 0.05:
            pts += 6
        elif rev > 0.0:
            pts += 3

    pe = fundamentals.get("pe_ratio")
    if pe is not None and pe > 0:
        if pe < 20:
            pts += 8
        elif pe < 35:
            pts += 5
        elif pe < 50:
            pts += 2

    de = fundamentals.get("debt_equity")
    if de is not None and de >= 0:
        if de < 0.5:
            pts += 7
        elif de < 1.5:
            pts += 4
        else:
            pts += 1

    return min(25, pts)


def score_catalyst(
    filing_date: pd.Timestamp,
    earnings_dates: list[pd.Timestamp],
) -> int:
    """Component 4: Upcoming Catalysts (0-20). Uses earnings proximity as proxy."""
    if not earnings_dates:
        return 0
    for edate in sorted(earnings_dates):
        days = (edate - filing_date).days
        if 0 <= days <= 30:
            return 15
        if 0 <= days <= 60:
            return 10
        if 0 <= days <= 90:
            return 5
    return 0


def score_news_alignment(
    filing_date: pd.Timestamp,
    close_series: Optional[pd.Series],
) -> int:
    """Component 5: News Alignment (0-15). Uses 30-day price momentum as sentiment proxy."""
    if close_series is None or close_series.empty:
        return 4

    window = close_series.loc[
        (close_series.index >= filing_date - timedelta(days=35))
        & (close_series.index <= filing_date)
    ]
    if len(window) < 3:
        return 4

    p0 = float(window.iloc[0])
    p1 = float(window.iloc[-1])
    if p0 <= 0:
        return 4

    mom = (p1 - p0) / p0
    if mom > 0.05:
        return 12
    if mom >= 0:
        return 8
    if mom >= -0.05:
        return 4
    return 1


def score_trade(
    trade: dict,
    all_trades: pd.DataFrame,
    legislator_committees: list[str],
    sector: Optional[str],
    cluster_count: int,
    fundamentals: Optional[dict],
    earnings_dates: list[pd.Timestamp],
    close_series: Optional[pd.Series],
) -> dict:
    """
    Score a single congressional buy disclosure.
    Returns a copy of `trade` with all score fields added.
    """
    c1 = score_congressional_signal(
        trade.get("amount", ""),
        legislator_committees,
        sector,
        cluster_count,
    )
    c2 = score_related_persons(
        trade["ticker"],
        trade["legislator"],
        trade["filing_date"],
        all_trades,
    )
    c3 = score_fundamentals(fundamentals)
    c4 = score_catalyst(trade["filing_date"], earnings_dates)
    c5 = score_news_alignment(trade["filing_date"], close_series)

    return {
        **trade,
        "score": c1 + c2 + c3 + c4 + c5,
        "score_c1_signal": c1,
        "score_c2_related": c2,
        "score_c3_fundamentals": c3,
        "score_c4_catalyst": c4,
        "score_c5_news": c5,
    }
