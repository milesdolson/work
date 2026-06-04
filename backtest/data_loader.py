"""
data_loader.py - Data fetching, cleaning, and caching.

Sources:
  1. Congressional trades: wallstreetlocal (primary) or house/senate-stock-watcher JSON
  2. Senate EFTS API for additional coverage
  3. Price data: yfinance with SQLite cache
  4. Committee membership: unitedstates/congress-legislators YAML
  5. Fundamentals: yfinance quarterly financials
"""

from __future__ import annotations

import difflib
import io
import json
import re
import sqlite3
import time
import warnings
from datetime import timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yaml
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_DIR = Path("data")

# ---------------------------------------------------------------------------
# Data source URLs in priority order
# ---------------------------------------------------------------------------
HOUSE_URLS = [
    "https://raw.githubusercontent.com/leftmove/wallstreetlocal/main/apps/server/data/house.csv",
    "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json",
    "https://raw.githubusercontent.com/jermwatt/house-stock-watcher/main/data/all_transactions.json",
    "https://raw.githubusercontent.com/timothycarambat/house-stock-watcher-data/master/aggregate/all_transactions.json",
]
SENATE_URLS = [
    # GitHub-hosted aggregate file - most reliable free source
    "https://raw.githubusercontent.com/timothycarambat/senate-stock-watcher-data/master/aggregate/all_transactions.json",
    "https://raw.githubusercontent.com/leftmove/wallstreetlocal/main/apps/server/data/senate.csv",
    "https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/aggregate/all_transactions.json",
    "https://senate-stock-watcher-data.s3.amazonaws.com/daily/all_transactions.json",
    "https://raw.githubusercontent.com/jermwatt/senate-stock-watcher/main/data/all_transactions.json",
]
SENATE_EFTS_URL = (
    "https://efts.senate.gov/LATEST/search.json"
    "?q=%22stock%22&dateRange=custom&fromDate=2023-01-01&toDate=2024-12-31&hits.hits.total.value=1000"
)
COMMITTEES_URL = (
    "https://raw.githubusercontent.com/unitedstates/congress-legislators"
    "/main/committees-current.yaml"
)
COMMITTEE_MEMBERSHIP_URL = (
    "https://raw.githubusercontent.com/unitedstates/congress-legislators"
    "/main/committee-membership-current.yaml"
)
LEGISLATORS_CURRENT_URL = (
    "https://raw.githubusercontent.com/unitedstates/congress-legislators"
    "/main/legislators-current.yaml"
)
LEGISLATORS_HISTORICAL_URL = (
    "https://raw.githubusercontent.com/unitedstates/congress-legislators"
    "/main/legislators-historical.yaml"
)

# ---------------------------------------------------------------------------
# SQLite cache
# ---------------------------------------------------------------------------

def init_db(db_path: str = "price_cache.db") -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            ticker    TEXT NOT NULL,
            date      TEXT NOT NULL,
            open      REAL,
            close     REAL,
            PRIMARY KEY (ticker, date)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fundamentals (
            ticker          TEXT NOT NULL,
            year            INTEGER NOT NULL,
            quarter         INTEGER NOT NULL,
            revenue_growth  REAL,
            pe_ratio        REAL,
            debt_equity     REAL,
            PRIMARY KEY (ticker, year, quarter)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS earnings_dates (
            ticker TEXT NOT NULL,
            date   TEXT NOT NULL,
            PRIMARY KEY (ticker, date)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sectors (
            ticker TEXT PRIMARY KEY,
            sector TEXT
        )
    """)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Congressional trade data
# ---------------------------------------------------------------------------

def _parse_date(val) -> Optional[pd.Timestamp]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip()
    from datetime import datetime as _dt
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%d-%b-%Y", "%m-%d-%Y", "%Y/%m/%d"):
        try:
            return pd.Timestamp(_dt.strptime(s, fmt))
        except (ValueError, AttributeError):
            pass
    try:
        return pd.Timestamp(s)
    except Exception:
        return None


def _clean_ticker(val) -> Optional[str]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    t = str(val).strip().upper().replace("/", "-")
    if not t or len(t) > 10:
        return None
    if not re.match(r"^[A-Z][A-Z0-9.\-]{0,9}$", t):
        return None
    # Skip broad ETFs - we want individual stock signals
    if t in {"SPY", "QQQ", "VOO", "VTI", "IVV", "IWM", "GLD", "SLV", "TLT",
             "HYG", "LQD", "EEM", "VEA", "VXUS", "BND", "AGG"}:
        return None
    return t


def _normalize_columns(df: pd.DataFrame, chamber: str) -> pd.DataFrame:
    """
    Detect and rename columns to standard schema regardless of source format.
    Standard: filing_date, transaction_date, ticker, amount, transaction_type,
              owner, legislator, party, state, asset_description
    """
    cols_lower = {c.lower().strip().replace(" ", "_"): c for c in df.columns}

    mapping = {}
    for target, candidates in [
        ("filing_date",       ["disclosure_date", "filing_date", "filed", "date_received", "disclosure"]),
        ("transaction_date",  ["transaction_date", "date_transacted", "transactiondate", "date"]),
        ("ticker",            ["ticker", "stock_ticker", "symbol", "stock"]),
        ("amount",            ["amount", "trade_size", "value", "range"]),
        ("transaction_type",  ["type", "transaction_type", "trade_type", "transactiontype"]),
        ("owner",             ["owner", "filer", "filing_for"]),
        ("legislator",        ["representative", "senator", "name", "representative_name",
                                "senator_name", "fullname", "full_name"]),
        ("party",             ["party"]),
        ("state",             ["state"]),
        ("asset_description", ["asset_description", "description", "asset_name",
                                "company_name", "company", "name_of_security"]),
    ]:
        for c in candidates:
            if c in cols_lower:
                mapping[cols_lower[c]] = target
                break

    df = df.rename(columns=mapping)
    for col in ["filing_date", "ticker", "amount", "transaction_type", "owner", "legislator"]:
        if col not in df.columns:
            df[col] = None
    if "party" not in df.columns:
        df["party"] = None
    if "state" not in df.columns:
        df["state"] = None
    df["chamber"] = chamber
    return df


def _load_json_source(url: str) -> Optional[pd.DataFrame]:
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # Handle both list and {"transactions": [...]} formats
        if isinstance(data, list):
            return pd.DataFrame(data)
        for key in ("transactions", "data", "results", "disclosures"):
            if key in data:
                return pd.DataFrame(data[key])
        return pd.DataFrame(data)
    except Exception:
        return None


def _load_csv_source(url: str) -> Optional[pd.DataFrame]:
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return pd.read_csv(io.StringIO(resp.text), low_memory=False)
    except Exception:
        return None


def _try_sources(urls: list[str], chamber: str, cache_path: Path) -> Optional[pd.DataFrame]:
    # Check local cache first
    if cache_path.exists():
        try:
            df = pd.read_csv(cache_path, low_memory=False)
            print(f"  {chamber}: {len(df)} records from local cache")
            return df
        except Exception:
            pass

    for url in urls:
        df = None
        if url.endswith(".csv"):
            df = _load_csv_source(url)
        elif url.endswith(".json"):
            df = _load_json_source(url)
        else:
            # Try CSV first, then JSON
            df = _load_csv_source(url)
            if df is None:
                df = _load_json_source(url)

        if df is not None and not df.empty:
            df.to_csv(cache_path, index=False)
            print(f"  {chamber}: {len(df)} records from {url}")
            return df
        print(f"  {chamber}: failed {url}")

    return None


def _load_senate_efts(start: str = "2023-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    """Load Senate trades from the EFTS search API (paginated)."""
    records = []
    page = 1
    while True:
        try:
            url = (
                f"https://efts.senate.gov/LATEST/search.json"
                f"?q=%22stock%22&dateRange=custom&fromDate={start}&toDate={end}"
                f"&hits.hits._source=true&from={len(records)}"
            )
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])
            if not hits:
                break
            for hit in hits:
                records.append(hit.get("_source", {}))
            if len(hits) < 10:
                break
        except Exception:
            break
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def load_congressional_trades(backtest_start="2023-01-01", backtest_end="2024-12-31") -> pd.DataFrame:
    """
    Load, clean, and return all congressional BUY disclosures in the backtest window.
    Prints a data quality / provenance report.
    """
    DATA_DIR.mkdir(exist_ok=True)
    frames = []

    print("\n=== Congressional Trade Data Provenance ===")

    for chamber, urls in [("house", HOUSE_URLS), ("senate", SENATE_URLS)]:
        cache_path = DATA_DIR / f"{chamber}_raw.csv"
        df = _try_sources(urls, chamber, cache_path)
        if df is None:
            print(f"  WARNING: could not load {chamber} data from any source")
            continue
        df = _normalize_columns(df, chamber)
        # Parse filing and transaction dates up front so fallback works.
        df["filing_date"] = df["filing_date"].apply(_parse_date)
        if "transaction_date" in df.columns:
            df["transaction_date"] = df["transaction_date"].apply(_parse_date)
        # If source has no filing_date (e.g. senate-stock-watcher only has
        # transaction_date), use transaction_date + 30 days as a proxy to
        # approximate the STOCK Act disclosure lag.
        if df["filing_date"].isna().all() and "transaction_date" in df.columns:
            df["filing_date"] = df["transaction_date"].apply(
                lambda d: d + pd.Timedelta(days=30) if pd.notna(d) else None
            )
            print(f"  {chamber}: filing_date not in source; using transaction_date + 30d as proxy")
        frames.append(df)

    if not frames:
        raise RuntimeError("No congressional trade data available from any source.")

    combined = pd.concat(frames, ignore_index=True)
    raw_count = len(combined)

    # Parse dates and tickers
    combined["filing_date"] = combined["filing_date"].apply(_parse_date)
    combined["ticker"] = combined["ticker"].apply(_clean_ticker)

    # Filter: buys only, valid ticker, valid date
    combined = combined.dropna(subset=["filing_date", "ticker"])
    buy_mask = combined["transaction_type"].fillna("").str.lower().str.contains(
        "purchase|buy|bought", na=False
    )
    combined = combined[buy_mask]

    # Restrict to backtest window
    combined = combined[
        (combined["filing_date"] >= pd.Timestamp(backtest_start))
        & (combined["filing_date"] <= pd.Timestamp(backtest_end))
    ]
    combined = combined.sort_values("filing_date").reset_index(drop=True)

    print(f"\n  Raw records (all types): {raw_count}")
    print(f"  Buy disclosures in {backtest_start} to {backtest_end}: {len(combined)}")
    print(f"  Date range: {combined['filing_date'].min().date()} to {combined['filing_date'].max().date()}")
    print(f"  Unique legislators: {combined['legislator'].nunique()}")
    print(f"  Unique tickers: {combined['ticker'].nunique()}")
    print(f"  House: {(combined['chamber']=='house').sum()} | Senate: {(combined['chamber']=='senate').sum()}")

    if len(combined) == 0:
        raise RuntimeError("No qualifying trades after filtering.")

    return combined


# ---------------------------------------------------------------------------
# Price data (SQLite-cached yfinance)
# ---------------------------------------------------------------------------

def batch_fetch_prices(
    tickers: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    conn: sqlite3.Connection,
) -> dict[str, pd.DataFrame]:
    """
    Download price histories for all tickers in bulk.
    Checks SQLite cache first; only fetches uncached tickers.
    Returns {ticker: DataFrame(index=date, columns=[open, close])}.
    """
    price_cache: dict[str, pd.DataFrame] = {}
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    # Load from cache
    to_fetch = []
    for t in tickers:
        rows = conn.execute(
            "SELECT date, open, close FROM prices WHERE ticker=? AND date>=? AND date<=?",
            (t, start_str, end_str),
        ).fetchall()
        if rows and len(rows) > 10:
            df = pd.DataFrame(rows, columns=["date", "open", "close"])
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            price_cache[t] = df
        else:
            to_fetch.append(t)

    if not to_fetch:
        return price_cache

    print(f"  Downloading prices for {len(to_fetch)} tickers (in batches of 50)...")

    BATCH = 50
    for i in range(0, len(to_fetch), BATCH):
        chunk = to_fetch[i: i + BATCH]
        pct = int((i / len(to_fetch)) * 100)
        print(f"    {pct}% ({i}/{len(to_fetch)})...", end="\r", flush=True)
        try:
            raw = yf.download(
                chunk,
                start=start_str,
                end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
                group_by="ticker",
            )
            if raw.empty:
                continue

            # Single ticker: flat columns; multi-ticker: (metric, ticker) multi-index
            if len(chunk) == 1:
                t = chunk[0]
                if "Open" in raw.columns and "Close" in raw.columns:
                    df = raw[["Open", "Close"]].copy()
                    df.columns = ["open", "close"]
                    df.index = df.index.tz_localize(None)
                    df = df.dropna()
                    _cache_price_df(t, df, conn)
                    price_cache[t] = df
            else:
                for t in chunk:
                    try:
                        if ("Open", t) in raw.columns and ("Close", t) in raw.columns:
                            df = raw[["Open", "Close"]].xs(t, axis=1, level=1)
                        elif "Open" in raw.columns.get_level_values(0):
                            df = pd.DataFrame({
                                "open": raw["Open"][t],
                                "close": raw["Close"][t],
                            })
                        else:
                            continue
                        df.columns = ["open", "close"]
                        df.index = df.index.tz_localize(None)
                        df = df.dropna()
                        if not df.empty:
                            _cache_price_df(t, df, conn)
                            price_cache[t] = df
                    except Exception:
                        continue
        except Exception as e:
            # Fallback: individual fetches for this chunk
            for t in chunk:
                try:
                    time.sleep(0.2)
                    df = yf.Ticker(t).history(
                        start=start_str,
                        end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
                        auto_adjust=True,
                    )
                    if not df.empty:
                        df = df[["Open", "Close"]].copy()
                        df.columns = ["open", "close"]
                        df.index = df.index.tz_localize(None)
                        df = df.dropna()
                        _cache_price_df(t, df, conn)
                        price_cache[t] = df
                except Exception:
                    pass
    print()  # newline after progress
    return price_cache


def _cache_price_df(ticker: str, df: pd.DataFrame, conn: sqlite3.Connection) -> None:
    rows = [
        (ticker, d.strftime("%Y-%m-%d"), float(r["open"]), float(r["close"]))
        for d, r in df.iterrows()
        if not (np.isnan(r["open"]) or np.isnan(r["close"]))
    ]
    if rows:
        conn.executemany(
            "INSERT OR REPLACE INTO prices (ticker, date, open, close) VALUES (?,?,?,?)",
            rows,
        )
        conn.commit()


def get_open_price(
    ticker: str,
    date: pd.Timestamp,
    price_cache: dict[str, pd.DataFrame],
) -> Optional[float]:
    """Return market-open price on or after `date` using cached data."""
    if ticker not in price_cache:
        return None
    df = price_cache[ticker]
    valid = df[df.index >= date]
    if valid.empty:
        return None
    return float(valid["open"].iloc[0])


def get_close_price(
    ticker: str,
    date: pd.Timestamp,
    price_cache: dict[str, pd.DataFrame],
) -> Optional[float]:
    """Return most recent close price on or before `date`."""
    if ticker not in price_cache:
        return None
    df = price_cache[ticker]
    valid = df[df.index <= date]
    if valid.empty:
        return None
    return float(valid["close"].iloc[-1])


# ---------------------------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------------------------

def get_fundamentals(
    ticker: str,
    as_of_date: pd.Timestamp,
    conn: sqlite3.Connection,
) -> Optional[dict]:
    """Fetch or compute fundamentals dict for ticker as of as_of_date."""
    y, q = as_of_date.year, (as_of_date.month - 1) // 3

    # Check cache (look up to 2 quarters back)
    for offset in range(3):
        yy, qq = y, q - offset
        while qq < 0:
            qq += 4
            yy -= 1
        row = conn.execute(
            "SELECT revenue_growth, pe_ratio, debt_equity FROM fundamentals "
            "WHERE ticker=? AND year=? AND quarter=?",
            (ticker, yy, qq),
        ).fetchone()
        if row:
            return {"revenue_growth": row[0], "pe_ratio": row[1], "debt_equity": row[2]}

    try:
        time.sleep(0.2)
        t = yf.Ticker(ticker)

        revenue_growth = None
        qf = t.quarterly_financials
        if qf is not None and not qf.empty:
            rev_keys = [k for k in qf.index if "revenue" in str(k).lower()]
            if rev_keys:
                rev = qf.loc[rev_keys[0]].sort_index(ascending=True)
                if len(rev) >= 5:
                    r_now = float(rev.iloc[-1])
                    r_prev = float(rev.iloc[-5])
                    if r_prev and r_prev > 0:
                        revenue_growth = (r_now - r_prev) / abs(r_prev)

        info = t.info
        pe = info.get("trailingPE") or info.get("forwardPE")
        pe = float(pe) if pe and pe == pe else None

        debt_equity = None
        bs = t.quarterly_balance_sheet
        if bs is not None and not bs.empty:
            eq_keys = [k for k in bs.index if "stockholder" in str(k).lower()
                       or "equity" in str(k).lower()]
            dt_keys = [k for k in bs.index if "long" in str(k).lower()
                       and "debt" in str(k).lower()]
            if eq_keys and dt_keys:
                eq = float(bs.loc[eq_keys[0]].iloc[0] or 0)
                dt = float(bs.loc[dt_keys[0]].iloc[0] or 0)
                if eq > 0:
                    debt_equity = dt / eq

        result = {"revenue_growth": revenue_growth, "pe_ratio": pe, "debt_equity": debt_equity}
        conn.execute(
            "INSERT OR REPLACE INTO fundamentals "
            "(ticker, year, quarter, revenue_growth, pe_ratio, debt_equity) VALUES (?,?,?,?,?,?)",
            (ticker, y, q, revenue_growth, pe, debt_equity),
        )
        conn.commit()
        return result
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Earnings dates
# ---------------------------------------------------------------------------

def get_earnings_dates(ticker: str, conn: sqlite3.Connection) -> list[pd.Timestamp]:
    """Return historical earnings dates for ticker."""
    rows = conn.execute(
        "SELECT date FROM earnings_dates WHERE ticker=? ORDER BY date", (ticker,)
    ).fetchall()
    if rows:
        return [pd.Timestamp(r[0]) for r in rows]

    try:
        time.sleep(0.15)
        t = yf.Ticker(ticker)
        dates = []

        ed = getattr(t, "earnings_dates", None)
        if ed is not None and not ed.empty:
            dates = [d.tz_localize(None) if d.tzinfo else pd.Timestamp(d) for d in ed.index]
        else:
            cal = t.calendar
            if cal is not None and not cal.empty:
                date_cols = [c for c in cal.columns if "earnings" in str(c).lower()]
                if date_cols:
                    val = cal[date_cols[0]].iloc[0]
                    if val:
                        dates = [pd.Timestamp(val)]

        if dates:
            conn.executemany(
                "INSERT OR REPLACE INTO earnings_dates (ticker, date) VALUES (?,?)",
                [(ticker, d.strftime("%Y-%m-%d")) for d in dates],
            )
            conn.commit()
        return dates
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Sector
# ---------------------------------------------------------------------------

def get_sector(ticker: str, conn: sqlite3.Connection) -> Optional[str]:
    row = conn.execute("SELECT sector FROM sectors WHERE ticker=?", (ticker,)).fetchone()
    if row:
        return row[0]
    try:
        time.sleep(0.1)
        sector = yf.Ticker(ticker).info.get("sector")
        if sector:
            conn.execute("INSERT OR REPLACE INTO sectors (ticker, sector) VALUES (?,?)", (ticker, sector))
            conn.commit()
        return sector
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Committee membership
# ---------------------------------------------------------------------------

def load_committee_membership() -> dict[str, list[str]]:
    """
    Return {legislator_name_lower: [committee_name, ...]} from congress-legislators YAML.
    """
    print("\n  Loading committee membership data...")
    DATA_DIR.mkdir(exist_ok=True)

    try:
        # 1. Load committee IDs -> names
        resp = requests.get(COMMITTEES_URL, timeout=30)
        resp.raise_for_status()
        committees_list = yaml.safe_load(resp.text) or []
        id_to_name: dict[str, str] = {}
        for c in committees_list:
            name = c.get("name", "")
            for id_field in ("thomas_id", "senate_id", "house_committee_id"):
                cid = c.get(id_field)
                if cid:
                    id_to_name[cid] = name

        # 2. Load committee membership {committee_id: [{bioguide, name, ...}]}
        resp2 = requests.get(COMMITTEE_MEMBERSHIP_URL, timeout=30)
        resp2.raise_for_status()
        membership = yaml.safe_load(resp2.text) or {}

        # 3. Load legislator names for bioguide -> full_name mapping
        bioguide_to_name: dict[str, str] = {}
        for url in [LEGISLATORS_CURRENT_URL, LEGISLATORS_HISTORICAL_URL]:
            try:
                resp3 = requests.get(url, timeout=30)
                resp3.raise_for_status()
                for leg in (yaml.safe_load(resp3.text) or []):
                    bid = leg.get("id", {}).get("bioguide")
                    nm = leg.get("name", {})
                    official = nm.get("official_full") or f"{nm.get('first','')} {nm.get('last','')}".strip()
                    if bid and official:
                        bioguide_to_name[bid] = official
            except Exception:
                continue

        # 4. Build name -> committees
        name_to_committees: dict[str, list[str]] = {}
        for committee_id, members in membership.items():
            committee_name = id_to_name.get(committee_id, committee_id)
            for member in (members or []):
                bid = member.get("bioguide")
                # Also try name directly from membership
                raw_name = member.get("name", "")
                name = bioguide_to_name.get(bid, "") if bid else raw_name
                if not name:
                    name = raw_name
                if name:
                    name_to_committees.setdefault(name.lower(), []).append(committee_name)

        print(f"  Committees loaded for {len(name_to_committees)} legislators")
        return name_to_committees

    except Exception as e:
        print(f"  WARNING: committee data failed: {e}. Scores will use 0 for committee component.")
        return {}


def fuzzy_committee_lookup(
    legislator_name: str,
    committee_map: dict[str, list[str]],
) -> list[str]:
    """Match a legislator name to their committee memberships using fuzzy lookup."""
    if not legislator_name or not committee_map:
        return []
    name_lower = str(legislator_name).lower().strip()

    if name_lower in committee_map:
        return committee_map[name_lower]

    # Last name match
    last = name_lower.split()[-1] if name_lower.split() else ""
    if len(last) > 3:
        candidates = [k for k in committee_map if last in k.split()]
        if len(candidates) == 1:
            return committee_map[candidates[0]]

    # Fuzzy match
    matches = difflib.get_close_matches(name_lower, committee_map.keys(), n=1, cutoff=0.65)
    if matches:
        return committee_map[matches[0]]

    return []
