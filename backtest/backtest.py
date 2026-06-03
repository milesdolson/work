"""
backtest.py - Government Signal Investment Strategy Backtester.

Usage: python backtest.py

Evaluates congressional trade signals against VOO and VXUS benchmarks
over Jan 1 2023 - Dec 31 2024.
"""

from __future__ import annotations

import sys
import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

import data_loader as dl
import score as sc

warnings.filterwarnings("ignore")

RESULTS_DIR = Path("results")
START = pd.Timestamp("2023-01-01")
END = pd.Timestamp("2024-12-31")
PRICE_START = START - timedelta(days=40)   # for 30-day momentum lookback
PRICE_END = END + timedelta(days=100)      # for 90-day exit prices
INITIAL_VALUE = 100_000.0
HOLD_DAYS = 90
MIN_SCORE = 50
MAX_POSITIONS = 5
RISK_FREE_RATE = 0.05  # annual


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def compute_cluster_counts(trades_df: pd.DataFrame) -> dict[int, int]:
    """
    For each trade index, count how many OTHER legislators bought the same
    ticker within a 30-day window around its filing_date.
    Returns {index: count}.
    """
    counts: dict[int, int] = {}
    for ticker, group in trades_df.groupby("ticker"):
        if len(group) < 2:
            for idx in group.index:
                counts[idx] = 0
            continue
        dates = group["filing_date"]
        for idx in group.index:
            d = dates[idx]
            window = (dates >= d - timedelta(days=30)) & (dates <= d + timedelta(days=30))
            counts[idx] = int(window.sum()) - 1  # exclude self
    return counts


def run_scoring_pipeline(
    trades_df: pd.DataFrame,
    conn,
    price_cache: dict,
    committee_map: dict,
    missing_tickers: list,
) -> pd.DataFrame:
    """Score every qualifying trade. Returns DataFrame with score columns."""
    print(f"\n  Scoring {len(trades_df)} trades...")

    # Pre-fetch sectors, earnings, fundamentals for unique tickers
    unique_tickers = list(trades_df["ticker"].dropna().unique())
    sectors: dict[str, str] = {}
    earnings: dict[str, list] = {}
    fundamentals_cache: dict[str, dict] = {}

    print(f"  Fetching metadata for {len(unique_tickers)} tickers...")
    for i, ticker in enumerate(unique_tickers):
        if i % 25 == 0:
            print(f"    {i}/{len(unique_tickers)}...", end="\r", flush=True)
        sectors[ticker] = dl.get_sector(ticker, conn)
        earnings[ticker] = dl.get_earnings_dates(ticker, conn)
    print()

    cluster_counts = compute_cluster_counts(trades_df)
    scored_records = []

    for idx, row in trades_df.iterrows():
        ticker = row["ticker"]
        filing_date = row["filing_date"]

        # Skip if no price data
        if ticker not in price_cache:
            missing_tickers.append(ticker)
            continue

        trade_dict = row.to_dict()
        committees = dl.fuzzy_committee_lookup(str(row.get("legislator", "")), committee_map)
        sector = sectors.get(ticker)

        # Fundamentals: fetch and cache per ticker (not per filing date for speed)
        if ticker not in fundamentals_cache:
            fundamentals_cache[ticker] = dl.get_fundamentals(ticker, filing_date, conn)
        fund = fundamentals_cache[ticker]

        future_earnings = [e for e in earnings.get(ticker, []) if e >= filing_date]
        close_series = price_cache[ticker]["close"]
        cluster_count = cluster_counts.get(idx, 0)

        scored = sc.score_trade(
            trade=trade_dict,
            all_trades=trades_df,
            legislator_committees=committees,
            sector=sector,
            cluster_count=cluster_count,
            fundamentals=fund,
            earnings_dates=future_earnings,
            close_series=close_series,
        )
        scored_records.append(scored)

    result = pd.DataFrame(scored_records)
    result = result.sort_values("filing_date").reset_index(drop=True)

    qualify = (result["score"] >= MIN_SCORE).sum()
    print(f"  Scored {len(result)} trades; {qualify} qualify at score >= {MIN_SCORE}")
    return result


# ---------------------------------------------------------------------------
# Backtest simulation
# ---------------------------------------------------------------------------

def run_backtest(
    scored_df: pd.DataFrame,
    price_cache: dict,
    missing_tickers: list,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Weekly-rebalanced long-only backtest.

    Entry: open price the day AFTER filing_date.
    Exit: open price 90 calendar days after entry.
    Portfolio: top 5 qualifying signals per week, equal weight.

    Returns (portfolio_history_df, closed_positions_df).
    """
    qualifying = scored_df[scored_df["score"] >= MIN_SCORE].copy()
    qualifying = qualifying.sort_values("filing_date").reset_index(drop=True)
    print(f"\n  Simulating {len(qualifying)} qualifying trades...")

    cash = INITIAL_VALUE
    open_positions: list[dict] = []
    closed_positions: list[dict] = []
    portfolio_history: list[dict] = []

    weeks = pd.date_range(START, END, freq="W-MON")

    for week_start in weeks:
        week_end = week_start + pd.Timedelta(days=6)

        # 1. Close matured positions
        still_open = []
        for pos in open_positions:
            if pos["exit_date"] <= week_start:
                exit_price = dl.get_open_price(pos["ticker"], pos["exit_date"], price_cache)
                if exit_price is None:
                    exit_price = dl.get_close_price(pos["ticker"], pos["exit_date"], price_cache)
                if exit_price is None:
                    exit_price = pos["entry_price"]  # flat return as fallback
                ret = (exit_price - pos["entry_price"]) / pos["entry_price"]
                cash += pos["position_value"] * (1 + ret)
                pos["exit_price"] = exit_price
                pos["actual_return"] = ret
                closed_positions.append(pos)
            else:
                still_open.append(pos)
        open_positions = still_open

        # 2. Mark-to-market portfolio value
        pos_value = 0.0
        for pos in open_positions:
            price = dl.get_close_price(pos["ticker"], week_start, price_cache) or pos["entry_price"]
            pos_value += pos["shares"] * price
        total_value = cash + pos_value

        portfolio_history.append({"date": week_start, "value": total_value})

        # 3. Fill vacant slots with new trades
        slots = MAX_POSITIONS - len(open_positions)
        if slots <= 0:
            continue

        new_trades = qualifying[
            (qualifying["filing_date"] >= week_start)
            & (qualifying["filing_date"] <= week_end)
        ].sort_values("score", ascending=False)

        held = {p["ticker"] for p in open_positions}
        new_trades = new_trades[~new_trades["ticker"].isin(held)]
        new_trades = new_trades.head(slots)

        for _, trade in new_trades.iterrows():
            entry_date = trade["filing_date"] + timedelta(days=1)
            entry_price = dl.get_open_price(trade["ticker"], entry_date, price_cache)
            if entry_price is None or entry_price <= 0:
                continue

            # Equal weight: 1/5 of current portfolio value
            position_size = min(total_value / MAX_POSITIONS, cash)
            if position_size <= 0:
                continue

            cash -= position_size
            open_positions.append({
                **trade.to_dict(),
                "entry_date": entry_date,
                "entry_price": entry_price,
                "exit_date": entry_date + timedelta(days=HOLD_DAYS),
                "position_value": position_size,
                "shares": position_size / entry_price,
            })

    # Close remaining positions at end
    for pos in open_positions:
        exit_price = (
            dl.get_close_price(pos["ticker"], END, price_cache)
            or pos["entry_price"]
        )
        ret = (exit_price - pos["entry_price"]) / pos["entry_price"]
        cash += pos["position_value"] * (1 + ret)
        pos["exit_price"] = exit_price
        pos["actual_return"] = ret
        closed_positions.append(pos)

    portfolio_history.append({"date": END, "value": cash})

    ph = pd.DataFrame(portfolio_history)
    cp = pd.DataFrame(closed_positions) if closed_positions else pd.DataFrame()
    print(f"  Simulation complete: {len(closed_positions)} positions closed")
    return ph, cp


def build_benchmark(ticker: str, price_cache: dict, alloc: float = 1.0) -> pd.DataFrame:
    """Buy-and-hold benchmark: alloc * INITIAL_VALUE into ticker on START date."""
    if ticker not in price_cache:
        raise RuntimeError(f"No price data for benchmark {ticker}")
    hist = price_cache[ticker]
    entry = hist[hist.index >= START]
    if entry.empty:
        raise RuntimeError(f"No data for {ticker} from {START}")
    entry_price = float(entry["open"].iloc[0])
    shares = (INITIAL_VALUE * alloc) / entry_price
    weeks = pd.date_range(START, END, freq="W-MON")
    rows = []
    for w in weeks:
        p = dl.get_close_price(ticker, w, {ticker: hist}) or entry_price
        rows.append({"date": w, "value": shares * p})
    rows.append({"date": END, "value": shares * float(hist[hist.index <= END]["close"].iloc[-1])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(
    ph: pd.DataFrame,
    closed: pd.DataFrame,
    bench_voo: pd.DataFrame,
) -> dict:
    vals = ph["value"].values.astype(float)
    weekly_ret = np.diff(vals) / vals[:-1]

    total_ret = (vals[-1] - vals[0]) / vals[0]
    n_years = (ph["date"].iloc[-1] - ph["date"].iloc[0]).days / 365.25
    ann_ret = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1

    rf_weekly = (1 + RISK_FREE_RATE) ** (1 / 52) - 1
    excess = weekly_ret - rf_weekly
    vol = np.std(excess)
    sharpe = (np.mean(excess) / vol * np.sqrt(52)) if vol > 0 else 0

    downside = excess[excess < 0]
    d_vol = np.std(downside)
    sortino = (np.mean(excess) / d_vol * np.sqrt(52)) if d_vol > 0 else 0

    cummax = np.maximum.accumulate(vals)
    dd = (vals - cummax) / cummax
    max_dd = float(np.min(dd))

    # Max drawdown duration
    dd_dur = 0
    in_dd, dd_start = False, 0
    for i, (v, cm) in enumerate(zip(vals, cummax)):
        if v < cm and not in_dd:
            in_dd, dd_start = True, i
        elif v >= cm and in_dd:
            dd_dur = max(dd_dur, i - dd_start)
            in_dd = False
    max_dd_days = dd_dur * 7

    out = {
        "total_return_pct": round(total_ret * 100, 2),
        "annualized_return_pct": round(ann_ret * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "max_drawdown_duration_days": max_dd_days,
        "total_trades": 0,
        "win_rate_pct": 0.0,
        "avg_position_return_pct": 0.0,
        "median_position_return_pct": 0.0,
        "pct_beating_voo_pct": 0.0,
    }

    if not closed.empty and "actual_return" in closed.columns:
        rets = closed["actual_return"].dropna().values
        out["total_trades"] = len(rets)
        out["win_rate_pct"] = round(float((rets > 0).mean() * 100), 1)
        out["avg_position_return_pct"] = round(float(np.mean(rets) * 100), 2)
        out["median_position_return_pct"] = round(float(np.median(rets) * 100), 2)

        # VOO 90-day equivalent return (annualised from full-period return)
        voo_total = (bench_voo["value"].iloc[-1] / bench_voo["value"].iloc[0]) - 1
        n_w = max(len(bench_voo) - 1, 1)
        voo_90d = (1 + voo_total) ** (90 / (n_w * 7)) - 1
        out["pct_beating_voo_pct"] = round(float((rets > voo_90d).mean() * 100), 1)

    return out


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def chart_equity_curve(ph, bench_voo, bench_mixed) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ph["date"], y=ph["value"].round(2),
        name="Congressional Signal Strategy",
        line=dict(color="#2563eb", width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=bench_voo["date"], y=bench_voo["value"].round(2),
        name="VOO (100% buy & hold)",
        line=dict(color="#16a34a", width=2, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=bench_mixed["date"], y=bench_mixed["value"].round(2),
        name="60% VOO / 40% VXUS",
        line=dict(color="#9333ea", width=2, dash="dot"),
    ))

    # Shade drawdown periods
    vals = ph["value"].values
    dates = ph["date"].values
    cummax = np.maximum.accumulate(vals)
    in_dd, dd_start = False, None
    for i, (v, cm) in enumerate(zip(vals, cummax)):
        if v < cm and not in_dd:
            in_dd, dd_start = True, dates[i]
        elif v >= cm and in_dd:
            in_dd = False
            fig.add_vrect(x0=dd_start, x1=dates[i],
                          fillcolor="rgba(239,68,68,0.08)", line_width=0)

    fig.update_layout(
        title="Congressional Signal Strategy vs Benchmarks (Jan 2023 - Dec 2024)",
        xaxis_title="Date", yaxis_title="Portfolio Value ($)",
        template="plotly_white", hovermode="x unified",
        legend=dict(x=0.01, y=0.99), height=600,
    )
    return fig


def chart_score_scatter(closed: pd.DataFrame):
    if closed.empty or "actual_return" not in closed.columns:
        return go.Figure(), 0, 1, 0
    df = closed.dropna(subset=["actual_return", "score"]).copy()
    df["return_pct"] = df["actual_return"] * 100

    slope, intercept, r, p, _ = stats.linregress(df["score"], df["return_pct"])
    x_line = np.linspace(df["score"].min(), df["score"].max(), 100)

    hover_cols = [c for c in ["ticker", "legislator", "party", "filing_date"] if c in df.columns]
    fig = px.scatter(
        df, x="score", y="return_pct",
        color="score", color_continuous_scale="RdYlGn",
        hover_data=hover_cols,
        labels={"score": "Composite Score (0-100)", "return_pct": "90-Day Return (%)"},
        title=f"Score vs 90-Day Return | R²={r**2:.3f}, p={p:.3f}, slope={slope:+.2f}%/pt",
    )
    fig.add_trace(go.Scatter(
        x=x_line, y=slope * x_line + intercept,
        mode="lines", name=f"Regression (R²={r**2:.3f})",
        line=dict(color="crimson", width=2, dash="dash"),
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(template="plotly_white", height=600)
    return fig, r**2, p, slope


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    missing_tickers: list[str] = []

    print("=" * 60)
    print("Congressional Signal Investment Backtest")
    print(f"Period: {START.date()} to {END.date()}")
    print(f"Strategy: top-{MAX_POSITIONS} weekly, 90-day hold, score >= {MIN_SCORE}")
    print("=" * 60)

    # Init DB
    conn = dl.init_db()

    # Step 1: Load trades
    print("\n[1/7] Loading congressional trade data...")
    trades_df = dl.load_congressional_trades()

    # Step 2: Load committee data
    print("\n[2/7] Loading committee membership...")
    committee_map = dl.load_committee_membership()

    # Step 3: Fetch all prices upfront
    unique_tickers = list(trades_df["ticker"].dropna().unique())
    bench_tickers = ["VOO", "VXUS"]
    all_tickers = list(set(unique_tickers + bench_tickers))
    print(f"\n[3/7] Fetching price history for {len(all_tickers)} tickers...")
    price_cache = dl.batch_fetch_prices(all_tickers, PRICE_START, PRICE_END, conn)
    print(f"  Got data for {len(price_cache)}/{len(all_tickers)} tickers")

    # Step 4: Score
    print("\n[4/7] Scoring trades...")
    scored_df = run_scoring_pipeline(trades_df, conn, price_cache, committee_map, missing_tickers)

    # Step 5: Run backtest
    print("\n[5/7] Running backtest simulation...")
    ph, closed = run_backtest(scored_df, price_cache, missing_tickers)

    # Step 6: Benchmarks
    print("\n[6/7] Building benchmark curves...")
    bench_voo = build_benchmark("VOO", price_cache, alloc=1.0)
    bench_vxus = build_benchmark("VXUS", price_cache, alloc=1.0)

    # 60/40 mixed: combine weekly values
    voo_series = bench_voo.set_index("date")["value"]
    vxus_series = bench_vxus.set_index("date")["value"]
    vxus_aligned = vxus_series.reindex(voo_series.index, method="ffill")
    mixed_vals = 0.6 * voo_series + 0.4 * vxus_aligned
    bench_mixed = pd.DataFrame({"date": mixed_vals.index, "value": mixed_vals.values})

    # Step 7: Outputs
    print("\n[7/7] Generating outputs...")

    # Stats
    strategy_stats = compute_stats(ph, closed, bench_voo)
    voo_stats = compute_stats(bench_voo, pd.DataFrame(), bench_voo)
    mixed_stats = compute_stats(bench_mixed, pd.DataFrame(), bench_voo)

    print("\n=== Summary Statistics ===")
    print(f"{'Metric':<38} {'Strategy':>12} {'VOO':>12} {'60/40':>12}")
    print("-" * 76)
    for key in ["total_return_pct", "annualized_return_pct", "sharpe_ratio",
                "sortino_ratio", "max_drawdown_pct", "max_drawdown_duration_days",
                "total_trades", "win_rate_pct", "avg_position_return_pct",
                "median_position_return_pct", "pct_beating_voo_pct"]:
        sv = strategy_stats.get(key, "-")
        vv = voo_stats.get(key, "-")
        mv = mixed_stats.get(key, "-")
        print(f"  {key:<36} {str(sv):>12} {str(vv):>12} {str(mv):>12}")

    # Save stats CSV
    stats_rows = [
        {"strategy": "Congressional Signal", **strategy_stats},
        {"strategy": "VOO (buy & hold)", **voo_stats},
        {"strategy": "60% VOO / 40% VXUS", **mixed_stats},
    ]
    pd.DataFrame(stats_rows).to_csv(RESULTS_DIR / "stats.csv", index=False)

    # Trades CSV
    if not closed.empty:
        score_cols = [c for c in closed.columns if c.startswith("score")]
        export_cols = (
            ["ticker", "legislator", "party", "filing_date", "score"]
            + score_cols
            + ["entry_price", "exit_price", "actual_return", "entry_date", "exit_date"]
        )
        export_cols = [c for c in export_cols if c in closed.columns]
        out = closed[export_cols].copy()
        if "actual_return" in out.columns:
            out["return_pct"] = (out["actual_return"] * 100).round(2)
        out.to_csv(RESULTS_DIR / "trades.csv", index=False)

        if "actual_return" in closed.columns:
            by_ret = out.sort_values("return_pct", ascending=False)
            print("\n=== Top 10 Trades ===")
            display = [c for c in ["ticker", "legislator", "score", "return_pct"] if c in by_ret.columns]
            print(by_ret[display].head(10).to_string(index=False))
            print("\n=== Bottom 10 Trades ===")
            print(by_ret[display].tail(10).to_string(index=False))

    # Monthly alpha CSV
    ph_m = ph.set_index("date").resample("ME").last().reset_index()
    voo_m = bench_voo.set_index("date").resample("ME").last().reset_index()
    ph_m["strategy_ret"] = ph_m["value"].pct_change() * 100
    voo_m["voo_ret"] = voo_m["value"].pct_change() * 100
    alpha_df = ph_m[["date", "strategy_ret"]].merge(voo_m[["date", "voo_ret"]], on="date").dropna()
    alpha_df["alpha"] = alpha_df["strategy_ret"] - alpha_df["voo_ret"]
    alpha_df["cumulative_alpha"] = alpha_df["alpha"].cumsum()
    alpha_df.to_csv(RESULTS_DIR / "alpha.csv", index=False)

    # Equity curve
    eq_fig = chart_equity_curve(ph, bench_voo, bench_mixed)
    eq_fig.write_html(str(RESULTS_DIR / "equity_curve.html"))
    try:
        eq_fig.write_image(str(RESULTS_DIR / "equity_curve.png"), scale=2)
    except Exception:
        pass  # kaleido not installed

    # Score vs return scatter
    if not closed.empty and "actual_return" in closed.columns:
        sc_fig, r2, p_val, slope = chart_score_scatter(closed)
        sc_fig.write_html(str(RESULTS_DIR / "score_vs_return.html"))
        try:
            sc_fig.write_image(str(RESULTS_DIR / "score_vs_return.png"), scale=2)
        except Exception:
            pass

        print(f"\n=== Scoring System Predictive Power ===")
        print(f"  R2     = {r2:.4f}")
        print(f"  p-val  = {p_val:.4f}")
        print(f"  Slope  = {slope:+.3f}% return per score point")
        sig = p_val < 0.05
        print(f"  {'SIGNIFICANT (p < 0.05) - scoring predicts returns' if sig else 'NOT SIGNIFICANT (p >= 0.05) - insufficient evidence'}")

    # Component correlation
    score_cols = [c for c in closed.columns if c.startswith("score_c")] if not closed.empty else []
    if score_cols and "actual_return" in closed.columns:
        df_corr = closed[score_cols + ["actual_return"]].dropna()
        if not df_corr.empty:
            corr = df_corr.corr()["actual_return"].drop("actual_return").sort_values(ascending=False)
            print("\n=== Component Correlation with 90-Day Return ===")
            labels = {
                "score_c1_signal": "C1 Congressional Signal",
                "score_c2_related": "C2 Related Persons",
                "score_c3_fundamentals": "C3 Fundamentals",
                "score_c4_catalyst": "C4 Catalyst",
                "score_c5_news": "C5 News/Momentum",
            }
            for col, val in corr.items():
                print(f"  {labels.get(col, col):<30} r = {val:+.3f}")

    # Missing tickers log
    unique_missing = sorted(set(missing_tickers))
    if unique_missing:
        (RESULTS_DIR / "missing_tickers.txt").write_text("\n".join(unique_missing))
        print(f"\n  {len(unique_missing)} tickers had no price data -> results/missing_tickers.txt")

    final_val = ph["value"].iloc[-1]
    total_ret = (final_val - INITIAL_VALUE) / INITIAL_VALUE * 100
    voo_final = bench_voo["value"].iloc[-1]
    voo_ret = (voo_final - INITIAL_VALUE) / INITIAL_VALUE * 100
    print(f"\n=== Final Result ===")
    print(f"  Strategy:  ${final_val:,.0f} ({total_ret:+.1f}%)")
    print(f"  VOO:       ${voo_final:,.0f} ({voo_ret:+.1f}%)")
    print(f"  Alpha:     {total_ret - voo_ret:+.1f}pp")
    print(f"\n  Outputs saved to {RESULTS_DIR}/")
    print("    equity_curve.html, score_vs_return.html")
    print("    stats.csv, trades.csv, alpha.csv")


if __name__ == "__main__":
    main()
