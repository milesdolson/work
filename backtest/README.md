# Congressional Signal Investment Backtester

Evaluates a government-signal investing strategy against VOO and VXUS benchmarks
using free public congressional trade disclosure data. No API keys required.

## Setup

```bash
cd backtest/
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python backtest.py
```

The first run downloads and caches all data. Subsequent reruns use the local
SQLite cache (`price_cache.db`) and skip yfinance requests.

**Runtime:** 20-60 minutes depending on network speed (300+ tickers to fetch).

## Data Sources

| Source | URL | Notes |
|---|---|---|
| House trades | wallstreetlocal GitHub CSV (primary) | Falls back to house-stock-watcher S3 JSON |
| Senate trades | wallstreetlocal GitHub CSV (primary) | Falls back to senate-stock-watcher S3 JSON |
| Price data | Yahoo Finance via yfinance | Cached in SQLite after first fetch |
| Committee membership | unitedstates/congress-legislators YAML | Fetched on first run |
| Fundamentals | Yahoo Finance via yfinance | Cached in SQLite |

## Strategy Logic

- **Universe:** All congressional BUY disclosures in the dataset, Jan 2023 - Dec 2024
- **Scoring:** Each disclosure scored 0-100 across 5 components (see below)
- **Minimum threshold:** 50/100 to enter the portfolio
- **Entry:** Market open the day AFTER the filing date (not transaction date)
- **Exit:** Market open 90 calendar days after entry
- **Portfolio:** Top 5 scoring disclosures per week, equal weight (20% each)
- **Rebalance:** Weekly - new signals fill vacated slots
- **Cash:** Undeployed allocation earns 0%

## Scoring Components (0-100 total)

| # | Component | Max | Description |
|---|---|---|---|
| 1 | Congressional Signal Quality | 25 | Trade size tier + committee relevance + cluster bonus |
| 2 | Related Persons Activity | 15 | Spouse/dependent trades in same ticker |
| 3 | Fundamentals | 25 | Revenue growth, P/E ratio, debt/equity |
| 4 | Upcoming Catalysts | 20 | Earnings proximity (proxy for FDA/contract catalysts) |
| 5 | News Alignment | 15 | 30-day price momentum (proxy for sentiment) |

## Outputs

All saved to `results/`:

| File | Description |
|---|---|
| `equity_curve.html` | Interactive equity curve: strategy vs VOO vs 60/40 |
| `score_vs_return.html` | Score vs 90-day return scatter with regression line |
| `stats.csv` | Summary statistics for strategy and both benchmarks |
| `trades.csv` | All closed positions with scores and returns |
| `alpha.csv` | Monthly alpha vs VOO |
| `missing_tickers.txt` | Tickers with no price data (often OTC/foreign) |

## Benchmarks

- **Benchmark A:** $100,000 into VOO on Jan 1 2023, buy & hold
- **Benchmark B:** $60,000 VOO + $40,000 VXUS on Jan 1 2023, buy & hold

## Known Limitations

**Committee data is current, not historical.** The `committee-membership-current.yaml`
reflects current assignments. Committee memberships change across Congress sessions,
so some relevance scores during 2023-2024 may be slightly inaccurate.

**Catalyst scoring uses earnings proximity as proxy.** FDA drug approvals, government
contract awards, and regulatory votes are the real high-value catalysts for
congressional signal trades - especially in healthcare and defense. These are
difficult to backtest cleanly with free data. Earnings proximity is the best
available free proxy but undercounts the biotech and defense sector signals.

**News alignment uses price momentum, not NLP.** Real news sentiment scoring
(e.g., FinBERT) would be more accurate than 30-day price momentum as a proxy
for news alignment. Momentum and actual sentiment correlate but diverge
around surprise events.

**Congressional disclosure lag.** Even using filing date (not transaction date)
as entry signal, the actual transaction may have occurred 30-45 days earlier.
This is an inherent limitation of congressional disclosure requirements, not
a backtesting error.

**Survivorship bias on delisted tickers.** yfinance will fail on tickers that
were delisted between 2023-2024 (bankruptcies, acquisitions, delistings).
These are logged to `missing_tickers.txt` and counted as positions that never
opened (conservative treatment). A more accurate treatment would use the last
available price as the exit price.

**Free data may have cleaning artifacts.** The house/senate-stock-watcher datasets
are community-maintained and may have occasional missing disclosures, typos in
tickers, or date format inconsistencies. A paid source like Quiver Quant or
Capitol Trades API would provide cleaner data.

## Rerunning

The SQLite cache (`price_cache.db`) persists across runs. To force a full
re-download of all data, delete `price_cache.db` and the `data/` directory.

To change the backtest parameters, edit the constants at the top of `backtest.py`:
- `START`, `END` - backtest window
- `INITIAL_VALUE` - starting portfolio size
- `HOLD_DAYS` - position hold period
- `MIN_SCORE` - minimum score threshold
- `MAX_POSITIONS` - portfolio slots

## score.py as an importable module

`score.py` is designed to be imported directly by the live daily scanner:

```python
from backtest.score import score_trade

result = score_trade(
    trade={"ticker": "AAPL", "amount": "$50,001 - $100,000", ...},
    all_trades=trades_df,
    legislator_committees=["Senate Commerce, Science, and Transportation"],
    sector="Information Technology",
    cluster_count=0,
    fundamentals={"revenue_growth": 0.08, "pe_ratio": 28, "debt_equity": 0.3},
    earnings_dates=[pd.Timestamp("2026-07-01")],
    close_series=price_series,
)
print(result["score"])  # 0-100
```
