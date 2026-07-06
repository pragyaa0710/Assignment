# Trader Performance vs. Bitcoin Market Sentiment

This was my submission for the Primetrade.ai data science hiring assignment. The task: figure out if there's any real relationship between how traders on Hyperliquid perform and what the broader market "feels" (Fear vs. Greed), using historical trade data and the Bitcoin Fear & Greed Index.

## What's in here

- **`trader_sentiment_analysis.py`** — the full analysis script. Loads both datasets, merges them by date, and works through profitability, position sizing, long/short behavior, account-level patterns, and generates all the charts.
- **`Trader_Sentiment_Analysis_Report.docx`** — the write-up with the actual findings, charts, and takeaways in a readable format.

## How to run it

You'll need `historical_data.csv` (Hyperliquid trade data) and `fear_greed_index.csv` (the sentiment index) in the same folder as the script, then:

```bash
pip install pandas matplotlib numpy
python trader_sentiment_analysis.py
```

It'll print out summary tables to the console and save four PNG charts alongside the script.

## What I found

- Traders actually did *better* at the sentiment extremes — Extreme Greed and Fear both had higher win rates than calm, Neutral markets. Not what I expected going in.
- Short positions were more profitable than longs across the board, especially during Greed — basically, fading the crowd's optimism paid off more than riding it.
- People sized up their positions during Fear rather than pulling back, which reads more like conviction/contrarian behavior than panic.

Full breakdown, numbers, and charts are in the report — this repo is just the code + document behind it.

## Notes

- The trade data didn't have an explicit leverage column, so position size (USD) was used as a stand-in for conviction/risk.
- This is a historical, exploratory analysis — not a backtested strategy or trading advice.
