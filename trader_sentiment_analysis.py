import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

pd.set_option('display.width', 140)
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

HIST_PATH = r'E:\COLLEGE\historical_data.csv'
FG_PATH = r'E:\COLLEGE\fear_greed_index.csv'

# ---------------------------------------------------------------------------
# 1. LOAD & MERGE
# ---------------------------------------------------------------------------
tr = pd.read_csv(HIST_PATH)
fg = pd.read_csv(FG_PATH)

tr['date'] = pd.to_datetime(tr['Timestamp IST'], format='%d-%m-%Y %H:%M').dt.date
fg['date'] = pd.to_datetime(fg['date']).dt.date

df = tr.merge(fg[['date', 'classification', 'value']], on='date', how='left')
print("Unmatched rows after merge:", df['classification'].isna().sum(), "/", len(df))

def simplify(c):
    if c in ['Fear', 'Extreme Fear']:
        return 'Fear'
    if c in ['Greed', 'Extreme Greed']:
        return 'Greed'
    return 'Neutral'

df['sentiment'] = df['classification'].apply(simplify)

def side_bucket(d):
    if 'Long' in d or d == 'Buy':
        return 'Long'
    if 'Short' in d or d == 'Sell':
        return 'Short'
    return 'Other'

df['pos_side'] = df['Direction'].apply(side_bucket)
df.to_pickle('merged.pkl')

# Only rows with a nonzero Closed PnL represent an actual realized close.
closes = df[df['Closed PnL'] != 0].copy()

# ---------------------------------------------------------------------------
# 2. PROFITABILITY BY SENTIMENT
# ---------------------------------------------------------------------------
order = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']

print("\n=== Profitability by sentiment classification (closed trades only) ===")
g = closes.groupby('classification').agg(
    trades=('Closed PnL', 'count'),
    avg_pnl=('Closed PnL', 'mean'),
    median_pnl=('Closed PnL', 'median'),
    win_rate=('Closed PnL', lambda x: (x > 0).mean()),
    total_pnl=('Closed PnL', 'sum'),
).reindex(order).round(2)
print(g)

print("\n=== Profitability by simplified sentiment (Fear / Neutral / Greed) ===")
g2 = closes.groupby('sentiment').agg(
    trades=('Closed PnL', 'count'),
    avg_pnl=('Closed PnL', 'mean'),
    win_rate=('Closed PnL', lambda x: (x > 0).mean()),
    total_pnl=('Closed PnL', 'sum'),
).round(2)
print(g2)

# ---------------------------------------------------------------------------
# 3. TRADING VOLUME & ACTIVITY BY SENTIMENT
# ---------------------------------------------------------------------------
print("\n=== Volume & activity by sentiment ===")
g3 = df.groupby('classification').agg(
    volume_usd=('Size USD', 'sum'),
    avg_trade_size_usd=('Size USD', 'mean'),
    trades=('Closed PnL', 'count'),
).reindex(order).round(1)
print(g3)

# ---------------------------------------------------------------------------
# 4. LONG VS SHORT PERFORMANCE BY SENTIMENT
# ---------------------------------------------------------------------------
print("\n=== Long vs. Short performance by sentiment ===")
g4 = closes[closes['pos_side'].isin(['Long', 'Short'])].groupby(['sentiment', 'pos_side']).agg(
    trades=('Closed PnL', 'count'),
    avg_pnl=('Closed PnL', 'mean'),
    win_rate=('Closed PnL', lambda x: (x > 0).mean()),
    total_pnl=('Closed PnL', 'sum'),
).round(2)
print(g4)

# ---------------------------------------------------------------------------
# 5. CORRELATION: NUMERIC SENTIMENT SCORE VS DAILY PnL / ACTIVITY
# ---------------------------------------------------------------------------
daily = closes.groupby('date').agg(daily_pnl=('Closed PnL', 'sum'), n_trades=('Closed PnL', 'count'))
daily = daily.merge(fg[['date', 'value']], on='date', how='left')
print("\nCorr(F&G value, daily total PnL):", round(daily['value'].corr(daily['daily_pnl']), 3))
print("Corr(F&G value, daily trade count):", round(daily['value'].corr(daily['n_trades']), 3))

# ---------------------------------------------------------------------------
# 6. ACCOUNT & ASSET CONCENTRATION
# ---------------------------------------------------------------------------
print("\n=== Top 10 accounts by total realized PnL ===")
acct = closes.groupby('Account').agg(
    total_pnl=('Closed PnL', 'sum'),
    trades=('Closed PnL', 'count'),
    win_rate=('Closed PnL', lambda x: (x > 0).mean()),
).sort_values('total_pnl', ascending=False)
print(acct.head(10).round(2))

print("\n=== Top-10 accounts' PnL split by sentiment ===")
top_accounts = acct.head(10).index
piv = closes[closes['Account'].isin(top_accounts)].pivot_table(
    index='Account', columns='sentiment', values='Closed PnL', aggfunc='sum'
).round(1)
print(piv)

print("\n=== Top-10 traded coins' PnL by sentiment ===")
top_coins = closes.groupby('Coin')['Size USD'].sum().sort_values(ascending=False).head(10).index
coin_g = closes.groupby(['Coin', 'sentiment'])['Closed PnL'].sum().unstack().fillna(0)
print(coin_g.loc[coin_g.index.intersection(top_coins)].round(1))

# ---------------------------------------------------------------------------
# 7. CHARTS
# ---------------------------------------------------------------------------
colors = ['#8B0000', '#E06666', '#999999', '#93C47D', '#274E13']

fig, ax1 = plt.subplots(figsize=(8, 5))
plot_g = g.reindex(order)
ax1.bar(plot_g.index, plot_g['avg_pnl'], color=colors)
ax1.set_ylabel('Average Closed PnL (USD)')
ax1.set_title('Average Trader PnL by Market Sentiment')
ax2 = ax1.twinx()
ax2.plot(plot_g.index, plot_g['win_rate'] * 100, color='black', marker='o', linewidth=2)
ax2.set_ylabel('Win Rate (%)')
ax2.set_ylim(0, 100)
plt.xticks(rotation=20)
fig.tight_layout()
fig.savefig('chart1_avgpnl_winrate.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(g3.index, g3['volume_usd'] / 1e6, color=colors)
ax.set_ylabel('Total Traded Volume (USD, Millions)')
ax.set_title('Total Trading Volume by Market Sentiment')
plt.xticks(rotation=20)
fig.tight_layout()
fig.savefig('chart2_volume.png')
plt.close(fig)

piv2 = closes[closes['pos_side'].isin(['Long', 'Short'])].pivot_table(
    index='sentiment', columns='pos_side', values='Closed PnL', aggfunc='mean'
).reindex(['Fear', 'Neutral', 'Greed'])
fig, ax = plt.subplots(figsize=(7, 5))
piv2.plot(kind='bar', ax=ax, color=['#3D85C6', '#CC4125'])
ax.set_ylabel('Average Closed PnL (USD)')
ax.set_title('Average PnL: Long vs Short by Sentiment')
plt.xticks(rotation=0)
fig.tight_layout()
fig.savefig('chart3_long_short.png')
plt.close(fig)

fg_ts = fg.copy()
fg_ts['date'] = pd.to_datetime(fg_ts['date'])
daily_ts = closes.copy()
daily_ts['date'] = pd.to_datetime(daily_ts['date'])
daily_ts = daily_ts.groupby('date')['Closed PnL'].sum().reset_index()
merged_ts = fg_ts.merge(daily_ts, on='date', how='inner').sort_values('date')

fig, ax1 = plt.subplots(figsize=(11, 5))
ax1.plot(merged_ts['date'], merged_ts['value'], color='#666666')
ax1.set_ylabel('Fear & Greed Index Value')
ax1.set_ylim(0, 100)
ax2 = ax1.twinx()
ax2.bar(merged_ts['date'], merged_ts['Closed PnL'], color='#3D85C6', alpha=0.4, width=1)
ax2.set_ylabel('Daily Total Closed PnL (USD)')
ax1.set_title('Market Sentiment vs Daily Trader PnL Over Time')
fig.tight_layout()
fig.savefig('chart4_timeline.png')
plt.close(fig)

print("\nAll charts saved. Analysis complete.")
