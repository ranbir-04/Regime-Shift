Regime-Shift

An HMM-based regime detector (Bull / Bear / Crisis) that drives a regime-conditional portfolio optimizer, backtested walk-forward against a static 60/40 and an equal-weight portfolio. Built to answer one question: does switching allocation based on detected market regime actually beat just holding a static portfolio, once you account for transaction costs and refuse to let the model see future data?

Everything lives in one notebook, run top to bottom: data pull, regime detection, optimization, backtest, results.

What it does:

Detects hidden market regimes with a 3-state Gaussian HMM on SPY/TLT/GLD returns + VIX, no manual labeling
Auto-labels the states by character (highest VIX = Crisis, highest return = Bull, rest = Bear/Neutral)
Picks weights based on regime — max-Sharpe in Bull, min-vol in Crisis, equal-weight otherwise
Refits the HMM from scratch every rebalance on a trailing 252-day window, so it never sees future data
Charges itself 8bps of transaction cost per unit of turnover on every rebalance
Reports Sharpe, Sortino, max drawdown, and Calmar against both baselines

Stack: Python, pandas, NumPy, hmmlearn, scipy, yfinance, matplotlib.

Running it

Open Regime_Shift_2.ipynb in Jupyter or Colab and run all cells. The first cell installs everything. No API keys or env vars needed, data pulls live from Yahoo Finance. Results will drift slightly run to run as new data comes in. If a cell errors on the yfinance call it's usually just a rate limit, re-run it.

Results

Walk-forward, 2008–present:

Metric	Regime-Shift	60/40	Equal Weight
Annual Return	0.083	0.101	0.091
Annual Vol	0.106	0.106	0.095
Sharpe	0.778	0.951	0.954
Sortino	1.001	1.246	1.311
Max Drawdown	-0.257	-0.272	-0.227
Calmar	0.321	0.371	0.400

Doesn't beat either baseline. Both static portfolios win on Sharpe, Sortino, and Calmar. The model's drawdown chart shows a rough 2022 stretch where it kept switching between the Bear/Neutral and Crisis labels instead of settling on one, and paid turnover cost each time it flipped.

Why it underperformed

With only 3 assets and VIX as inputs, Bear/Neutral and Crisis aren't very statistically distinct, so the HMM whipsaws between them — visible in the regime timeline plot, especially 2022–2024. That whipsaw is the main drag, more than the regime-switching logic itself.

Refitting the HMM on every 21-day rebalance (instead of fitting once and reusing it) is the only honest way to backtest this without look-ahead bias, but it means the state labels aren't fully stable rebalance to rebalance, which adds turnover on top of what the regime signal alone would need.

What I'd change next
Pull a real macro panel from FRED (CPI, yields, spreads) instead of using VIX as the only regime signal
Go past 3 asset classes
Grid-search n_states, window, and rebalance_freq instead of fixed defaults — skipped this to avoid overfitting the backtest to itself
Swap scipy for CVXPY if the optimization ever needs more than a box + sum-to-1 constraint
