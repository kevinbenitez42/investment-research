# Analytics

This package contains the core finance and time-series analysis logic used by notebooks and workflows.

Current module themes include:

- metric definitions and compute helpers
- momentum analytics
- relative-risk analytics
- distribution and drawdown analytics
- OHLC and close-based analysis helpers
- reusable pandas series utilities

This package should stay focused on computation and derived metrics, not on figure rendering or notebook display behavior.

## Data Shapes

Analytics functions should use predictable pandas shapes. Keep data retrieval in `Quantapp.data`, then pass normalized pandas objects into analytics.

### Single Series

Use a `Series` for one time series, such as one asset's close prices, returns, spread, or ratio.

```text
index = datetime
values = one metric through time
```

This shape is accepted by:

```python
compute.latest(close_series, metric=Metric().percent_return)
compute.rolling(close_series, metric=Metric().z_score, window=21)
compute.rolling_windows(close_series, metric=pd.Series.std, windows=[21, 50, 200])
```

### Multi-Asset Single Field

Use a wide `DataFrame` when each column is the same field for a different asset.

```text
index = datetime
columns = symbols
values = same field for every symbol, such as Close or daily return
```

Example:

```text
            AAPL    MSFT    SPY
date
2024-01-01  100.0   200.0   400.0
2024-01-02  102.0   198.0   405.0
```

This shape is accepted by:

```python
compute.latest(close_df, metric=Metric().percent_return)
compute.rolling(close_df, metric=Metric().z_score, window=21)
```

`compute.latest` and `compute.rolling` apply the metric column by column.
Use `compute.rolling_windows` when the same metric needs many windows at once.
Series input returns one column per window; DataFrame input returns MultiIndex columns
shaped as `(window, original_column)`.

### Single-Asset OHLCV

Use a flat `DataFrame` for one asset with multiple fields.

```text
index = datetime
columns = Open, High, Low, Close, Volume
```

Example:

```text
            Open    High    Low     Close   Volume
date
2024-01-01  100.0   103.0   99.0    102.0   1000000
2024-01-02  102.0   105.0   101.0   104.0   1200000
```

This shape is accepted by frame metrics:

```python
compute.latest_frame(ohlcv_df, metric=ohlc_metric.yang_zhang)
compute.rolling_frame(ohlcv_df, metric=ohlc_metric.yang_zhang, window=21)
```

`compute.latest_frame` and `compute.rolling_frame` pass the whole frame, or whole rolling frame window, into the metric.

### Multi-Asset OHLCV

Use a wide `DataFrame` with MultiIndex columns.

```text
index = datetime
columns level 0 = field   # Open, High, Low, Close, Volume
columns level 1 = symbol  # AAPL, MSFT, SPY
```

Set column level names:

```python
panel.columns.names = ["field", "symbol"]
```

Extract close prices across all assets:

```python
close_df = panel.xs("Close", axis=1, level="field")
```

Extract one asset's OHLCV frame:

```python
aapl_ohlcv = panel.xs("AAPL", axis=1, level="symbol")
```

This shape is accepted by by-asset frame metrics:

```python
compute.latest_by_asset(panel, metric=ohlc_metric.yang_zhang)
compute.rolling_by_asset(panel, metric=ohlc_metric.yang_zhang, window=21)
```

`compute.latest_by_asset` and `compute.rolling_by_asset` split the panel by symbol, then pass each asset's OHLCV frame into the frame metric.

## Compute Contracts

Use the compute function that matches the metric input shape:

```text
latest / rolling
    metric receives one Series

rolling_windows
    metric receives one Series per window when no vectorized path exists
    common reducers and selected metrics such as Metric.sharpe use optimized multi-window paths

latest_frame / rolling_frame
    metric receives one DataFrame

latest_by_asset / rolling_by_asset
    metric receives one asset DataFrame at a time from a MultiIndex panel
```
