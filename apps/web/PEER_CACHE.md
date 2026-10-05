# Peer benchmark refresh

The momentum dashboard refreshes Sector, Industry Group, Industry, and
Sub-Industry baskets when its notebook launch cell runs. Each basket excludes
the analyzed company and uses the current Wikipedia S&P 500/400/600 universe.
These are equal-weight research baskets, not official GICS index returns.
Historical membership is not reconstructed; current constituents introduce
survivorship bias into historical comparisons.

- `company_data/gics_companies.csv` is checked against Wikipedia every seven
  days. Its `.meta.json` records source URLs, last attempt, last successful
  check, row count, and content fingerprint. An untracked or modified CSV is
  checked again; its filesystem date is never treated as source verification.
- Company prices are shared across tickers and classification levels in
  `company_data/peer_prices`. Each symbol/period/interval is fetched at most once
  per UTC day, normally in one batch. The existing price API does not expose
  incremental ranges, so due symbols receive full adjusted history. This also
  incorporates historical corporate-action adjustments.
- Benchmark CSVs and `.meta.json` files live in `company_data/factor_peer_indexes`.
  They record selected and usable constituents, excluded symbols, classification,
  membership fingerprint, calculation settings, and price coverage. Membership
  changes and new price checks trigger rebuilding. At least two valid peer
  returns are required per historical observation; older observations may use
  fewer constituents than the present-day basket.
- Refresh failures preserve usable cached data and the original successful
  source-check timestamps. Automatic retries back off for one hour. Unknown
  classifications and conflicting duplicate symbols reject a company-list
  refresh rather than silently inventing classifications.
- The legacy Wikipedia label `Specialty Stores` maps to `Other Specialty Retail`
  (25504040). Identical duplicate classifications use large-, then mid-, then
  small-cap precedence. Distinct share-class symbols remain distinct constituents.
- OS file locks serialize concurrent notebook refreshes. Atomic replacement
  protects each saved CSV/metadata file from interrupted writes.

The dashboard shows each basket's peer count, Wikipedia last-check date, price
coverage, and any refresh problems. **Refresh peers now** bypasses the Wikipedia
seven-day limit; daily company prices are reused. The button updates saved data.
Rerun the notebook launch cell to apply it to all precomputed charts. The UI
explicitly states this instead of mixing refreshed data with old calculations.

The metadata describes when Wikipedia was checked, not a guarantee that its
classifications match a licensed, real-time GICS source.
