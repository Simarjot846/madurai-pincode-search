# Madurai Pincode Search: Filtered Vector Search Planner

Vector search with filters (zone, category, price) breaks when the filter is very selective.
A search for "saree" inside Madurai only sees a tiny part of the catalogue, so the usual
"search first, filter after" approach returns almost nothing.

This project builds a **query planner**. It estimates how many items a filter keeps, then picks
one of three search strategies. It is compared with the two simple approaches the search team
already tried.

The catalogue has 46,228 items. All results are from one Windows laptop. They do not show
production performance.

## Setup

Windows PowerShell. Run these from the project folder.

```
python -m venv .venv                 # make a private Python environment
.venv\Scripts\activate              # turn it on
pip install -r requirements.txt     # install the packages this project needs
```

## Run

```
python tests/run_benchmark.py   # MAIN RESULT: planner vs two baselines, 5 selectivity buckets
python tests/check_planner.py   # quick check on 7 hand-picked filters
python tests/check_flips.py     # does a wrong estimate ever change the decision?
python tests/run_calibrate.py   # tries every strategy on every bucket, used to set thresholds
python tests/run_mutate.py      # deletes 10% of items, inserts 5% shifted items, re-runs
python tests/run_break.py       # breaks the planner on purpose, 4 ways
python tests/check_workload.py  # shows how many filters landed in each bucket
```

Close other apps and screen recorders before timing runs. Latency numbers can move by 30% or
more with background load. Recall numbers stay the same from run to run.

## How it works

See `architecture.mmd` for the diagram.

1. **Estimate** the share of items the filter keeps. A random sample is used first. If the
   sample has fewer than 5 hits, a cheaper counter-based estimate is used instead.
2. **Choose** a strategy:
   - small result (up to 2,500 rows): `exact_scan`. Filter first, then check every survivor.
   - big result (5% or more): `postfilter_adaptive`. Search at least 64 partitions, then keep
     doubling until 10 matches are found.
   - in between: `partition_restricted`. Search only the partitions that hold a match.
3. **Search** with the chosen strategy.

The thresholds (`t_exact_rows=2500`, `t_post_sel=0.05`) were set from the measurements in
`run_calibrate.py`.

## Code layout

- `src/ivf.py`: the index (200 partitions)
- `src/filters.py`: filter checks and the two selectivity estimators
- `src/strategies.py`: the three search strategies
- `src/planner.py`: the planner that chooses between them
- `src/baselines.py`: the two simple approaches (pure post-filter, pure pre-filter)
- `src/workload.py`: builds the test filters, grouped into 5 selectivity buckets
- `src/benchmark.py`, `src/calibrate.py`, `src/mutate.py`, `src/break_planner.py`: the test code

## Results (recall@10, median latency per query)

From `python tests/run_benchmark.py`:

| Bucket | Planner | Pure post-filter | Pure pre-filter |
|---|---|---|---|
| ~100% | 0.983 / 13.7 ms | 0.930 / 3.9 ms | 1.000 / 36.2 ms |
| ~10% | 0.989 / 2.1 ms | 0.833 / 3.5 ms | 1.000 / 3.0 ms |
| ~1% | 1.000 / 0.23 ms | 0.497 / 3.5 ms | 1.000 / 0.25 ms |
| ~0.1% | 1.000 / 0.11 ms | 0.196 / 3.7 ms | 1.000 / 0.06 ms |
| handful (5-40 items) | 1.000 / 0.06 ms | 0.139 / 4.5 ms | 1.000 / 0.06 ms |

What the table shows:

- Pure post-filter loses most results as the filter gets narrower. This is the Madurai problem.
- The planner keeps recall at 0.98 or higher in every bucket.
- At ~100%, the planner is about 2.6x faster than pre-filter, but about 3.5x slower than pure
  post-filter. It pays that cost to get higher recall (0.983 vs 0.930).
- At ~10%, the planner is about 29% faster than pre-filter, with much better recall than
  post-filter (0.989 vs 0.833).
- At ~1% and below, the planner picks `exact_scan`, so it matches pre-filter. At ~0.1% it is
  slightly slower (0.11 vs 0.06 ms), probably because of the cost of the estimate step. This
  was not measured separately.
- The ~100% bucket has only 4 filters, so its latency spread is loose.

## Estimate check

The independence-based estimate is often wrong. For some category + zone pairs it is off by
up to about 35x.

I checked all 630 category x zone pairs. With the independence estimate, 0 decisions were
wrong. With the planner's estimate, 0 were wrong. So the estimate error was real, but it never
changed which strategy was picked at this scale. This covers category x zone pairs only. Price
ranges and 3-filter combinations were not checked.

## Mutation test

Deleted 10% of items (46,228 down to 43,917 after inserts) and inserted 5% new items. The new
items are all Pet Supplies in Madurai, at 1.5x price. The index was rebuilt. Only the
estimators were left stale in the "stale" planner.

- True share of the new hot filter: 5.278%. Stale estimate: 0.250%.
- Both the fresh and stale planners still chose `exact_scan`, so recall stayed 1.000.
- On the mutated data, planner recall was 0.990 at ~100%, 0.992 at ~10%, and 1.000 below that.

## Break test

| Broken planner | What happened |
|---|---|
| `indep_only` (no sample correction) | same as the good planner |
| `always_post` | recall 0.878 at ~0.1%, 0.939 at handful |
| `no_exact` | recall stayed fine, but 2-2.7x slower on small filters |
| `inverted` | worst: 28.6 ms at ~100% (2.8x slower), recall 0.878 at ~0.1% |

## Limitations

- At 46k items the planner is effectively two-tier (exact vs adaptive post-filter).
  `partition_restricted` was never the best choice here. It may help on larger catalogues,
  but this was not tested.
- The thresholds were tuned on the same data and workload that the results are reported on.
- The sample estimator made estimates more accurate, but no decision changed at this scale.
- The index was rebuilt after mutation. Incremental updates were not tested.
- Only category x zone pairs were checked for estimate errors.
- Latency is from one machine and moves with background load.