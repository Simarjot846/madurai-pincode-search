madurai-pincode-search

Vector search over about 46,000 items, where every query also has filters like category, zone or price. The part I actually spent time on is the planner: a small piece of code that looks at a query's filters, guesses how many items will match, and picks a search strategy to suit.

Why this exists

Filtered vector search sounds easy. You find the nearest neighbours, then throw away the ones that don't match the filter. It works fine when the filter is loose. It falls apart when the filter is tight.

I hit this with the Madurai zone. If a filter only matches a tiny slice of the data, the nearest neighbours you pull out of the index mostly don't match, so after filtering you're left with almost nothing and recall collapses. Plain post-filtering got recall down to 0.497, 0.196 and 0.139 on the narrower filters.

So the real question is: which strategy should you use, and when?

The three strategies
exact_scan: apply the filter first, then brute-force the survivors. Slow on big result sets, perfect on small ones.
postfilter_adaptive: search the index, filter afterwards. It starts by fetching 64 candidates and keeps doubling until it has enough matches. Fast when the filter is loose.
partition_restricted: only look inside the IVF partitions that can contain matching items.

The index is IVF with 200 lists.

How the planner decides

It estimates what fraction of items the filters will match, then:

If the estimated match count is small (at most 2,500 rows, about 5.4% of the data), use exact_scan.
Otherwise, if the filter is loose enough, use postfilter_adaptive.
Otherwise, use partition_restricted.

The estimate itself comes from two places. The first is a quick independence guess (multiply the individual filter selectivities together). The second is a correction from a sample of the real data. Independence is cheap but it's wrong when filters are correlated, and in this data they are.

Results

I tested five filter-size buckets, from "matches almost everything" down to "matches a handful of items". For each one: 4 filters, 8 queries per filter, k=10. Recall is against exact results, latency is the median per query.

bucket	planner recall	planner p50	always postfilter recall	always postfilter p50
~100%	0.975	10.36 ms	0.975	10.12 ms
~10%	0.984	1.58 ms	0.962	1.35 ms
~1%	1.000	0.22 ms	0.972	0.37 ms
~0.1%	1.000	0.11 ms	0.878	0.27 ms
handful	1.000	0.04 ms	0.939	0.25 ms

What I take from this:

On loose filters, the planner and adaptive post-filtering are about the same. At ~10% post-filtering is actually a bit faster (1.35 vs 1.58 ms). The planner's edge there is a slightly better recall, nothing more.
On tight filters the planner wins on both counts. At ~0.1% and "handful", recall stays at 1.000 while always-postfilter drops to 0.878 and 0.939, and it's also 2.5x to 6x slower.
The ~100% bucket tops out at 0.975 for everyone except the exact scan. That's the ANN index's ceiling, not something the planner can fix.

One more thing: the adaptive post-filter is not the naive one. The 64-and-double trick already rescues most of the damage, which is why the gap above is much smaller than the 0.497 / 0.196 / 0.139 I saw before.

Breaking the planner on purpose

To see which parts of the design actually matter, I broke it four ways (src/break_planner.py):

indep_only: drop the sample correction, use only the independence estimate.
always_post: always use adaptive post-filtering.
no_exact: never use exact_scan, so small filters go to partition_restricted.
inverted: flip the decision, post-filter small filters and exact-scan big ones.

What happened:

inverted was the worst, as expected. 28.6 ms instead of 10.4 ms at ~100%, and recall of 0.878 at ~0.1%.
no_exact kept recall fine but was 2x to 2.7x slower on the small buckets (0.22 vs 0.11 ms at ~0.1%).
always_post is covered in the table above.
indep_only looked identical to the full planner.

That last one deserves a straight answer, because I expected it to matter.

The independence guess really is off. The worst pairs I found are underestimated by about 14x (true 1.05%, estimated 0.069%) and overestimated by about 35x (true 0.006%, estimated 0.225%). Zone 22, which is Madurai, shows up in all five of the worst overestimates. Clothing in Madurai is 2.2% of the data but the independence guess says 0.28%.

But the decision never changed. I checked all 630 category x zone pairs (tests/check_flips.py) and counted how often each estimate sent a query down the wrong path. Both the independence estimate and the planner's estimate got it wrong 0 times. In every pair, the true count and the guessed count land on the same side of the 2,500-row line, so the planner picks the same strategy either way.

So the honest summary is: the sample correction makes the estimate more accurate, but at 46k items I could not find a single case where that changed what the planner did. I'd expect it to matter on a bigger dataset or with more filters combined, but I haven't tested that, so I'm not claiming it.

Limits of these numbers
Only 32 or so queries per bucket, and each latency is a single run. Differences like 0.984 vs 0.962 recall, or 1.35 vs 1.58 ms, could easily be noise. Treat those as "about the same", not as wins.
One dataset, one k (10), one machine.
The flip check only covers category x zone pairs. I didn't test price ranges or three-filter combinations.
The IVF index loses a little recall on its own, so nothing here reaches 1.000 on the big buckets unless it uses the exact scan.
Running it

Clone the repo, make a virtual environment, install the packages the code imports (numpy at the least), and run the scripts from the project root.

python tests/run_break.py     # finder for the worst estimates + the comparison table
python tests/check_flips.py   # counts wrong decisions over all category x zone pairs

The first one builds the index, so it takes a bit. The second doesn't build anything and finishes in seconds.

Layout
data/                # the dataset and the processed vectors the tests load
src/
  dataset.py         # loads vectors, metadata, categories, titles
  ivf.py             # the IVF index
  filters.py         # predicates and exact masks
  strategies.py      # exact_scan and the other strategies
  planner.py         # the real planner
  break_planner.py   # the four broken planners and the comparison code
  workload.py        # builds the filter buckets
tests/               # run_break.py, check_flips.py and the earlier step tests
