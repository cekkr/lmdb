# Benchmarks

> The smoke harness now appends queue-drain snapshots automatically whenever the quality queue
> crosses its threshold. Look for sections titled “Queue Drain (auto smoke harness)” for those runs.

## 2026-07-27 - Graph context memory integration validation

- **Scope:** functional validation that the new Cheetah graph context memory works end to end from
  DB-SLM training and inference. This is an **integration** record, not a quality or throughput
  benchmark: no A/B comparison of generation quality was run, and the corpus is far too small to
  support one.
- **Environment:** macOS (Darwin 25.5.0), Python 3.14 in `.venv`, Cheetah submodule at `8ecdf35`
  ("Add complete sentence references to graph recall"), server launched headless in a bounded
  `screen` session on `127.0.0.1:4455`. Isolated Cheetah database `graph_e2e`; scratch SQLite and
  a 40-record NDJSON subset of `datasets/emotion_data.json` under a session scratch directory. The
  server binary predates the run; no Go code was changed.
- **Command (training):**

  ```bash
  DBSLM_CHEETAH_DATABASE=graph_e2e DBSLM_GRAPH_MEMORY=1 DBSLM_SQLITE_PATH=<scratch>/e2e.sqlite3 \
  PYTHONPATH=src .venv/bin/python src/train.py <scratch>/emotion_data.json \
    --db <scratch>/e2e.sqlite3 --reset --ngram-order 4 \
    --json-chunk-size 20 --max-json-lines 40 --chunk-eval-percent 0.1 --eval-samples 2 \
    --graph-memory --graph-term-index-rebuild --max-runtime-seconds 600
  ```

- **Training result:** completed ingest of 15,577 tokens / 15,571 n-grams in ~392 s with a Cheetah
  Top-K hit ratio of ~54.53%. Graph observation reported, per chunk:

  | Chunk | Records | Nodes | Edges applied/requested | Created | Updated | Failed | References | Graph time |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | chunk1 | 20 | 165 | 1067/1067 | 991 | 76 | 0 | 20 | 11.18 s |
  | chunk2 | 20 | 94 | 1059/1059 | 846 | 213 | 0 | 20 | 12.96 s |

  Zero failed edges across 2,126 upserts. The rising `updated` share between chunks is the expected
  signal that the second chunk reused ids minted by the first. `GRAPH_TERM_INDEX action=rebuild`
  then indexed 542 nodes / 2,264 terms. Graph ingest accounted for roughly 6% of wall-clock time at
  this size; that ratio is untested at scale.
- **Command (inference):**

  ```bash
  DBSLM_CHEETAH_DATABASE=graph_e2e DBSLM_GRAPH_MEMORY=1 DBSLM_SQLITE_PATH=<scratch>/e2e.sqlite3 \
  PYTHONPATH=src .venv/bin/python src/run.py --db <scratch>/e2e.sqlite3 \
    --graph-memory --graph-recall-log \
    --prompt "How does envy shape the way a team works together?" --max-response-words 60
  ```

- **Inference result:** the turn recalled `seeds=6, associations=8, terms=4, truncated=0` and
  produced a correctly framed response. No recalled reference sentence appeared in the visible
  output, confirming the internal-context boundary. The same prompt with `--no-graph-memory` ran
  without a recall and produced a comparable (equally low-quality) response; at 40 records neither
  output carries information about generation quality.
- **Protocol round-trip:** a separate scripted check against the live server exercised
  `GRAPH_NODE_SET` with reference sentences, `GRAPH_NODE_GET` reference merging across runs,
  `GRAPH_EDGE_SET_BATCH`, `GRAPH_RECALL` with exact-id / free-text / spaced (base64) seeds and
  `min_sources=2`, `GRAPH_SIMILAR`, and `GRAPH_TERM_INDEX action=stats`. Reference hydration
  required `include_seeds=1`, since a seed node is otherwise excluded from its own answer.
- **Not measured:** quality deltas with graph bias on/off, decoder latency impact, behaviour at
  250/1,000 records, and any tuning of `DBSLM_GRAPH_BIAS_WEIGHT` or
  `DBSLM_GRAPH_RECALL_PRECISION`. These are tracked in [`NEXT_STEPS.md`](../NEXT_STEPS.md).

## 2026-07-24 - Eight-hour emotion-data repair validation

- Scope and window: repaired empty/scaffolded evaluation responses while continuously supervising
  emotion-data training from 2026-07-23 23:16:12 CEST through the hard deadline at
  2026-07-24 07:16:12 CEST. The initial reproduction used the requested order-5 recursive/reset
  command. After the initial reset, every repair restart reused
  `var/train_resume.json` without `--reset`, retained the same metrics/log artifacts, and reduced
  `--json-chunk-size` from 500 to 2 so each recoverable chunk and trainer segment could remain
  bounded. Cheetah database `default` on port 4455 was never reset again.
- Runtime ownership: the exact `screen` session was `dbslm_emotion_8h`; supervisor segments were
  capped at one hour and the final segment was capped to the remaining 1,685 seconds. The
  supervisor stopped at deadline epoch `1784870172`; afterward no trainer or screen session
  remained, while the launch-owned Cheetah server (PID 43957) remained healthy and listening.
- Repaired faults found by live probes: marker-only decodes no longer become empty user responses;
  structural tag IDs are excluded from session-cache and prediction-table mixtures; training and
  evaluation now share dataset-configured prompt framing; dependency annotations remain a
  side-channel rather than serialized training text; malformed dependency artifacts are rejected;
  MKN rebuilds prefer current SQLite ingest rows and explicitly flush changed Cheetah mirrors;
  metrics append atomically across resumes; and spaced/case-variant end markers such as
  `| end |` are stripped before exposure. The command also gained recoverable runtime-boundary
  support used by the supervisor.
- Restart disclosure: repairs caused at-least-once replay of chunks 5, 6, 7, 8, and 34. The hard
  deadline arrived after chunk 45 ingest and its periodic evaluation but before its hold-out,
  adversarial work, and resume commit. Consequently `var/train_resume.json` safely lists 44
  completed chunks and chunk 45 as current; continuing from it would replay chunk 45. These
  replays mean this run is repair evidence, **not** a clean throughput or quality benchmark. The
  hard alarm bypassed trainer finalization, so both resume and metrics JSON retain a stale
  `status=running`; post-deadline `screen`, process, and listener checks are the source of truth for
  lifecycle state.
- Final persisted observations: the last periodic probe reported 40,184 ingested tokens. The last
  fully paused segment (through chunk 43) reported 39,404 tokens / 39,232 n-grams and a Cheetah
  Top-K hit ratio of approximately 37.02%. Cheetah produced no transport, reducer, or availability
  error during the monitored window.
- Effective post-end-marker-fix evaluation (metrics events 128-169): 16 evaluations / 64 generated
  samples, all quality-gated. Empty responses, complete prompt tags, `Emotion:` frame labels, and
  exact or normalized end-marker leaks were each 0/64. Mean generated perplexity was 505.66
  (median 75.43, range 4.17-1,861.87), mean reference perplexity 238.19, mean length ratio 0.0612,
  and mean recorded quality score 0.5528. The quality score is not credible as a standalone
  usability measure here: 21/64 samples (32.8%) were the identical visible safety backstop, 21/64
  matched legacy dependency-field vocabulary, 4/64 retained partial `user|` scaffold fragments,
  and 34/64 contained JSON-like punctuation fragments.
- Held-out chunks 34-44 were similarly poor: 17/44 samples (38.6%) used the same safety backstop,
  12/44 matched dependency-field residue, and every sample remained severely too short. Chunk 43
  temporarily collapsed to 3/4 identical backstops (mean generated perplexity 1,233.85; length
  ratio 0.0352); chunk 44 recovered to 1/4 backstops but remained incoherent. The final chunk-45
  periodic probe used one backstop and reported generated perplexity 480.63, length ratio 0.0663,
  and quality score 0.5423.
- Artifacts: `var/eval_logs/train-8h-20260723-231612.log`,
  `var/eval_logs/train-8h-20260723-231612.json`, `var/train_resume.json`, and
  `var/db_slm.sqlite3`. Runtime artifacts are ignored and are not implementation owners.
- Verification after deadline: `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v`
  passed 30 tests; `PYTHONPATH=src .venv/bin/python scripts/run_paraphraser_regression.py` passed
  6/6 cases; `py_compile` passed for every changed Python source/test; and `git diff --check`
  passed.
- Unresolved quality gap: the original empty-response and full-tag/end-marker exposure defects are
  fixed, but the accumulated model state still produces short, incoherent text, frequent
  backstop collapse, and fragments learned during earlier dependency/scaffold contamination.
  A clean isolated-database rerun is required before attributing any quality trend to the repaired
  pipeline.

## 2026-07-23 - External Cheetah repair and emotion-data validation

- Scope: validated the new `cheetah-db/` submodule boundary, repaired generic storage/concurrency
  failures upstream in Cheetah commit `6866be9`, updated the DB-SLM adapter to Cheetah's canonical
  job API, and exercised training plus inference without `--backonsqlite`.
- Baseline reproduction on submodule commit `76a9786`: three equal-sized concurrent client inserts
  all read back the last payload and allocated only one value slot. A 512-write shared-prefix
  `PAIR_SET` run returned success for every request but retained only 57 keys (455 missing).
- Fixed live-server regression: 512 concurrent inserts plus pair registrations completed with
  `failures=0`, `missing=0`, `wrong_payload=0`; `PAIR_SCAN` and `PAIR_REDUCE` each saw all 512 rows.
  The adapter also decoded live count, probability, and continuation reducer payloads through
  `JOB submit` → `JOB status` → `JOB fetch`.
- Go verification: `go build ./...`, `go vet ./...`, `go test -count=1 ./src`, and
  `go test -race -count=1 ./src` all passed. The reduced adaptive benchmark
  (`CHEETAHDB_ADAPTIVE_BENCH=1 CHEETAHDB_ADAPTIVE_BENCH_KEYS=2000 go test -run
  TestAdaptivePairIndexBenchmark -count=1 -v ./src`) passed in 34.93 seconds. Stride 2 improved
  fixed/adaptive insert time from 5.671 s to 3.489 s and walk time from 865 ms to 2 ms; reported
  apparent storage fell from 305.8 MiB to 2.1 MiB (allocated storage 174 MiB to 1.8 MiB).
- Training command (bounded `screen` session, 1,200-second timeout):
  `DBSLM_BACKEND=cheetah-db DBSLM_CHEETAH_DATABASE=emotion_e2e_20260723 python3.11 src/train.py
  datasets/emotion_data.json --db /tmp/lmdb-emotion-e2e-20260723.sqlite3 --ngram-order 3
  --eval-interval 1000000 --json-chunk-size 20 --max-json-lines 20 --profile-ingest
  --metrics-export /tmp/lmdb-emotion-train-e2e-20260723.json`.
- Training result: 20 actual records / 824,988 staged bytes produced 380,200 tokens and 380,198
  n-grams. Ingest took 141.49 seconds, total runtime was 206.82 seconds, and prediction-table
  training completed 431 updates. The metrics file ended with `status=success`; neither adapter
  disable, reducer timeout, legacy reducer fallback, nor SQLite fallback occurred.
- Persisted Cheetah state: the named database occupied approximately 52 MiB. Post-training summaries
  reported `ctx:` 9,175 rows / 140,404 bytes / depth 8 and `cont:` 1,596 rows / 19,152 bytes /
  depth 4. The reducer payload cache reported 658 hits and zero misses during the inspection.
- Inference command reused the same server/database:
  `DBSLM_BACKEND=cheetah-db DBSLM_CHEETAH_DATABASE=emotion_e2e_20260723 python3.11 src/run.py
  --db /tmp/lmdb-emotion-e2e-20260723.sqlite3 --ngram-order 3 --prompt
  "How can curiosity help someone navigate an ethical dilemma?" --max-response-words 80`.
  It completed in 9.68 seconds and returned exactly one `|USER|` / `|RESPONSE|` / `|TAGS|` frame
  without nested or leaked internal scaffold tags. The answer remained generic, as expected from a
  20-record validation slice; quality and Top-K latency measurement remain for the 250/1,000-record
  evaluation-enabled follow-up in `NEXT_STEPS.md`.
- Ephemeral artifacts from this run were written to
  `/tmp/lmdb-emotion-{train,infer}-e2e-20260723.*`,
  `/tmp/lmdb-emotion-e2e-20260723.sqlite3`, and
  `/tmp/cheetah-emotion-e2e-20260723/emotion_e2e_20260723`.

## 2025-11-10 - Smoke Train (python3.11)

- Command: `python3.11 src/train.py datasets/emotion_data.json --db var/smoke-train-run.sqlite3 --reset --json-chunk-size 120 --max-json-lines 400 --eval-interval 1500 --eval-samples 2 --eval-pool-size 40 --profile-ingest`, followed by `python3.11 src/run.py --db var/smoke-train-run.sqlite3 --prompt "Summarize how the DB-SLM handles short validation runs." --user smoke-test --agent db-slm`.
- Metrics file: `var/eval_logs/train-20251110-215404.json` (success). Ingested 882,338 tokens (882,330 windows) across 4 held-out chunks; 200 probe batches recorded.
- Aggregate probe stats: quality avg/best/worst = 0.5991 / 0.6709 / 0.4894, structure_variety = 0.3169 / 0.4079 / 0.1474, common_token_penalty = 0.3040 / 0.3779 / 0.2613. Flagged samples: 171 / 267 (64.0%). Last probe ("880500 ingested tokens") scored quality 0.6104, structure variety 0.3657, penalty 0.2758.
- Qualitative notes: structural metrics already depress repetitive scaffold openings ("Zooming in..." etc.). Most probes hit the retry budget because the first candidates remained flagged, so increasing pool diversity or loosening the per-sample penalty might speed smoke runs. The REPL reply retained the tagged frame (`|USER|`, `|RESPONSE|`, `|TAGS|`) as expected.

## 2025-11-12 - cheetah-db benchmark (CHEETAHDB_BENCH)

- Command: `cd cheetah-db && CHEETAHDB_BENCH=1 CHEETAHDB_BENCH_DURATION=30s go test -run TestCheetahDBBenchmark -count=1 -v ./src`.
- Log: `var/eval_logs/cheetah_db_benchmark_20251112-130623.log`.
- Snapshot @ 25 s (24 workers, 256 B payloads): inserts=616, reads=509, pair_set=258, pair_get=158, pair_scan=62, errors=0 → ~64 total ops/s before the idle tail. Pair scans now show up explicitly so pagination stays covered.
- Warmup now seeds 512 inserts + `4 * workers` pair registrations, so the benchmark no longer generates transient `pair_get`/`pair_scan` misses.

## 2025-11-12 - cheetah-db benchmark (45s / 32 workers)

- Command: `cd cheetah-db && CHEETAHDB_BENCH=1 CHEETAHDB_BENCH_DURATION=45s CHEETAHDB_BENCH_WORKERS=32 go test -run TestCheetahDBBenchmark -count=1 -v ./src`.
- Log: `var/eval_logs/cheetah_db_benchmark_20251112-164324.log`.
- Snapshot timeline (value size 256 B): total_qps=90.4 @5s, 84.0 @10s, 73.2 @15s, 67.4 @20s, 63.8 @25s, 60.9 @30s, 59.0 @35s, 56.0 @40s, 10.9 during the stop drain (220.8 s). Inserts=1002, reads=663, pair_set=346, pair_get=265, pair_scan=123, errors=0.
- Higher concurrency kept pair scans saturated without introducing pagination errors; the tail slowdown is purely the graceful shutdown window.

## 2025-11-12 - cheetah-db benchmark (30s / 24 workers rerun)

- Command: `cd cheetah-db && CHEETAHDB_BENCH=1 CHEETAHDB_BENCH_DURATION=30s CHEETAHDB_BENCH_WORKERS=24 go test -run TestCheetahDBBenchmark -count=1 -v ./src`.
- Log: `var/eval_logs/cheetah_db_benchmark_20251112-164803.log`.
- Snapshot timeline: total_qps=95.6 @5s, 87.6 @10s, 78.7 @15s, 72.0 @20s, 66.7 @25s, 12.4 while draining at 148.5 s. Final counters: inserts=760, reads=528, pair_set=276, pair_get=186, pair_scan=94, errors=0.
- Confirms the warmup/pagination fixes at the default worker count: each 5-second bucket included pair scans and no reducer/EOF errors surfaced.

## 2025-11-12 - cheetah-only smoke ingest (30 min budget)

- Command: `DBSLM_BACKEND=cheetah-db python3.11 src/train.py datasets/emotion_data.json --db /tmp/db_slm_smoke.sqlite3 --ngram-order 3 --eval-interval 2000 --json-chunk-size 250 --max-json-lines 1000` (run inside WSL tmux `cheetah_smoke`, cheetah server running in `cheetahdb` tmux).
- Logs/artifacts: trainer trace in `var/cheetah_smoke_train_20251112-205914.log`; cheetah server stdout in `var/cheetah-server-linux.log`. Metrics JSON did not flush because we stopped the run exactly at the 30-minute mark per policy.
- Runtime/highlights: chunk `datasets/emotion_data.json#chunk1` finished (`543,747` tokens → `543,745` n-grams) around +297 s; the remaining time was spent on evaluation probes (6 prompts every 2k tokens) until the timeout cutoff at +1,794 s.
- Probe snapshots (quality, lex, ROUGE-L, ppl(gen), ppl(ref), sim, len_ratio) steadily improved: `0.59 / 0.14 / 0.12 / 1.83k / 9.2 / 0.69 / 0.91` @20k tokens, `0.60 / 0.15 / 0.11 / 1.79k / 8.7 / 0.65 / 0.94` @128k tokens. The quality gate still flags most samples due to low structure variety, so probes keep retrying until the 2-attempt budget is exhausted.
- Hot-path + latency: `Disabling cheetah hot-path adapter: pair_reduce counts failed` fired immediately, so the decoder fell back to SQLite and the observed cheetah Top-K hit ratio remained `0%`. Decoder latency percentiles were unavailable for this historical run; the external Cheetah repair and successful rerun are recorded in the 2026-07-23 section above.
