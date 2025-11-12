# Benchmarks

## 2025-11-10 - Smoke Train (python3.11)

- Command: `python3.11 src/train.py datasets/emotion_data.json --db var/smoke-train-run.sqlite3 --reset --json-chunk-size 120 --max-json-lines 400 --eval-interval 1500 --eval-samples 2 --eval-pool-size 40 --profile-ingest`, followed by `python3.11 src/run.py --db var/smoke-train-run.sqlite3 --prompt "Summarize how the DB-SLM handles short validation runs." --user smoke-test --agent db-slm`.
- Metrics file: `var/eval_logs/train-20251110-215404.json` (success). Ingested 882,338 tokens (882,330 windows) across 4 held-out chunks; 200 probe batches recorded.
- Aggregate probe stats: quality avg/best/worst = 0.5991 / 0.6709 / 0.4894, structure_variety = 0.3169 / 0.4079 / 0.1474, common_token_penalty = 0.3040 / 0.3779 / 0.2613. Flagged samples: 171 / 267 (64.0%). Last probe ("880500 ingested tokens") scored quality 0.6104, structure variety 0.3657, penalty 0.2758.
- Qualitative notes: structural metrics already depress repetitive scaffold openings ("Zooming in..." etc.). Most probes hit the retry budget because the first candidates remained flagged, so increasing pool diversity or loosening the per-sample penalty might speed smoke runs. The REPL reply retained the tagged frame (`|USER|`, `|RESPONSE|`, `|TAGS|`) as expected.

## 2025-11-12 - cheetah-db benchmark (CHEETAHDB_BENCH)

- Command: `cd cheetah-db && CHEETAHDB_BENCH=1 CHEETAHDB_BENCH_DURATION=30s go test -run TestCheetahDBBenchmark -count=1 -v`.
- Log: `var/eval_logs/cheetah_db_benchmark_20251112-130623.log`.
- Snapshot @ 25 s (24 workers, 256 B payloads): inserts=616, reads=509, pair_set=258, pair_get=158, pair_scan=62, errors=0 → ~64 total ops/s before the idle tail. Pair scans now show up explicitly so pagination stays covered.
- Warmup now seeds 512 inserts + `4 * workers` pair registrations, so the benchmark no longer generates transient `pair_get`/`pair_scan` misses.
