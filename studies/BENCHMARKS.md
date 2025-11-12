# Benchmarks

> The smoke harness now appends queue-drain snapshots automatically whenever the quality queue
> crosses its threshold. Look for sections titled “Queue Drain (auto smoke harness)” for those runs.

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

## 2025-11-12 - cheetah-db benchmark (45s / 32 workers)

- Command: `cd cheetah-db && CHEETAHDB_BENCH=1 CHEETAHDB_BENCH_DURATION=45s CHEETAHDB_BENCH_WORKERS=32 go test -run TestCheetahDBBenchmark -count=1 -v`.
- Log: `var/eval_logs/cheetah_db_benchmark_20251112-164324.log`.
- Snapshot timeline (value size 256 B): total_qps=90.4 @5s, 84.0 @10s, 73.2 @15s, 67.4 @20s, 63.8 @25s, 60.9 @30s, 59.0 @35s, 56.0 @40s, 10.9 during the stop drain (220.8 s). Inserts=1002, reads=663, pair_set=346, pair_get=265, pair_scan=123, errors=0.
- Higher concurrency kept pair scans saturated without introducing pagination errors; the tail slowdown is purely the graceful shutdown window.

## 2025-11-12 - cheetah-db benchmark (30s / 24 workers rerun)

- Command: `cd cheetah-db && CHEETAHDB_BENCH=1 CHEETAHDB_BENCH_DURATION=30s CHEETAHDB_BENCH_WORKERS=24 go test -run TestCheetahDBBenchmark -count=1 -v`.
- Log: `var/eval_logs/cheetah_db_benchmark_20251112-164803.log`.
- Snapshot timeline: total_qps=95.6 @5s, 87.6 @10s, 78.7 @15s, 72.0 @20s, 66.7 @25s, 12.4 while draining at 148.5 s. Final counters: inserts=760, reads=528, pair_set=276, pair_get=186, pair_scan=94, errors=0.
- Confirms the warmup/pagination fixes at the default worker count: each 5-second bucket included pair scans and no reducer/EOF errors surfaced.

## 2025-11-12 - cheetah-only smoke ingest (30 min budget)

- Command: `DBSLM_BACKEND=cheetah-db python3.11 src/train.py datasets/emotion_data.json --db /tmp/db_slm_smoke.sqlite3 --ngram-order 3 --eval-interval 2000 --json-chunk-size 250 --max-json-lines 1000` (run inside WSL tmux `cheetah_smoke`, cheetah server running in `cheetahdb` tmux).
- Logs/artifacts: trainer trace in `var/cheetah_smoke_train_20251112-205914.log`; cheetah server stdout in `var/cheetah-server-linux.log`. Metrics JSON did not flush because we stopped the run exactly at the 30-minute mark per policy.
- Runtime/highlights: chunk `datasets/emotion_data.json#chunk1` finished (`543,747` tokens → `543,745` n-grams) around +297 s; the remaining time was spent on evaluation probes (6 prompts every 2k tokens) until the timeout cutoff at +1,794 s.
- Probe snapshots (quality, lex, ROUGE-L, ppl(gen), ppl(ref), sim, len_ratio) steadily improved: `0.59 / 0.14 / 0.12 / 1.83k / 9.2 / 0.69 / 0.91` @20k tokens, `0.60 / 0.15 / 0.11 / 1.79k / 8.7 / 0.65 / 0.94` @128k tokens. The quality gate still flags most samples due to low structure variety, so probes keep retrying until the 2-attempt budget is exhausted.
- Hot-path + latency: `Disabling cheetah hot-path adapter: pair_reduce counts failed` fired immediately, so the decoder fell back to SQLite and the observed cheetah Top-K hit ratio remained `0%`. Decoder latency percentiles are unavailable for this run; rerun after fixing `PAIR_REDUCE counts` so the metrics writer can flush `var/eval_logs/train-*.json`.

