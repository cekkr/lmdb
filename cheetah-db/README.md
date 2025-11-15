# cheetah-db

High-throughput key/value store with a trie-backed pair table purpose-built for statistical data
pipelines. cheetah-db ingests byte-encoded contexts, n-gram payloads, and arbitrary binary blobs,
keeps them partitioned by value size, and exposes TCP + CLI commands that stream results with
bounded memory use. The engine targets workloads where millions of probabilities, counters, or
other dense analytical slices must be served with predictable latency.

## Highlights

- **Byte-faithful layout.** Every entry is cataloged by byte length, table ID, and entry index, so
  reads turn into deterministic `ReadAt` calls instead of scanning variable-length payloads.
- **Trie-indexed pair table.** The `pairs/` directory holds fixed-size nodes that behave like a
  prefix tree. `PAIR_SCAN` and `PAIR_REDUCE` walk that trie, making namespace sweeps and reducer
  workloads practical even when the keyspace spans billions of n-gram contexts.
- **Payload caching.** `database.go` keeps a bounded cache (defaults: 16k entries ≈64 MB) keyed by
  `<value_size, table_id, entry_id>` so hot payloads never hit disk. Tune it with
  `CHEETAH_PAYLOAD_CACHE_ENTRIES`, `CHEETAH_PAYLOAD_CACHE_MB`, or
  `CHEETAH_PAYLOAD_CACHE_BYTES`, or disable caching entirely by setting any of them to `0`.
- **Multi-tenant databases.** `engine.go` multiplexes logical databases under `cheetah_data/<name>`
  and exposes them over both CLI and TCP, making it easy to isolate experiments or pilot rollouts.
- **Reducer streaming.** Reducers stream inline payloads through a bounded worker pool, overlap disk
  reads with encoding work, and emit cursor tokens so callers can page through arbitrarily large
  namespaces without restarting the command.

## Architecture at a Glance

- `main.go` boots the TCP listener (`0.0.0.0:4455` by default) plus the local CLI and routes commands
  to database handles returned by `engine.GetDatabase(name)`.
- `engine.go` lazily instantiates `Database` structs backed by `cheetah_data/<dbname>` and ensures all
  tables (main keys, values, recycling queues, pair trie) flush cleanly at shutdown.
- `database.go` orchestrates CRUD operations:
  - `MainKeysTable` stores compact metadata describing payload size + pointer offsets.
  - `ValuesTable` files hold fixed-width blobs grouped by byte length and table ID so offsets remain
    arithmetic instead of scan-based.
  - `RecycleTable` files keep tombstoned slots per value size so inserts can reuse space without
    compaction pauses.
  - `PairTable` nodes store child pointers and terminal flags independently, unlocking
    prefix-sharing namespaces such as `ctx:`, `ctxv:`, `prob:`, or `meta:`.
- `server.go` accepts newline-delimited commands (`INSERT`, `READ`, `PAIR_SCAN`, `PAIR_REDUCE`, …)
  over TCP so external adapters can talk to the engine without embedding Go code.

## Heavy Statistical Workloads

cheetah-db was designed to stage dense statistical datasets such as n-gram probabilities,
continuation metadata, and concept caches:

- Use dedicated namespaces (e.g., `ctx:`, `prob:<order>`, `cont:`, `meta:`) to keep context metadata,
  quantized probability tables, and continuation payloads separated yet streamable.
- Mirror Top-K caches, follow-up penalties, or other heavy slices directly into the trie so cache
  lookups never re-open SQLite or object stores.
- When benchmarking reducers, export `CHEETAHDB_BENCH=1` and run
  `go test -run TestCheetahDBBenchmark -count=1 -v` for reproducible throughput snapshots.
- Keep `cheetah-db/AI_REFERENCE.md` handy for cache-sizing tables, tmux/screen launch recipes, and
  namespace troubleshooting tips when you run sustained ingest/eval loops.

## Memory & SSD Guidelines

- Start the server with a payload-cache budget sized for your workload. 64 MB works for small corpora
  while 128–256 MB keeps multi-GB statistical slices in RAM. Increase the byte budget before raising
  the entry count so the cache retains whole payloads.
- Inserts seed the cache and deletes invalidate their slots, so chaining ingest → reduction → decoder
  stages benefits from a single long-lived process. Avoid restarting the server between stages unless
  you want to profile cold starts.
- To prime the cache after restarts, issue low-limit `PAIR_SCAN ctx:` passes (following cursors) or
  scripted `READ` loops over the namespaces you are about to benchmark. This shifts the I/O churn
  into RAM, keeping SSD wear predictable.
- When profiling disk I/O, set `CHEETAH_PAYLOAD_CACHE_ENTRIES=0` (or the MB/byte variants) to disable
  caching entirely, then re-enable it immediately afterward for day-to-day runs.

## Building & Running

```bash
cd cheetah-db
bash build.sh              # produces ./cheetah-server
./cheetah-server           # interactive CLI + TCP listener
# or launch headless
CHEETAH_HEADLESS=1 ./cheetah-server
```

Environment variables:

- `CHEETAH_DATA_DIR` — root directory for database folders (defaults to `cheetah_data`).
- `CHEETAH_PAYLOAD_CACHE_ENTRIES` / `_MB` / `_BYTES` — cache tuning knobs.
- `CHEETAH_LOG_LEVEL` — set to `debug` for verbose reducer + trie traces.

The binary prints `[cheetah_data/default]>` when ready. Use `DATABASE <name>` (CLI) to switch between
logical databases or send the same command over TCP.

## Command Reference

```
INSERT:<size> <payload>         # create payload, returns abs key
READ <abs_key>                  # fetch payload by key
EDIT:<size> <abs_key> <payload> # overwrite payload in-place
PAIR_SET <hex_prefix> <payload> # map trie prefix to payload key
PAIR_SCAN <prefix> [limit]      # stream ordered namespace slices (cursors supported)
PAIR_REDUCE <mode> <prefix>     # stream reducer payloads (counts/probabilities/etc.)
DELETE <abs_key>                # tombstone entry
RECYCLE <value_size>            # report recycle stats per table
```

- Prefix strings (`ctx:`, `ctxv:`, `prob:2`, etc.) are treated as raw bytes; encode binary prefixes
  as `x<HEX>`.
- `PAIR_SCAN` replies include `items=<hex_prefix>:<abs_key>` pairs plus `next_cursor=<token>` when
  additional pages remain. Reissue the command with `CURSOR <token>` (TCP) or `PAIR_SCAN <prefix> <limit> <token>` (CLI) to continue.
- `PAIR_REDUCE` includes inline base64 payloads so reducers can hydrate counters/probabilities
  without extra `READ` calls. Each response also includes `next_cursor` when more items exist.

## Streaming Helpers

- `PAIR_SCAN <prefix> [limit]` favors namespace exhaustiveness: `PAIR_SCAN ctx: 100` walks the first
  100 contexts alphabetically, while `PAIR_SCAN * 0` streams the entire trie.
- `HotPathAdapter`-style integrations typically alternate `PAIR_SCAN` with `READ` to hydrate payloads;
  keep a cache of `<value_size, table_id, entry_id>` tuples nearby when building custom adapters.

## Reducer Hooks

- `PAIR_REDUCE counts ctx:` aggregates follower counts directly inside Go and emits the packed
  payloads inline, allowing MKNS-style reducers or other statistical aggregators to run without SQL.
- Custom reducers can be registered in Go (see `commands.go`). Each reducer receives the pair-trie
  iterator and can emit any payload format; clients decode the base64 payload per reducer contract.

## Operational Notes

- Prefer `screen` or `tmux` for long-lived sessions. Launch commands with explicit timeouts (≤30 min
  unless otherwise justified) so stalled reducers cannot block future sessions.
- `CHEETAH_HEADLESS=1` disables the interactive CLI while keeping the TCP listener up. When running in
  WSL or remote shells, pair it with `screen -dmS cheetahdb ...` and monitor `screen -ls` /
  `screen -wipe` before rebuilding.
- Rotate logs under `var/eval_logs/`—benchmark helpers emit files such as
  `cheetah_db_benchmark_<timestamp>.log` so you can diff throughput across cache sizes or reducer
  tweaks.

For deep operational checklists (tmux helpers, namespace triage, cache sizing matrices), see
`AI_REFERENCE.md` in this directory.
