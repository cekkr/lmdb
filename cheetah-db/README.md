# cheetah-db

Ultra-rapid key/value database engine being refit to serve as the low-latency adapter for the LMDB
Python project (`src/db_slm`). The original “cheetah” fork already implements a byte-oriented storage
engine in Go; this directory tracks the work needed to evolve it into a drop-in backend (`backend =
cheetah-db`) for the DB-SLM stack.

## Why it exists

- **Bridge for extreme-latency workloads.** SQLite remains the reference implementation, but it tops
  out when Level 1 counts, caches, and Level 2 conversation traces all compete for the same file.
  `cheetah-db` is meant to absorb those hot paths with a persistent, mmap-friendly layout and a
  purpose-built TCP/CLI front end.
- **Byte-faithful keying.** Each entry is cataloged by byte length and stored inside homogenous
  tables (`values_<size>_<tableID>.table`), keeping lookups O(1) with deterministic offsets.
- **Tree-indexed pair table.** The `pairs/` directory pre-allocates 256-entry pages that behave like
  a trie. This powers rapid pair/sequence lookups and will become the substrate for context hashes
  and vector-matrix anchors once the adapter is wired into DB-SLM.

## Current architecture snapshot

- `main.go` boots both a TCP server (`0.0.0.0:4455`) and a local CLI, multiplexing commands across
  multiple logical databases (`engine.GetDatabase(name)`).
- `engine.go` lazily instantiates `Database` handles under `cheetah_data/<dbname>` and guarantees
  graceful shutdown of every open table.
- `database.go` orchestrates CRUD operations, recycling, and the pair trie:
  - `MainKeysTable` (`main_keys.table`) stores the 6-byte entries that describe value size +
    location.
  - `ValuesTable` files store fixed-width blobs grouped by byte length and table ID; offsets are
    computed by entry index so reads never have to scan variable-length segments.
  - `RecycleTable` files keep tombstoned slots per byte length so inserts can reuse space without
    compacting.
  - `PairTable` nodes under `pairs/` encode byte-wise trees whose leaves can store absolute keys or
    child pointers, giving us a hardware-friendly trie for contextual lookups.
- `server.go` accepts newline-delimited commands over TCP (same grammar as the CLI) so we can script
  adapters before we embed the engine directly.

## Adapter goals for DB-SLM

To register `cheetah-db` as a DB adapter alongside SQLite and MariaDB (`DBSLMSettings.backend`),
the engine must grow the following behaviors:

1. **Rapid statistical computation + caching.** Level 1 relies on large fan-out aggregations (counts,
   continuation tables, quantized probabilities, Top-K ranks). `cheetah-db` must expose batched
   reducers or server-side stored procedures so the Python side can trigger recomputation without
   copying data back to SQLite first.
2. **Byte-based contextualization.** DB-SLM contexts are byte-tokenized hashes. The pair tables need
   to surface APIs that can auto-sort dynamic sets (e.g., context hashes → next token IDs) so
   probabilities can be refined by traversing deeper into the trie without retraining.
3. **Multi-file brute-force sweeps.** Level 2/3 materializations often need to scan entire matrix
   orderings. The engine must coordinate scans across parallel files, stream results, and keep hot
   references stable even when data is sharded across disks.

Document every milestone in `AI_REFERENCE.md` and flag missing capabilities in `NEXT_STEPS.md` so the
adapter status stays visible to future maintainers.

## Python bridge status

- `src/db_slm` now includes a `cheetah-db` hot-path adapter. Set `DBSLM_BACKEND=cheetah-db` (or
  `DBSLM_CHEETAH_MIRROR=1` to mirror without switching the primary backend) and start the Go server
  before running `src/train.py`/`src/run.py`.
- The trainer publishes every newly discovered context plus the Top-K quantized probabilities
  produced by the MKNS smoother through the TCP API (`INSERT`, `PAIR_SET`, etc.), so low-latency
  reads land in cheetah while SQLite remains authoritative for schema-wide updates.
- The decoder consults the cheetah mirror first when sampling candidates; if the Go service is
  offline or missing a context, it automatically falls back to SQLite. This gives us byte-faithful
  keying and immediate Top-K slices with no additional SQL load.
- Ordered trie streaming is now available via the `PAIR_SCAN` command and the
  `HotPathAdapter.scan_namespace()` helper, so DB-SLM can iterate over namespaces (contexts, cached
  Top-K buckets, etc.) without touching SQLite. The next integration step is to route Level 1 reads
  (context registry, Top-K coverage checks) through this iterator so SQLite/MariaDB can be removed.
- Absolute vector ordering is live: each context now gets a deterministic `ctxv:` alias derived from
  the nested token structure. `engine.context_relativism()` streams the corresponding contexts +
  ranked continuations directly from `PAIR_SCAN ctxv:`, and the decoder uses those slices whenever
  a Top-K entry is missing.
- `PAIR_REDUCE counts` aggregates follower counts in-place so MKNS rebuilds can pull entire context
  registries through TCP. The Python smoother mirrors those projections back into `cnt:<order>`
  namespaces after every rebuild, keeping Go + SQLite in sync without extra SQL.
- Metadata (context dimensions, decode profiles, cache lambdas) is persisted under the `meta:`
  namespace so new Python processes can cold-start without re-reading SQLite tables.
- Upcoming roadmap items (statistical reducers, ordered trie slices) should extend the same adapter
  so DB-SLM can eventually run entirely on the Go engine once Level 2/3 tables get equivalents.

## Build & run

```bash
cd cheetah-db
go test ./...        # once tests exist
bash build.sh        # produces ./cheetah-server
./cheetah-server     # starts TCP server + interactive CLI
```

Example CLI session:

```
[cheetah_data/default]> INSERT:5 hello
SUCCESS,key=1
[cheetah_data/default]> READ 1
SUCCESS,size=5,value=hello
[cheetah_data/default]> PAIR_SCAN ctx: 5
SUCCESS,count=1,items=6374783a:4
```

Over TCP you can send the same commands (newline-terminated). Use `DATABASE <name>` to switch logical
stores in either mode.

### Streaming helpers

- `PAIR_SCAN <prefix> [limit]` returns namespace-ordered slices without SQLite. Use `*` as the prefix
  to stream the entire trie or `x<HEX>` to target binary prefixes. Responses include `items=...`
  with `hex_value:abs_key` pairs so clients can fetch the referenced payloads via `READ`.
- The Python adapter exposes this via `CheetahClient.pair_scan()` and
  `HotPathAdapter.scan_namespace()`, which automatically expand namespace strings (e.g., `ctx:`) and
  strip the prefix from the returned values.

### Reducer hooks

- `PAIR_REDUCE <mode> <prefix> [limit]` piggybacks on the pair iterator but applies lightweight
  reducers server-side before returning results. The first implemented mode (`PAIR_REDUCE counts
  cnt:<order>`) streams raw follower counts for all contexts mirrored in `cnt:` namespaces so Python
  can rebuild MKNS statistics without hitting SQLite.
- Future reducers (`PAIR_REDUCE probs`, `PAIR_REDUCE queue-drain`, etc.) should follow the same
  union-by-mode pattern. Update `cheetah-db/CONCEPTS.md` when adding new modes so the RPC contract
  remains easy to discover from the Go side.

## Directory map

- `commands.go` – user-facing operations (INSERT/READ/EDIT/DELETE/PAIRSET/PAIRGET/etc.).
- `tables.go` – on-disk table implementations (`MainKeysTable`, `ValuesTable`, `RecycleTable`,
  `PairTable`).
- `types.go` – constants and binary layouts shared across tables.
- `helpers.go` – low-level utilities (value allocations, binary encoding helpers).
- `server.go` / `main.go` – TCP server + CLI wiring.

## Next steps

- Expand `PAIR_REDUCE` beyond counts (probabilities/backoff slices) so MKNS can stream quantized
  rows directly from Go.
- Plumb ordered sweeps for Level 2/3 namespaces so cache/bias/concept jobs never have to query
  SQLite once the metadata tables are fully mirrored.
- Capture decoder latency + Top-K hit-rate snapshots in this README after each cheetah-only smoke
  train so we can watch coverage trend toward 100%.
- Add regression tests around absolute-vector ordering and reducer semantics to guard the binary
  layouts before we start inlining cheetah inside the Python repo.
