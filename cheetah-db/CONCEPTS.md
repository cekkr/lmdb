This the base code of project "cheetah" for a ultra-rapid fast access database. It has to be re-adapted to work as database engine for LMDB project with the name of "cheetah-db". 

At its base, cheetah allows to extremly fast access to data from a key using a perfectly compartmentalized key cataloging by byte size, fast tree access algorithms, and various levels of caching.

For the "lmdb" version to be implemented, it must be adapted as best as possible to the needs of the Python project, further implementing:
- Rapid statistical computation (and associated caching)
- Rapid key contextualization: this requires byte-based tokenization of values ​​and efficient auto-sorting of dynamic sets (more specific probabilities can be obtained by delving deeper into the context of the prediction, without requiring ad hoc retraining on the Python side) of vector matrices and associated probabilistic links.
- This requires brute-force analysis of the data relationships in the DB (as well as efficient single-ordering of matrices to maintain fixed references), maximizing the rapid data access speed and the use of multiple hard disk files to retrieve and compare data and obtain more relevant statistical results.

### Context relativism contract

- Python now emits every context twice: once under `ctx:<hash>` (payload = order + token ids) and
  again under `ctxv:<vector>` where `<vector>` is the Absolute Vector Order encoding of the nested
  token structure. Deterministic ordering is critical so that a client-provided tree such as
  `[[[12,45],[43,21]], [[643,23]]]` hashes to the exact same byte slice no matter how it was
  originally ingested.
- `PAIR_SCAN ctxv:` already provides everything needed to service `engine.context_relativism()`:
  iterate the namespace, look up the referenced payload via `READ`, and re-use the Top-K entry via
  `PAIR_GET topk:<order>` when present. Future Go-native reducers can inline that second hop.

### Reducer RPCs

- Reducer responses now inline their payloads:
  `SUCCESS,reducer=<mode>,count=N,items=<hex_key>:<abs_key>:<base64_payload>`. Clients never need to
  issue a follow-up `READ`.
- `PAIR_REDUCE counts cnt:<order>` walks the `cnt:` namespace and returns the serialized follower
  list encoded by `CheetahSerializer.encode_counts` (version byte, order, follower count, followed by
  `(token_id, count)` tuples in big-endian form).
- `PAIR_REDUCE probabilities prob:<order>` streams quantized log-probabilities and backoff alphas.
  Each payload includes `version (1 byte)`, `order (1 byte)`, `entry_count (uint16)`, then per-entry
  `(token_id uint32, q_logprob uint8, backoff_alpha uint16 | 0xFFFF for None)`.
- `PAIR_REDUCE continuations cont:` exposes Level 1 continuation metadata. Payload layout:
  `version (1 byte)`, `token_id (uint32)`, `num_contexts (uint32)`.
- Upcoming reducers (`PAIR_REDUCE cache`, Level 2 bias presets, queue-drain snapshots) should follow
  the same command grammar (`PAIR_REDUCE <mode> <prefix> [limit]`) and document their payload
  layouts here so both the CLI and TCP clients can decode them safely.
