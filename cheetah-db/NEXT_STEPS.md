# To do:
- Implement the shard-health gossip + RPC fan-out layer so the fork scheduler can actively place/move shards across remote cheetah-server instances instead of staying single-node.
- Replace the simulated WebGPU probability merger with a real GPU/WebGPU backend (Vulkan/WGSL or Dawn) and persist per-host benchmarks to auto-select accelerators.
- Wire prediction-table training into ingest so context matrices refresh automatically without manual `PREDICT_TRAIN` loops.
