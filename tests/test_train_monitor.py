from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from train import InferenceMonitor, build_parser, iter_json_chunks


class InferenceMonitorSchedulingTests(unittest.TestCase):
    def test_dependency_metadata_stays_outside_generation_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "records.ndjson"
            path.write_text(
                json.dumps({"prompt": "Explain empathy.", "response": "It supports care."}) + "\n",
                encoding="utf-8",
            )
            dependency = object()
            with patch("train.build_dependency_layer", return_value=dependency):
                chunks = list(iter_json_chunks(path, 1, 0, 0.0, None))

        self.assertEqual(len(chunks), 1)
        self.assertNotIn("DependencyLayer:", chunks[0].train_text)
        self.assertNotIn('"lemma"', chunks[0].train_text)
        assert chunks[0].prediction_records is not None
        self.assertIs(chunks[0].prediction_records[0].prompt_dependencies, dependency)
        self.assertIs(chunks[0].prediction_records[0].response_dependencies, dependency)

    def test_parser_accepts_recoverable_runtime_budget(self) -> None:
        args = build_parser(":memory:").parse_args(["--max-runtime-seconds", "2400"])

        self.assertEqual(args.max_runtime_seconds, 2400)

    def test_large_chunk_runs_one_cycle_and_skips_crossed_threshold_backlog(self) -> None:
        monitor = InferenceMonitor.__new__(InferenceMonitor)
        monitor.interval = 5_000
        monitor.next_threshold = 5_000
        monitor.dataset = [object()]
        cycles: list[int] = []
        monitor._run_cycle = cycles.append  # type: ignore[method-assign]

        monitor.maybe_run(26_000)

        self.assertEqual(cycles, [26_000])
        self.assertEqual(monitor.next_threshold, 30_000)

        monitor.maybe_run(29_999)
        self.assertEqual(cycles, [26_000])

        monitor.maybe_run(30_000)
        self.assertEqual(cycles, [26_000, 30_000])
        self.assertEqual(monitor.next_threshold, 35_000)


if __name__ == "__main__":
    unittest.main()
