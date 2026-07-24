from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from db_slm.evaluation import EvalLogWriter, EvaluationSampleResult


class EvalLogWriterTests(unittest.TestCase):
    def test_log_batch_records_sample_without_printing_raw_prompt_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            writer = EvalLogWriter(Path(tmp_dir) / "metrics.json", {})
            sample = EvaluationSampleResult(
                index=1,
                label="hold-out",
                prompt="|USER|: prompt\n|RESPONSE|: ",
                reference="reference",
                generated="generated",
                context_tokens={},
                metrics={"quality_score": 0.5},
            )
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                writer.log_eval_batch("hold-out", [sample], {"quality_score_mean": 0.5})

            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(writer.events[0]["samples"][0]["generated"], "generated")
            snapshot = json.loads((Path(tmp_dir) / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(snapshot["status"], "running")
            self.assertIsNone(snapshot["completed_at"])
            self.assertEqual(snapshot["events"][0]["samples"][0]["generated"], "generated")

            resumed = EvalLogWriter(
                Path(tmp_dir) / "metrics.json",
                {"pid": 2},
                append_existing=True,
            )
            self.assertEqual(resumed.run_id, writer.run_id)
            self.assertEqual(len(resumed.events), 1)
            self.assertEqual(resumed.events[0]["samples"][0]["generated"], "generated")


if __name__ == "__main__":
    unittest.main()
