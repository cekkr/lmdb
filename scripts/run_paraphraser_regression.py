#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Tuple

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from db_slm.metrics import lexical_overlap
from db_slm.pipeline import SimpleParaphraser


def load_cases(path: Path) -> list[dict]:
    cases: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"[paraphraser] Invalid JSON on line {line_no}: {exc}") from exc
    if not cases:
        raise SystemExit(f"[paraphraser] No regression cases found in {path}")
    return cases


def run_case(paraphraser: SimpleParaphraser, case: dict) -> Tuple[bool, str]:
    expectation = case["expectation"]
    prompt = case["prompt"]
    response = case["response"]
    if expectation == "guard":
        assert paraphraser.should_guard(prompt), f"{case['id']}: expected guard but paraphraser wanted rewrites"
        rewritten = paraphraser.rephrase(prompt, response)
        overlap = lexical_overlap(response, rewritten)
        return True, f"guarded (overlap={overlap:.2f})"

    if expectation == "rewrite":
        rewritten = paraphraser.rephrase(prompt, response)
        assert not paraphraser.should_guard(prompt), f"{case['id']}: unexpectedly guarded prompt"
        assert rewritten.strip(), f"{case['id']}: paraphraser returned empty text"
        assert rewritten.strip() != response.strip(), f"{case['id']}: rewrite did not change the response"
        overlap = lexical_overlap(response, rewritten)
        return True, f"rewritten (overlap={overlap:.2f})"

    raise SystemExit(f"[paraphraser] Unknown expectation '{expectation}' in case {case['id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the SimpleParaphraser guard rails.")
    parser.add_argument(
        "--dataset",
        default="studies/paraphraser_regression.jsonl",
        help="Path to the JSONL regression dataset (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed to stabilize paraphraser variation (default: %(default)s).",
    )
    args = parser.parse_args()
    random.seed(args.seed)
    paraphraser = SimpleParaphraser()
    cases = load_cases(Path(args.dataset))
    results: list[str] = []
    for case in cases:
        passed, info = run_case(paraphraser, case)
        status = "PASS" if passed else "FAIL"
        reason = case.get("reason", "")
        suffix = f" - {reason}" if reason else ""
        print(f"[paraphraser] {status} {case['id']}: {info}{suffix}")
        results.append(status)
    total = len(results)
    failures = results.count("FAIL")
    print(f"[paraphraser] Completed {total} case(s) with {failures} failure(s).")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
