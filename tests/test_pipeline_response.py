from __future__ import annotations

import unittest

from db_slm.pipeline import DBSLMEngine, ResponseBackstop


class _Memory:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str, str]] = []

    def log_message(self, conversation_id: str, role: str, content: str) -> None:
        self.messages.append((conversation_id, role, content))

    def context_window(self, conversation_id: str) -> str:
        return "\n".join(content for convo, _, content in self.messages if convo == conversation_id)


class _Tokenizer:
    decoded = "|END|"

    def encode(self, text: str, **_: object) -> list[int]:
        return [1] if text else []

    def decode(self, token_ids: list[int]) -> str:
        return self.decoded if token_ids else ""


class _Decoder:
    def __init__(self) -> None:
        self.calls = 0

    def decode(self, *_: object, **__: object) -> list[int]:
        self.calls += 1
        return [99]


class _ContextWindows:
    def enabled(self) -> bool:
        return False


class _Cache:
    def update(self, *_: object) -> None:
        return None


class _Vocabulary:
    def token_id(self, _: str) -> int:
        return 0


class _LowResourceHelper:
    def maybe_paraphrase(self, _: str, response: str, **__: object) -> str:
        return response


class PipelineResponseTests(unittest.TestCase):
    @staticmethod
    def _engine() -> DBSLMEngine:
        engine = DBSLMEngine.__new__(DBSLMEngine)
        engine.memory = _Memory()
        engine.tokenizer = _Tokenizer()
        engine.vocab = _Vocabulary()
        engine.cache = _Cache()
        engine.store = type("_Store", (), {"order": 5})()
        engine.context_windows = _ContextWindows()
        engine.decoder = _Decoder()
        engine._low_resource_helper = _LowResourceHelper()
        engine._response_backstop = ResponseBackstop()
        engine._prompt_tag_retry_attempts = 3
        engine._prompt_tag_token_ids = set()
        engine._run_concept_layer = lambda *_: None
        engine._contains_prompt_artifacts = lambda _: False
        return engine

    def test_raw_response_retries_marker_only_decode_then_emits_visible_backstop(self) -> None:
        engine = self._engine()

        response = engine.respond(
            "conversation",
            "|USER|: Explain gratitude.\n|RESPONSE|:",
            scaffold_response=False,
            rng_seed=7,
        )

        self.assertEqual(engine.decoder.calls, 3)
        self.assertTrue(response.strip())
        self.assertGreaterEqual(len(response.split()), 12)
        self.assertNotIn("|END|", response)
        self.assertNotIn("|USER|", response)
        self.assertNotIn("|RESPONSE|", response)
        self.assertEqual(engine.memory.messages[-1], ("conversation", "assistant", response))

    def test_raw_response_strips_spaced_case_insensitive_end_marker(self) -> None:
        engine = self._engine()
        engine.tokenizer.decoded = "A visible answer.| end | trailing control text"

        response = engine.respond(
            "conversation",
            "|USER|: Explain sympathy.\n|RESPONSE|:",
            scaffold_response=False,
            rng_seed=9,
        )

        self.assertEqual(engine.decoder.calls, 1)
        self.assertEqual(response, "A visible answer.")
        self.assertNotIn("end", response.lower())

    def test_raw_response_retries_spaced_marker_only_decode(self) -> None:
        engine = self._engine()
        engine.tokenizer.decoded = "| End |"

        response = engine.respond(
            "conversation",
            "|USER|: Explain sympathy.\n|RESPONSE|:",
            scaffold_response=False,
            rng_seed=10,
        )

        self.assertEqual(engine.decoder.calls, 3)
        self.assertTrue(response.strip())
        self.assertNotIn("| End |", response)

    def test_raw_response_retries_dependency_json_then_emits_clean_backstop(self) -> None:
        engine = self._engine()
        engine.tokenizer.decoded = (
            'helpful text, " lemma " : "help", " dep " : "ROOT"'
        )

        response = engine.respond(
            "conversation",
            "|USER|: Explain empathy.\n|RESPONSE|:",
            scaffold_response=False,
            rng_seed=11,
        )

        self.assertEqual(engine.decoder.calls, 3)
        self.assertTrue(response.strip())
        self.assertNotIn('" lemma "', response)
        self.assertNotIn('" dep "', response)
        self.assertGreaterEqual(len(response.split()), 12)

    def test_configured_plain_frame_label_requires_colon_to_be_an_artifact(self) -> None:
        aliases = DBSLMEngine._derive_prompt_tag_aliases(
            self._engine(),
            ["Emotion:"],
        )
        engine = self._engine()
        engine._prompt_tag_aliases = aliases

        contains_artifacts = lambda response: DBSLMEngine._contains_prompt_artifacts(
            engine,
            response,
        )
        self.assertTrue(contains_artifacts("Emotion: Surprise"))
        self.assertTrue(contains_artifacts("emotion : surprise"))
        self.assertFalse(
            contains_artifacts(
                "The emotion can influence attention without repeating the frame."
            )
        )


if __name__ == "__main__":
    unittest.main()
