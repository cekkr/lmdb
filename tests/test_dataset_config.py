from __future__ import annotations

import unittest
from pathlib import Path

from db_slm.dataset_config import load_dataset_config


class DatasetConfigPromptCompositionTests(unittest.TestCase):
    def test_trailing_context_and_canonical_tag_match_training_frame(self) -> None:
        config = load_dataset_config(Path("datasets/emotion_data.json"))
        payload = {
            "prompt": "How does Envy affect leadership?",
            "response": "It can motivate or undermine a team.",
            "emotion": "Envy",
        }

        self.assertEqual(
            config.compose_prompt(payload),
            "\n".join(
                [
                    "|USER|: How does Envy affect leadership?",
                    "Emotion: Envy",
                    "|CTX|:emotion:envy",
                ]
            ),
        )
        self.assertEqual(
            config.prompt_tag_tokens(),
            ("|USER|:", "|RESPONSE|:", "Emotion:", "|CTX|:"),
        )

    def test_preface_context_remains_before_user_prompt(self) -> None:
        config = load_dataset_config(Path("datasets/GPTeacher.json"))
        payload = {
            "instruction": "Reverse the input.",
            "input": "Hello",
            "response": "olleH",
        }

        self.assertEqual(
            config.compose_prompt(payload),
            "|INSTRUCTION|: Reverse the input.\n|USER|: Hello",
        )


if __name__ == "__main__":
    unittest.main()
