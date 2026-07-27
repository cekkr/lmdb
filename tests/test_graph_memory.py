from __future__ import annotations

import base64
import json
import unittest

from db_slm.adapters.base import NullHotPathAdapter
from db_slm.adapters.cheetah import CheetahClient, CheetahHotPathAdapter
from db_slm.pipeline import _CORRECTION_Q_BIAS, DBSLMEngine
from db_slm.graph_memory import (
    GraphContextMemory,
    GraphContextSignal,
    content_terms,
    context_node_id,
    slugify,
    term_node_id,
    term_text_from_node_id,
)


class _FakeArc:
    def __init__(self, lemma: str, head: str, dep: str) -> None:
        self.token = lemma
        self.lemma = lemma
        self.head = head
        self.dep = dep
        self.pos = "NOUN"


class _FakeLayer:
    def __init__(self, groups: dict[str, tuple[str, ...]], arcs: tuple[_FakeArc, ...] = ()) -> None:
        self.backend = "fake"
        self.strong_token_groups = groups
        self.arcs = arcs
        self.token_count = len(arcs)


class _FakeRecord:
    def __init__(
        self,
        prompt: str,
        response: str,
        context_tokens: dict[str, str] | None = None,
        prompt_dependencies: _FakeLayer | None = None,
        response_dependencies: _FakeLayer | None = None,
    ) -> None:
        self.prompt = prompt
        self.response = response
        self.context_tokens = context_tokens or {}
        self.prompt_dependencies = prompt_dependencies
        self.response_dependencies = response_dependencies


class _ScriptedSocketClient(CheetahClient):
    """A CheetahClient whose transport is a scripted command -> response map."""

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        super().__init__(host="127.0.0.1", port=4455)
        self.commands: list[str] = []
        self._responses = responses or {}
        self.default_response = "SUCCESS"

    def _command(self, text: str) -> str | None:  # type: ignore[override]
        command = text.strip()
        self.commands.append(command)
        verb = command.split(" ", 1)[0]
        return self._responses.get(verb, self.default_response)

    def command_args(self, verb: str) -> dict[str, str]:
        for command in self.commands:
            if command.split(" ", 1)[0] != verb:
                continue
            args: dict[str, str] = {}
            for token in command.split(" ")[1:]:
                if "=" in token:
                    key, value = token.split("=", 1)
                    args[key] = value
            return args
        return {}


def _encode_payload(payload: object) -> str:
    return base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")


_RECALL_PAYLOAD = {
    "seeds": [
        {"term": "grateful", "matches": [{"id": "term:grateful", "score": 0.5, "match": "lexical"}]}
    ],
    "unresolved": ["nonsense"],
    "associations": [
        {
            "id": "term:kindness",
            "score": 0.79,
            "novelty": 0.4,
            "distance": 1,
            "source_count": 2,
            "bridge": True,
            "labels": ["dbslm_term"],
            "references": [
                {"id": "ref_1", "text": "Kindness was the whole answer.", "source": "chunk1"}
            ],
            "sources": [{"seed": "term:grateful", "activation": 0.55, "hops": 1}],
            "via": [
                {
                    "from": "term:grateful",
                    "to": "term:kindness",
                    "type": "precedes",
                    "weight": 0.9,
                    "confidence": 1.0,
                    "modality": "certain",
                }
            ],
        },
        {
            "id": "ctx:emotion:joy",
            "score": 0.4,
            "novelty": 0.2,
            "distance": 2,
            "source_count": 1,
            "sources": [{"seed": "term:grateful", "activation": 0.4, "hops": 2}],
            "via": [],
        },
    ],
}

_RECALL_RESPONSE = (
    "SUCCESS,command=GRAPH_RECALL,seeds=2,resolved=1,visited=7,expanded=10,hydrated=28,"
    "references=1,count=2,bridges=1,truncated=1,precision=0.200,payload="
    + _encode_payload(_RECALL_PAYLOAD)
)


class GraphIdentityTests(unittest.TestCase):
    def test_ids_are_stable_single_protocol_tokens(self) -> None:
        self.assertEqual(slugify("Très  Étrange!"), "tres_etrange")
        self.assertEqual(context_node_id("Emotion", "Joy "), "ctx:emotion:joy")
        self.assertEqual(term_node_id("Gratitude."), "term:gratitude")
        for node_id in (context_node_id("emotion", "very happy"), term_node_id("thank you")):
            self.assertNotIn(" ", node_id)

    def test_term_text_round_trips_only_for_term_nodes(self) -> None:
        self.assertEqual(term_text_from_node_id("term:deep_gratitude"), "deep gratitude")
        self.assertEqual(term_text_from_node_id("ctx:emotion:joy"), "")

    def test_content_terms_drop_structural_tags_and_function_words(self) -> None:
        terms = content_terms("|USER|: I am very grateful for the help |RESPONSE|:")
        self.assertNotIn("user", terms)
        self.assertNotIn("response", terms)
        self.assertNotIn("for", terms)
        self.assertIn("grateful", terms)


class CheetahGraphProtocolTests(unittest.TestCase):
    def test_node_set_base64_encodes_references_and_keeps_ids_tokenized(self) -> None:
        client = _ScriptedSocketClient()

        written = client.graph_node_set(
            "ctx:emotion:joy",
            labels=["dbslm_context"],
            references=[{"text": "A sentence with spaces.", "source": "chunk1"}],
        )

        self.assertTrue(written)
        args = client.command_args("GRAPH_NODE_SET")
        self.assertEqual(args["id"], "ctx:emotion:joy")
        self.assertEqual(args["labels"], "dbslm_context")
        decoded = json.loads(base64.b64decode(args["references"]).decode("utf-8"))
        self.assertEqual(decoded[0]["text"], "A sentence with spaces.")
        # One command, one line: no raw whitespace may leak into the argument.
        self.assertEqual(len(client.commands[0].split("\n")), 1)

    def test_node_set_rejects_an_id_that_is_not_a_single_token(self) -> None:
        client = _ScriptedSocketClient()

        self.assertFalse(client.graph_node_set("cat sitter"))
        self.assertEqual(client.commands, [])

    def test_node_set_clears_references_with_the_dash_token(self) -> None:
        client = _ScriptedSocketClient()

        client.graph_node_set("term:stale", clear_references=True)

        self.assertEqual(client.command_args("GRAPH_NODE_SET")["references"], "-")

    def test_recall_seeds_with_spaces_use_the_base64_form(self) -> None:
        client = _ScriptedSocketClient({"GRAPH_RECALL": _RECALL_RESPONSE})

        client.graph_recall(["deep gratitude", "term:help"])

        seeds = client.command_args("GRAPH_RECALL")["seeds"]
        self.assertTrue(seeds.startswith("base64:"))
        decoded = base64.b64decode(seeds[len("base64:") :]).decode("utf-8")
        self.assertEqual(decoded, "deep gratitude,term:help")

    def test_recall_clamps_bounds_to_the_server_maximums(self) -> None:
        client = _ScriptedSocketClient({"GRAPH_RECALL": _RECALL_RESPONSE})

        client.graph_recall(
            ["term:help"],
            hops=99,
            branch_limit=99999,
            budget=999999999,
            references=True,
            reference_limit=9999,
        )

        args = client.command_args("GRAPH_RECALL")
        self.assertEqual(args["hops"], "6")
        self.assertEqual(args["branch_limit"], "1024")
        self.assertEqual(args["budget"], "262144")
        self.assertEqual(args["reference_limit"], "256")

    def test_recall_parses_header_counters_and_payload(self) -> None:
        client = _ScriptedSocketClient({"GRAPH_RECALL": _RECALL_RESPONSE})

        result = client.graph_recall(["grateful"], references=True)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.count, 2)
        self.assertEqual(result.bridges, 1)
        self.assertTrue(result.truncated)
        self.assertEqual(result.precision, 0.2)
        self.assertEqual(result.unresolved, ("nonsense",))
        self.assertEqual(result.seed_resolutions[0].matches[0].node_id, "term:grateful")
        first = result.associations[0]
        self.assertEqual(first.node_id, "term:kindness")
        self.assertTrue(first.bridge)
        self.assertEqual(first.references[0].text, "Kindness was the whole answer.")
        self.assertEqual(first.via[0].edge_type, "precedes")
        self.assertEqual(first.via[0].modality, "certain")

    def test_edge_set_batch_reports_created_and_updated_counters(self) -> None:
        client = _ScriptedSocketClient(
            {
                "GRAPH_EDGE_SET_BATCH": (
                    "SUCCESS,requested=3,applied=2,created=1,updated=1,failed=1,payload="
                    + _encode_payload([{"index": 2, "error": "invalid_confidence"}])
                )
            }
        )

        batch = client.graph_edge_set_batch(
            [{"from": "a", "to": "b", "type": "precedes"}],
            continue_on_error=True,
        )

        assert batch is not None
        self.assertEqual((batch.requested, batch.applied, batch.created), (3, 2, 1))
        self.assertEqual((batch.updated, batch.failed), (1, 1))
        args = client.command_args("GRAPH_EDGE_SET_BATCH")
        self.assertEqual(args["continue_on_error"], "1")
        decoded = json.loads(base64.b64decode(args["items"]).decode("utf-8"))
        self.assertEqual(decoded[0]["from"], "a")

    def test_error_responses_answer_none_rather_than_raising(self) -> None:
        client = _ScriptedSocketClient({"GRAPH_RECALL": "ERROR,graph_recall_requires_seeds"})

        self.assertIsNone(client.graph_recall(["term:help"]))

    def test_term_index_stats_parse_enabled_and_entries(self) -> None:
        client = _ScriptedSocketClient(
            {"GRAPH_TERM_INDEX": "SUCCESS,command=GRAPH_TERM_INDEX,action=stats,enabled=1,entries=21"}
        )

        stats = client.graph_term_index("stats")

        assert stats is not None
        self.assertTrue(stats.enabled)
        self.assertEqual(stats.entries, 21)

    def test_disabled_adapter_answers_without_touching_the_socket(self) -> None:
        client = _ScriptedSocketClient()
        adapter = CheetahHotPathAdapter(client=client)
        adapter._enabled = False

        self.assertIsNone(adapter.graph_recall(["term:help"]))
        self.assertFalse(adapter.graph_node_set("term:help"))
        self.assertEqual(client.commands, [])


class _RecordingHotPath(NullHotPathAdapter):
    """Captures graph calls so ingest/recall behaviour can be asserted offline."""

    def __init__(self, recall_result=None, stored_references=None) -> None:
        self.nodes: list[dict[str, object]] = []
        self.batches: list[list[dict[str, object]]] = []
        self.recalls: list[dict[str, object]] = []
        self._recall_result = recall_result
        self._stored_references = stored_references or {}

    def graph_node_set(self, node_id, *, labels=None, props=None, references=None, clear_references=False):
        self.nodes.append(
            {
                "id": node_id,
                "labels": tuple(labels or ()),
                "references": [dict(ref) for ref in references or ()],
            }
        )
        return True

    def graph_node_get(self, node_id):
        return self._stored_references.get(node_id)

    def graph_edge_set_batch(self, items, *, continue_on_error=True, default_type=None, default_props=None):
        self.batches.append([dict(item) for item in items])
        from db_slm.cheetah_types import GraphEdgeBatchResult

        return GraphEdgeBatchResult(
            requested=len(items),
            applied=len(items),
            created=len(items),
            updated=0,
            failed=0,
        )

    def graph_recall(self, seeds, **kwargs):
        self.recalls.append({"seeds": list(seeds), **kwargs})
        return self._recall_result


class GraphContextMemoryIngestTests(unittest.TestCase):
    @staticmethod
    def _record() -> _FakeRecord:
        return _FakeRecord(
            prompt="|USER|: Why do I feel grateful?\n|RESPONSE|:",
            response="Because kindness was offered freely.",
            context_tokens={"emotion": "joy"},
            response_dependencies=_FakeLayer(
                {"entities": ("kindness",), "actions": ("offer",)},
                arcs=(_FakeArc("kindness", "offer", "nsubj"),),
            ),
        )

    def test_observation_writes_context_and_term_nodes_with_typed_edges(self) -> None:
        hot_path = _RecordingHotPath()
        memory = GraphContextMemory(hot_path)

        stats = memory.observe_records([self._record()], source_label="chunk1")

        self.assertEqual(stats.records, 1)
        node_ids = {entry["id"] for entry in hot_path.nodes}
        self.assertIn("ctx:emotion:joy", node_ids)
        self.assertIn("term:kindness", node_ids)
        edge_types = {item["type"] for item in hot_path.batches[0]}
        self.assertIn("evokes", edge_types)
        self.assertIn("precedes", edge_types)
        self.assertIn("dep_nsubj", edge_types)
        for item in hot_path.batches[0]:
            self.assertLessEqual(item["weight"], 1.0)
            self.assertGreater(item["weight"], 0.0)

    def test_reference_sentences_land_on_the_context_node(self) -> None:
        hot_path = _RecordingHotPath()
        memory = GraphContextMemory(hot_path)

        stats = memory.observe_records([self._record()], source_label="chunk1")

        context_writes = [entry for entry in hot_path.nodes if entry["id"] == "ctx:emotion:joy"]
        self.assertEqual(len(context_writes), 1)
        references = context_writes[0]["references"]
        self.assertEqual(references[0]["text"], "Because kindness was offered freely.")
        self.assertEqual(references[0]["source"], "chunk1")
        self.assertEqual(stats.references_attached, 1)

    def test_stored_references_are_merged_instead_of_overwritten(self) -> None:
        from db_slm.cheetah_types import GraphNodeRecord, GraphReferenceSentence

        stored = GraphNodeRecord(
            node_id="ctx:emotion:joy",
            references=(GraphReferenceSentence(reference_id="ref_old", text="An earlier run."),),
        )
        hot_path = _RecordingHotPath(stored_references={"ctx:emotion:joy": stored})
        memory = GraphContextMemory(hot_path)

        memory.observe_records([self._record()], source_label="chunk2")

        write = next(entry for entry in hot_path.nodes if entry["id"] == "ctx:emotion:joy")
        texts = [reference["text"] for reference in write["references"]]
        self.assertIn("An earlier run.", texts)
        self.assertIn("Because kindness was offered freely.", texts)

    def test_repeated_records_do_not_duplicate_a_reference_or_rewrite_nodes(self) -> None:
        hot_path = _RecordingHotPath()
        memory = GraphContextMemory(hot_path)

        memory.observe_records([self._record(), self._record()], source_label="chunk1")

        context_writes = [entry for entry in hot_path.nodes if entry["id"] == "ctx:emotion:joy"]
        self.assertEqual(len(context_writes), 1)
        self.assertEqual(len(context_writes[0]["references"]), 1)
        # A node that gains nothing from the second record is not rewritten, but
        # its relations are still upserted so edge weights stay current.
        self.assertEqual(len(hot_path.nodes), len(set(e["id"] for e in hot_path.nodes)))
        self.assertEqual(len(hot_path.batches), 2)

    def test_an_over_budget_response_is_cut_at_a_sentence_boundary(self) -> None:
        hot_path = _RecordingHotPath()
        memory = GraphContextMemory(hot_path, reference_chars=64)
        record = _FakeRecord(
            "|USER|: tell me\n|RESPONSE|:",
            "Kindness arrived first. Then patience followed slowly. "
            "And finally the long quiet week ended.",
            {"emotion": "joy"},
        )

        memory.observe_records([record])

        write = next(e for e in hot_path.nodes if e["id"] == "ctx:emotion:joy")
        text = write["references"][0]["text"]
        self.assertTrue(text.endswith("."))
        self.assertLessEqual(len(text), 64)
        self.assertEqual(text, "Kindness arrived first. Then patience followed slowly.")

    def test_a_response_with_no_sentence_break_in_budget_stores_no_reference(self) -> None:
        hot_path = _RecordingHotPath()
        memory = GraphContextMemory(hot_path, reference_chars=64)
        record = _FakeRecord(
            "|USER|: tell me\n|RESPONSE|:",
            "kindness patience gratitude resilience curiosity empathy humility "
            "generosity attention devotion.",
            {"emotion": "joy"},
        )

        stats = memory.observe_records([record])

        self.assertEqual(stats.references_attached, 0)
        write = next(e for e in hot_path.nodes if e["id"] == "ctx:emotion:joy")
        self.assertEqual(write["references"], [])

    def test_record_without_a_response_is_skipped(self) -> None:
        hot_path = _RecordingHotPath()
        memory = GraphContextMemory(hot_path)

        stats = memory.observe_records([_FakeRecord("|USER|: hello", "")])

        self.assertEqual(stats.records, 0)
        self.assertEqual(stats.skipped_records, 1)
        self.assertEqual(hot_path.nodes, [])

    def test_terms_per_record_stay_within_the_configured_cap(self) -> None:
        hot_path = _RecordingHotPath()
        memory = GraphContextMemory(hot_path, max_terms_per_side=2, max_dependency_arcs=0)
        record = _FakeRecord(
            prompt="alpha bravo charlie delta echo foxtrot",
            response="golf hotel india juliet kilo lima",
        )

        memory.observe_records([record])

        term_nodes = [entry["id"] for entry in hot_path.nodes if entry["id"].startswith("term:")]
        self.assertEqual(len(term_nodes), 4)

    def test_a_null_adapter_makes_graph_memory_unavailable(self) -> None:
        memory = GraphContextMemory(None)
        self.assertFalse(memory.available())
        self.assertEqual(memory.observe_records([self._record()]).records, 0)
        self.assertIsNone(memory.recall("anything"))


class GraphContextMemoryRecallTests(unittest.TestCase):
    @staticmethod
    def _result():
        return CheetahClient._parse_graph_recall_response(_RECALL_RESPONSE)

    def test_seeds_put_declared_context_values_before_prompt_words(self) -> None:
        memory = GraphContextMemory(_RecordingHotPath())

        seeds = memory.build_seeds(
            "|USER|: Why am I grateful?", context_tokens={"emotion": "joy"}
        )

        self.assertEqual(seeds[0], "ctx:emotion:joy")
        self.assertIn("grateful", seeds)
        self.assertNotIn("user", seeds)

    def test_recall_projects_terms_and_reference_sentences(self) -> None:
        hot_path = _RecordingHotPath(recall_result=self._result())
        memory = GraphContextMemory(hot_path, reference_limit=4)

        signal = memory.recall("I feel grateful", context_tokens={"emotion": "joy"})

        assert signal is not None
        self.assertEqual(signal.term_weights, {"kindness": 0.79})
        self.assertEqual(signal.context_text, "Kindness was the whole answer.")
        self.assertTrue(signal.truncated)
        self.assertEqual(signal.unresolved, ("nonsense",))
        self.assertEqual(hot_path.recalls[0]["hops"], memory.recall_hops)
        self.assertTrue(hot_path.recalls[0]["references"])

    def test_recall_asks_for_seed_nodes_so_their_sentences_hydrate(self) -> None:
        hot_path = _RecordingHotPath(recall_result=self._result())
        memory = GraphContextMemory(hot_path, recall_references=True)

        memory.recall("I feel grateful")

        self.assertTrue(hot_path.recalls[0]["include_seeds"])

    def test_recall_without_references_does_not_ask_for_seed_nodes(self) -> None:
        hot_path = _RecordingHotPath(recall_result=self._result())
        memory = GraphContextMemory(hot_path, recall_references=False)

        memory.recall("I feel grateful")

        self.assertFalse(hot_path.recalls[0]["include_seeds"])
        self.assertFalse(hot_path.recalls[0]["references"])

    def test_seed_terms_contribute_sentences_but_never_bias(self) -> None:
        payload = {
            "seeds": [
                {
                    "term": "grateful",
                    "matches": [{"id": "term:grateful", "score": 0.9, "match": "exact"}],
                }
            ],
            "associations": [
                {
                    "id": "term:grateful",
                    "score": 0.99,
                    "novelty": 0.0,
                    "distance": 0,
                    "source_count": 1,
                    "references": [{"id": "r1", "text": "Gratitude was recorded here."}],
                    "sources": [],
                    "via": [],
                },
                {
                    "id": "term:kindness",
                    "score": 0.6,
                    "novelty": 0.3,
                    "distance": 1,
                    "source_count": 1,
                    "sources": [],
                    "via": [],
                },
            ],
        }
        response = (
            "SUCCESS,command=GRAPH_RECALL,seeds=1,resolved=1,visited=2,expanded=1,hydrated=2,"
            "references=1,count=2,bridges=0,truncated=0,precision=0.200,payload="
            + _encode_payload(payload)
        )
        result = CheetahClient._parse_graph_recall_response(response)
        memory = GraphContextMemory(_RecordingHotPath())

        signal = memory.signal_from_result(result, seeds=["grateful"])

        self.assertEqual(signal.term_weights, {"kindness": 0.6})
        self.assertEqual(signal.context_text, "Gratitude was recorded here.")

    def test_seed_count_never_exceeds_the_configured_maximum(self) -> None:
        memory = GraphContextMemory(_RecordingHotPath(), max_seeds=3)

        seeds = memory.build_seeds("alpha bravo charlie delta echo foxtrot golf")

        self.assertEqual(len(seeds), 3)


class _BiasVocabulary:
    """Only `kindness` is in the vocabulary; `unseen` never was trained."""

    _known = {"kindness": 7, "|response|:": 3}

    def lookup(self, token_text: str) -> int | None:
        return self._known.get(token_text)

    def token_id(self, token_text: str) -> int:
        raise AssertionError("recall must not mint vocabulary")


class _BiasTokenizer:
    def tokenize(self, text: str, add_special_tokens: bool = True, **_: object):
        return text.split(), None


class GraphBiasProjectionTests(unittest.TestCase):
    """`DBSLMEngine._graph_token_bias` on a lightweight engine fixture."""

    @staticmethod
    def _engine(weight: float):
        engine = DBSLMEngine.__new__(DBSLMEngine)
        engine.vocab = _BiasVocabulary()
        engine.tokenizer = _BiasTokenizer()
        engine.graph_bias_weight = weight
        engine._prompt_tag_token_ids = {3}
        return engine

    @staticmethod
    def _signal(term_weights):
        return GraphContextSignal(seeds=("x",), term_weights=term_weights)

    def test_bias_stays_at_or_below_a_user_correction(self) -> None:
        engine = self._engine(1.0)

        bias = DBSLMEngine._graph_token_bias(engine, self._signal({"kindness": 1.0}))

        self.assertEqual(bias, {7: _CORRECTION_Q_BIAS})

    def test_default_weight_is_a_nudge_not_an_override(self) -> None:
        engine = self._engine(0.25)

        bias = DBSLMEngine._graph_token_bias(engine, self._signal({"kindness": 0.8}))

        assert bias is not None
        self.assertLess(bias[7], _CORRECTION_Q_BIAS)
        self.assertGreater(bias[7], 0)

    def test_an_unknown_term_is_dropped_rather_than_added_to_the_vocabulary(self) -> None:
        engine = self._engine(1.0)

        bias = DBSLMEngine._graph_token_bias(engine, self._signal({"unseen": 1.0}))

        self.assertIsNone(bias)

    def test_a_recalled_prompt_tag_is_never_biased(self) -> None:
        engine = self._engine(1.0)

        bias = DBSLMEngine._graph_token_bias(engine, self._signal({"|response|:": 1.0}))

        self.assertIsNone(bias)

    def test_zero_weight_disables_the_bias_but_not_the_signal(self) -> None:
        engine = self._engine(0.0)

        self.assertIsNone(
            DBSLMEngine._graph_token_bias(engine, self._signal({"kindness": 1.0}))
        )


if __name__ == "__main__":
    unittest.main()
