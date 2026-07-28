import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from eval import CommunicationEvaluator
from models import CVCommunicator
from scripts.analyze_receiver_conditioning import VARIANTS, analyze, discover


class ReceiverConditioningControlTests(unittest.TestCase):
    def test_derangement_is_deterministic_and_has_no_fixed_points(self):
        first = CommunicationEvaluator._make_derangement(500, 42)
        second = CommunicationEvaluator._make_derangement(500, 42)
        self.assertEqual(first, second)
        self.assertEqual(sorted(first), list(range(500)))
        self.assertTrue(all(index != target for index, target in enumerate(first)))

    def test_replay_budget_matches_realized_total(self):
        communicator = SimpleNamespace(
            merge_sink=4,
            merge_recent=8,
            replay_target_budget=None,
            replay_layer_budget={},
        )
        lengths = [174] * 32
        cache = SimpleNamespace(
            key_cache=[
                SimpleNamespace(shape=(1, 8, length, 128))
                for length in lengths
            ]
        )
        target = 0.275448
        CVCommunicator.set_replay_budget(communicator, target, cache)
        kept = lengths[0] + sum(
            round(communicator.replay_layer_budget[layer] * lengths[layer])
            for layer in range(1, len(lengths))
        )
        actual = kept / sum(lengths)
        self.assertLess(abs(actual - target), 1e-3)

    def test_shuffled_budget_preserves_multiset_and_breaks_alignment(self):
        replay = {
            index: {
                "budget": 0.1 + index * 0.01,
                "id": f"id-{index}",
                "source_idx": index,
            }
            for index in range(20)
        }
        shuffled = CommunicationEvaluator._shuffle_budget_replay(
            replay, seed=42
        )
        self.assertEqual(
            sorted(row["budget"] for row in replay.values()),
            sorted(row["budget"] for row in shuffled.values()),
        )
        self.assertTrue(
            all(index != row["source_idx"] for index, row in shuffled.items())
        )
        self.assertTrue(
            all(row["id"] == f"id-{index}" for index, row in shuffled.items())
        )

    def test_paired_analyzer_uses_common_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for variant in VARIANTS:
                run = root / "pair1" / "hotpotqa" / variant / f"rc_{variant}_0000"
                run.mkdir(parents=True)
                score = 1.0 if variant == "correct" else 0.0
                with (run / "per_sample.jsonl").open("w") as handle:
                    handle.write(
                        json.dumps(
                            {"_meta": {"query_condition_mode": variant}}
                        )
                        + "\n"
                    )
                    for index in range(4):
                        handle.write(
                            json.dumps(
                                {
                                    "idx": index,
                                    "id": f"id-{index}",
                                    "score": score,
                                    "budget": 0.3,
                                    "replay_target_budget": 0.3,
                                    "query_sketch_bytes": (
                                        0 if variant in {"query_free", "sender_context_q"} else 16
                                    ),
                                }
                            )
                            + "\n"
                        )
            paths = discover(root)
            cells, macro = analyze(paths, 200, 42, 1e-3)
            self.assertEqual(len(paths), 6)
            self.assertEqual(len(cells), 5)
            self.assertEqual(len(macro), 5)
            self.assertTrue(all(row["mean_delta"] == 1.0 for row in macro))


if __name__ == "__main__":
    unittest.main()
