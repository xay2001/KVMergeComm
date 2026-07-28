import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.analyze_brekv_heldout_and_shuffled import (
    BYTE_FIELDS,
    DEV_CELLS,
    HELDOUT_CELLS,
    PAIRS,
    SELECTION_RATIOS,
    TASKS,
    build_oracle,
    build_policies,
    build_shuffled,
    choose_dev_ratio,
    per_task_best,
)


def rows(score, budgets=(0.25, 0.35)):
    output = {}
    for index, budget in enumerate(budgets):
        output[f"{index}::id-{index}"] = {
            "score": float(score),
            "budget": float(budget),
            BYTE_FIELDS[0]: 300.0,
            BYTE_FIELDS[1]: 200.0,
            BYTE_FIELDS[2]: 100.0,
        }
    return output


def write_rows(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for index, (key, row) in enumerate(data.items()):
            handle.write(json.dumps({"idx": index, "id": key.split("::", 1)[1], **row}) + "\n")


class BrekvHeldoutAnalysisTests(unittest.TestCase):
    def make_cells(self):
        cells = {}
        for pair in PAIRS:
            for task in TASKS:
                fixed = {}
                for ratio in SELECTION_RATIOS:
                    score = 1.0 if ratio >= 0.4 else 0.0
                    fixed[ratio] = rows(score)
                # Development alone favors .3; held-out tasks favor .4.
                if (pair, task) in DEV_CELLS:
                    fixed[0.3] = rows(2.0)
                cells[(pair, task)] = {"fixed": fixed, "brekv": rows(0.75)}
        return cells

    def test_development_selection_and_heldout_are_separate(self):
        cells = self.make_cells()
        self.assertEqual(choose_dev_ratio(cells), 0.3)
        self.assertTrue(all(cell not in DEV_CELLS for cell in HELDOUT_CELLS))
        self.assertEqual(len(HELDOUT_CELLS), 54)
        best = per_task_best(cells)
        self.assertTrue(all(best[task] == 0.4 for task in TASKS))
        _, summary = build_policies(
            cells, 0.3, best, 100, np.random.default_rng(7)
        )
        self.assertTrue(all(row["n_cells"] == 54 for row in summary))
        self.assertEqual(
            next(row for row in summary if row["policy"] == "B-ReKV")["total_bytes"],
            300.0,
        )

    def test_exact_oracle_and_shuffled_budget_multiset(self):
        cells = self.make_cells()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            r07 = {}
            shuffled = {}
            for pair in PAIRS:
                for task in TASKS:
                    r07_path = root / "r07" / pair / task / "per_sample.jsonl"
                    write_rows(r07_path, rows(1.0))
                    r07[(pair, task)] = r07_path
                    if (pair, task) in HELDOUT_CELLS:
                        shuffled_path = root / "shuffled" / pair / task / "per_sample.jsonl"
                        write_rows(shuffled_path, rows(0.25, budgets=(0.35, 0.25)))
                        shuffled[(pair, task)] = shuffled_path
            oracle_rows, oracle_summary, bins = build_oracle(cells, r07, 1.0)
            self.assertEqual(len(oracle_rows), 108)
            self.assertTrue(all(row["oracle_ratio"] == 0.4 for row in oracle_rows))
            self.assertTrue(bins)
            detail, summary = build_shuffled(
                cells, shuffled, 100, np.random.default_rng(9), 1e-9
            )
            self.assertEqual(len(detail), 54)
            self.assertTrue(summary["budget_multisets_identical"])
            self.assertGreater(summary["score_delta"], 0)
            self.assertEqual(
                next(row for row in oracle_summary if row["scope"] == "macro")["n"],
                108,
            )


if __name__ == "__main__":
    unittest.main()
