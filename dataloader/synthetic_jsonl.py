import json
import os

from .base_evaluator import BaseEvaluator


class SyntheticJSONLEvaluator(BaseEvaluator):
    """QA evaluator over a pre-built JSONL file (controlled-context experiments).

    Each line must contain: prompt_A (context), prompt_B (question), answer, id.
    Path comes from the SYNTH_JSONL_PATH environment variable so the standard
    com.py entrypoint can run controlled context-length / evidence-sparsity
    datasets without a new CLI flag.
    """

    def __init__(self, path=None):
        super().__init__()
        path = path or os.environ.get("SYNTH_JSONL_PATH", "")
        if not path or not os.path.isfile(path):
            raise ValueError(
                f"synthetic_jsonl task requires SYNTH_JSONL_PATH to point to a JSONL file, got: {path!r}"
            )
        self.max_tokens = 48
        self.truncate_input = True
        self.multiple_answers = False
        self.n_samples = None
        data = []
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        if not data:
            raise ValueError(f"synthetic_jsonl file is empty: {path}")
        self.data = data
        self.name = "synthetic_jsonl"
        self.source_path = path
