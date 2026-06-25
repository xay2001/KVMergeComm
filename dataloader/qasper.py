from .base_evaluator import BaseEvaluator
from datasets import load_dataset
from pathlib import Path


QASPER_DATA_DIR = Path(__file__).resolve().parents[1] / "datasets" / "scrolls" / "qasper"


class QaSperEvaluator(BaseEvaluator):
    def __init__(self, n_samples=500):
        super().__init__()
        self.max_tokens = 128
        self.truncate_input = True
        self.multiple_answers = False
        self.n_samples = n_samples
        self.data = self.load_data()
        self.name = "qasper"
        
    def load_data(self):
        validation_file = QASPER_DATA_DIR / "validation.jsonl"
        if not validation_file.exists():
            raise FileNotFoundError(
                f"Missing {validation_file}. Run `python scripts/prepare_qasper.py` first."
            )

        dataset = load_dataset("json", data_files={"validation": str(validation_file)})["validation"]
        dataset = self.random_sample(dataset)
        dataset = dataset.map(lambda x: {
            "prompt_A": x["input"][x["input"].index("\n\n")+2:].strip(), 
            "prompt_B": x["input"][:x["input"].index("\n\n")].strip(), 
            "answer": x["output"],
        })
        return dataset
