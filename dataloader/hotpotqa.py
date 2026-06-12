from .base_evaluator import BaseEvaluator
from datasets import load_dataset
from pathlib import Path

def construct_support(item):
    supporting_facts = item["supporting_facts"]
    context = item["context"]
    results = []
    for title, sent_id in zip(supporting_facts["title"], supporting_facts["sent_id"]):
        for context_title, sentences in zip(context["title"], context["sentences"]):
            if context_title == title:
                try:
                    results.append(sentences[sent_id])
                except:
                    print(f"Warning: {item['id']} has a supporting fact that is not in the context")
    return results

class HotpotQAEvaluator(BaseEvaluator):
    def __init__(self, multi_agent=False, n_samples=500):
        super().__init__()
        self.max_tokens = 48
        self.truncate_input = True
        self.multiple_answers = False
        self.n_samples = n_samples
        if multi_agent:
            self.data = self.load_data_multi_sender()
        else:
            self.data = self.load_data_single_sender()
        self.name = "hotpotqa"
        self.multi_agent = multi_agent
    
    def load_hotpotqa_dataset(self):
        project_root = Path(__file__).resolve().parents[1]
        local_dataset_path = project_root / "datasets" / "hotpot_qa"
        dataset_path = str(local_dataset_path) if local_dataset_path.exists() else "hotpotqa/hotpot_qa"
        return load_dataset(dataset_path, "distractor")["validation"]
        
    def load_data_single_sender(self):
        dataset = self.load_hotpotqa_dataset()
        dataset = self.random_sample(dataset)
        dataset = dataset.map(lambda x: {"support": construct_support(x)})
        dataset = dataset.map(lambda x: {"prompt_A": "\n".join(x["support"])})
        dataset = dataset.map(lambda x: {"prompt_B": x["question"]})
        return dataset

    def load_data_multi_sender(self):
        dataset = self.load_hotpotqa_dataset()
        dataset = dataset.shuffle(seed=self.random_state)
        dataset = dataset.map(lambda x: {"support": construct_support(x)})
        # keep only those with at least 2 supporting facts
        dataset = dataset.filter(lambda x: len(x["support"]) >= 2)
        if self.n_samples is not None:
            dataset = dataset.select(range(self.n_samples))
        dataset = dataset.map(lambda x: {"prompt_A1": "\n".join(x["support"][:len(x["support"])//2])})
        dataset = dataset.map(lambda x: {"prompt_A2": "\n".join(x["support"][len(x["support"])//2:])})
        dataset = dataset.map(lambda x: {"prompt_B": x["question"]})
        return dataset
