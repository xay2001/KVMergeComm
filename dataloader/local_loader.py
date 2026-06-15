from pathlib import Path

DATASETS_DIR = Path(__file__).resolve().parents[1] / "datasets"


def local_or_hub(local_name: str, hub_id: str) -> str:
    """Return a local dataset path under datasets/<local_name> if it exists,
    otherwise fall back to the HuggingFace hub id. Lets every loader prefer the
    locally downloaded copy (see scripts/download_datasets.py)."""
    p = DATASETS_DIR / local_name
    return str(p) if p.exists() else hub_id
