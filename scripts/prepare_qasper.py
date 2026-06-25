#!/usr/bin/env python3
"""Prepare SCROLLS/Qasper as local JSONL files for modern datasets versions."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


QASPER_URL = "https://huggingface.co/datasets/tau/scrolls/resolve/main/qasper.zip"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "scrolls" / "qasper"
SPLITS = ("train", "validation", "test")


def download(url: str, destination: Path) -> None:
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, destination.open("wb") as out_file:
        shutil.copyfileobj(response, out_file)


def extract_qasper(zip_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        for split in SPLITS:
            member = f"qasper/{split}.jsonl"
            if member not in names:
                raise FileNotFoundError(f"{member} not found in {zip_path}")

            target = output_dir / f"{split}.jsonl"
            with archive.open(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            print(f"Wrote {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=QASPER_URL, help="Qasper zip URL")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where train/validation/test JSONL files are written",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "qasper.zip"
        download(args.url, zip_path)
        extract_qasper(zip_path, args.output_dir)


if __name__ == "__main__":
    main()
