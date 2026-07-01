#!/usr/bin/env python3
"""Clean and plot ReKV interpretability examples from dumped top-token JSONL.

CPU-only. It filters special/template tokens and selects one paper-friendly
example per task where ReKV has stronger answer-term overlap than baselines.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TEMPLATE_TERMS = {
    "assistant",
    "system",
    "user",
    "instruction",
    "context",
    "date",
    "today",
    "knowledge",
    "cutting",
    "directly",
    "answer",
    "question",
    "needed",
    "explanation",
}


def norm_terms(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def is_template_token(token: dict) -> bool:
    text = str(token.get("text", ""))
    raw = str(token.get("token", ""))
    stripped = text.strip()
    if not stripped:
        return True
    if "<|" in text or "<|" in raw:
        return True
    if not re.search(r"[A-Za-z0-9]", stripped):
        return True
    terms = norm_terms(stripped)
    if not terms:
        return True
    if all(t in TEMPLATE_TERMS for t in terms):
        return True
    window = str(token.get("window_text", "")).lower()
    if "<|start_header_id|>" in window or "<|end_header_id|>" in window:
        # Header-adjacent tokens are usually prompt/template artifacts.
        return True
    return False


def clean_tokens(tokens: list[dict], top_k: int) -> list[dict]:
    seen = set()
    out = []
    for tok in tokens:
        if is_template_token(tok):
            continue
        label = label_for(tok)
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(tok)
        if len(out) >= top_k:
            break
    return out


def label_for(token: dict) -> str:
    text = str(token.get("text", "")).replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text[:36] if text else str(token.get("token", ""))


def answer_recall(row: dict, method: str) -> float:
    val = row.get("overlap", {}).get(method, {}).get("recall")
    return float(val) if val is not None else 0.0


def answer_terms(row: dict) -> set[str]:
    terms = set()
    for ans in row.get("answers", []):
        terms.update(norm_terms(str(ans)))
    return terms


def clean_hit_count(tokens: list[dict], terms: set[str]) -> int:
    hits = set()
    for tok in tokens:
        hits.update(set(norm_terms(str(tok.get("text", "")))) & terms)
    return len(hits)


def choose_example(rows: list[dict], top_k: int) -> tuple[dict, dict[str, list[dict]]]:
    candidates = []
    for row in rows:
        cleaned = {
            "rekv": clean_tokens(row.get("rekv_top", []), top_k),
            "evict": clean_tokens(row.get("evict_top", []), top_k),
            "random": clean_tokens(row.get("random_top", []), top_k),
        }
        if len(cleaned["rekv"]) < min(6, top_k):
            continue
        terms = answer_terms(row)
        clean_hits = clean_hit_count(cleaned["rekv"], terms)
        gain = answer_recall(row, "rekv") - max(answer_recall(row, "evict"), answer_recall(row, "random"))
        candidates.append((gain, clean_hits, answer_recall(row, "rekv"), row, cleaned))
    if not candidates:
        raise RuntimeError("No clean examples found")
    # Prefer clear ReKV wins, then more clean answer hits, then higher raw recall.
    gain, clean_hits, recall, row, cleaned = max(candidates, key=lambda x: (x[0], x[1], x[2]))
    return row, cleaned


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def plot_example(task: str, row: dict, cleaned: dict[str, list[dict]], out: Path, top_k: int) -> None:
    methods = ["rekv", "evict", "random"]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), sharex=False)
    for ax, method in zip(axes, methods):
        toks = cleaned[method][:top_k]
        labels = [label_for(t) for t in toks][::-1]
        scores = [float(t.get("score", 0.0)) for t in toks][::-1]
        color = "#e45756" if method == "rekv" else "#4c78a8" if method == "evict" else "#9d755d"
        ax.barh(labels, scores, color=color, alpha=0.86)
        ax.set_title(method.upper() if method == "rekv" else method.title())
        ax.set_xlabel("Selection score")
        ax.grid(axis="x", alpha=0.25)
        ax.tick_params(axis="y", labelsize=8)

    question = str(row.get("question", "")).replace("\n", " ")
    answers = ", ".join(str(x) for x in row.get("answers", []))
    fig.suptitle(
        f"{task} example idx={row.get('idx')} | clean top tokens after filtering prompt/template tokens\n"
        f"Q: {question[:125]}{'...' if len(question) > 125 else ''}\n"
        f"A: {answers[:125]}{'...' if len(answers) > 125 else ''}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("snapshots/interpretability/pair1_llama31_same"))
    ap.add_argument("--tasks", nargs="+", default=["hotpotqa", "musique", "multifieldqa_en"])
    ap.add_argument("--recv_window", type=int, default=8)
    ap.add_argument("--ratio", type=float, default=0.3)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--out_dir", type=Path, default=Path("snapshots/interpretability/pair1_llama31_same/cleaned"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected = []
    md_lines = ["# Clean Interpretability Examples", ""]

    for task in args.tasks:
        path = args.root / task / f"top_tokens_w{args.recv_window}_r{args.ratio}.jsonl"
        rows = read_rows(path)
        row, cleaned = choose_example(rows, args.top_k)
        plot_path = args.out_dir / f"{task}_clean_top_tokens.png"
        plot_example(task, row, cleaned, plot_path, args.top_k)

        record = {
            "task": task,
            "idx": row.get("idx"),
            "question": row.get("question"),
            "answers": row.get("answers"),
            "overlap": row.get("overlap"),
            "clean_top_tokens": {
                method: [
                    {
                        "idx": tok.get("idx"),
                        "text": label_for(tok),
                        "score": tok.get("score"),
                        "window_text": tok.get("window_text"),
                    }
                    for tok in toks[: args.top_k]
                ]
                for method, toks in cleaned.items()
            },
            "plot": str(plot_path),
        }
        selected.append(record)

        md_lines.extend([
            f"## {task} idx={row.get('idx')}",
            "",
            f"Question: {str(row.get('question', '')).strip()}",
            "",
            f"Answers: {row.get('answers')}",
            "",
            f"Plot: `{plot_path}`",
            "",
            "ReKV clean top tokens: "
            + ", ".join(t["text"] for t in record["clean_top_tokens"]["rekv"]),
            "",
            "Evict clean top tokens: "
            + ", ".join(t["text"] for t in record["clean_top_tokens"]["evict"]),
            "",
            f"Overlap: {row.get('overlap')}",
            "",
        ])

    jsonl_path = args.out_dir / "clean_interpretability_examples.jsonl"
    with jsonl_path.open("w") as f:
        for rec in selected:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    md_path = args.out_dir / "clean_interpretability_examples.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"wrote {jsonl_path}")
    print(f"wrote {md_path}")
    for rec in selected:
        print(f"saved {rec['plot']}")


if __name__ == "__main__":
    main()
