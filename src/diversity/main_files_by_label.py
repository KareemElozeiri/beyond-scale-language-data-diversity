import argparse
import csv
import hashlib
import json
import os
import re
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from datasets import Dataset
from transformers import AutoConfig, GPT2Tokenizer, GPT2LMHeadModel

from task2vec import Task2Vec
import task_similarity


def _load_texts_by_label(path: Path, text_field: str, label_field: str) -> dict[str, list[str]]:
    texts_by_label: dict[str, list[str]] = {}
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get(text_field, "")
                label = obj.get(label_field, None)
                if text is None or label is None:
                    continue
                text = str(text)
                label_str = str(label).strip()
                if not text or not label_str:
                    continue
                texts_by_label.setdefault(label_str, []).append(text)
    elif suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        try:
            csv.field_size_limit(sys.maxsize)
        except OverflowError:
            csv.field_size_limit(2**31 - 1)
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            if reader.fieldnames is None or label_field not in reader.fieldnames:
                raise ValueError(f"Label field '{label_field}' not found in {path}")
            for row in reader:
                text = row.get(text_field, "")
                label = row.get(label_field, None)
                if text is None or label is None:
                    continue
                text = str(text)
                label_str = str(label).strip()
                if not text or not label_str:
                    continue
                texts_by_label.setdefault(label_str, []).append(text)
    else:
        raise ValueError(f"Unsupported file type: {path}")
    return texts_by_label


def _sample_texts(rng: np.random.Generator, texts: list[str], batch_size: int) -> list[str]:
    if not texts:
        return []
    replace = len(texts) < batch_size
    idx = rng.choice(len(texts), size=batch_size, replace=replace)
    return [texts[i] for i in idx]


def _pca_2d(x: np.ndarray) -> np.ndarray:
    x = x - np.mean(x, axis=0, keepdims=True)
    u, s, _vt = np.linalg.svd(x, full_matrices=False)
    return u[:, :2] * s[:2]


def _plot_pca(embeddings, title: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    if len(embeddings) < 2:
        return
    vectors = [task_similarity.get_hessian(e, normalized=True) for e in embeddings]
    x = np.stack(vectors, axis=0)
    pts = _pca_2d(x)

    plt.figure(figsize=(6, 6))
    plt.scatter(pts[:, 0], pts[:, 1], s=10, alpha=0.6)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


_LABEL_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_label(label: str) -> str:
    label = str(label).strip()
    if not label:
        return "empty"
    label = _LABEL_SAFE_RE.sub("_", label)
    return label[:80] or "empty"


def _build_label_tags(labels: list[str]) -> dict[str, str]:
    used: set[str] = set()
    label_tags: dict[str, str] = {}
    for label in labels:
        base = _sanitize_label(label)
        tag = base
        if tag in used:
            digest = hashlib.sha1(label.encode("utf-8")).hexdigest()[:8]
            tag = f"{base}_{digest}"
        while tag in used:
            digest = hashlib.sha1((label + tag).encode("utf-8")).hexdigest()[:8]
            tag = f"{base}_{digest}"
        used.add(tag)
        label_tags[label] = tag
    return label_tags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", nargs="+", required=True, help="Paths to .jsonl/.csv/.tsv files.")
    parser.add_argument("--text_field", default="text", help="Field name to read text from.")
    parser.add_argument("--label_field", required=True, help="Label field/column to split tasks.")
    parser.add_argument("--output_dir", required=True, help="Directory to write .npy outputs.")
    parser.add_argument("--num_tasks", type=int, required=True, help="Number of Task2Vec embeddings to compute per label.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size per task.")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Max sequence length for tokenizer.")
    parser.add_argument("--finetune", action="store_true", help="Whether to finetune probe network.")
    parser.add_argument("--pretrained", action="store_true", help="Whether to use pretrained GPT-2.")
    parser.add_argument("--epochs", type=int, default=10, help="Finetuning epochs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--cache_dir", default="", help="HF cache dir.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    # Load GPT-2 model and tokenizer (pretrained or randomly initialized)
    if args.pretrained:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=args.cache_dir or None)
        model = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir=args.cache_dir or None)
    else:
        config = AutoConfig.from_pretrained("gpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=args.cache_dir or None)
        model = GPT2LMHeadModel(config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        assert tokenizer.pad_token_id == 50256

    def preprocess_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            max_length=args.max_seq_length,
            truncation=True,
            return_tensors="pt",
        )

    for input_file in args.input_files:
        path = Path(input_file)
        texts_by_label = _load_texts_by_label(path, args.text_field, args.label_field)
        if not texts_by_label:
            raise ValueError(
                f"No labeled texts found in {path} for label field '{args.label_field}'."
            )

        labels = sorted(texts_by_label.keys())
        label_tags = _build_label_tags(labels)
        all_embeddings, all_losses, all_labels = [], [], []

        for label in labels:
            label_texts = texts_by_label[label]
            if not label_texts:
                continue

            embeddings, losses = [], []
            for task_num in range(args.num_tasks):
                seed = args.seed + task_num
                rng = np.random.default_rng(seed)
                batch_texts = _sample_texts(rng, label_texts, args.batch_size)
                if not batch_texts:
                    break
                tokenized = Dataset.from_dict({"text": batch_texts}).map(
                    preprocess_function, batched=True, remove_columns=["text"]
                )
                tokenized.set_format("torch")

                classifier_opts = {
                    "finetune": args.finetune,
                    "seed": seed,
                    "epochs": args.epochs,
                    "task_batch_size": args.batch_size,
                }
                embedding, loss = Task2Vec(deepcopy(model), classifier_opts=classifier_opts).embed(tokenized)
                embeddings.append(embedding)
                all_embeddings.append(embedding)
                all_labels.append(label)
                if loss is not None:
                    losses.append(loss)
                    all_losses.append(loss)

            stem = path.stem
            label_tag = label_tags[label]
            label_prefix = f"{stem}__label_{label_tag}"
            np.save(output_dir / f"{label_prefix}_embeddings_{len(embeddings)}tasks.npy", embeddings)
            if losses:
                np.save(output_dir / f"{label_prefix}_loss_{len(losses)}tasks.npy", losses)
            _plot_pca(
                embeddings,
                f"{stem} label={label} Task2Vec PCA",
                output_dir / f"{label_prefix}_task2vec_pca.png",
            )
            if len(embeddings) > 1:
                distance_matrix = task_similarity.pdist(embeddings, distance="cosine")
                div_coeff, conf_interval = task_similarity.stats_of_distance_matrix(distance_matrix)
                print(f"{stem} label={label}: diversity coefficient={div_coeff:.6f}, ci={conf_interval:.6f}")
            else:
                print(f"{stem} label={label}: diversity coefficient unavailable (need >1 embedding)")

        if all_embeddings:
            stem = path.stem
            combined_prefix = f"{stem}_all_labels"
            np.save(
                output_dir / f"{combined_prefix}_embeddings_{len(all_embeddings)}tasks.npy",
                all_embeddings,
            )
            if all_losses:
                np.save(
                    output_dir / f"{combined_prefix}_loss_{len(all_losses)}tasks.npy",
                    all_losses,
                )
            np.save(
                output_dir / f"{combined_prefix}_labels_{len(all_labels)}tasks.npy",
                np.array(all_labels, dtype=object),
            )
            _plot_pca(
                all_embeddings,
                f"{stem} all labels Task2Vec PCA",
                output_dir / f"{combined_prefix}_task2vec_pca.png",
            )
            if len(all_embeddings) > 1:
                distance_matrix = task_similarity.pdist(all_embeddings, distance="cosine")
                div_coeff, conf_interval = task_similarity.stats_of_distance_matrix(distance_matrix)
                print(f"{stem} all labels: diversity coefficient={div_coeff:.6f}, ci={conf_interval:.6f}")
            else:
                print(f"{stem} all labels: diversity coefficient unavailable (need >1 embedding)")


if __name__ == "__main__":
    main()
