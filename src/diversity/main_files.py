import argparse
import csv
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from datasets import Dataset
from transformers import AutoConfig, GPT2Tokenizer, GPT2LMHeadModel

from task2vec import Task2Vec
import task_similarity


def _load_texts(path: Path, text_field: str) -> list[str]:
    texts: list[str] = []
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get(text_field, "")
                if text:
                    texts.append(text)
    elif suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        try:
            csv.field_size_limit(sys.maxsize)
        except OverflowError:
            csv.field_size_limit(2**31 - 1)
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                text = row.get(text_field, "")
                if text:
                    texts.append(text)
    else:
        raise ValueError(f"Unsupported file type: {path}")
    return texts


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", nargs="+", required=True, help="Paths to .jsonl/.csv/.tsv files.")
    parser.add_argument("--text_field", default="text", help="Field name to read text from.")
    parser.add_argument("--output_dir", required=True, help="Directory to write .npy outputs.")
    parser.add_argument("--num_tasks", type=int, required=True, help="Number of Task2Vec embeddings to compute.")
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
        texts = _load_texts(path, args.text_field)
        if not texts:
            print(f"Skipping {path} (no texts found).")
            continue

        embeddings, losses = [], []
        for task_num in range(args.num_tasks):
            seed = args.seed + task_num
            rng = np.random.default_rng(seed)
            batch_texts = _sample_texts(rng, texts, args.batch_size)
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
            if loss is not None:
                losses.append(loss)

        stem = path.stem
        np.save(output_dir / f"{stem}_embeddings_{len(embeddings)}tasks.npy", embeddings)
        if losses:
            np.save(output_dir / f"{stem}_loss_{len(losses)}tasks.npy", losses)
        _plot_pca(embeddings, f"{stem} Task2Vec PCA", output_dir / f"{stem}_task2vec_pca.png")
        if len(embeddings) > 1:
            distance_matrix = task_similarity.pdist(embeddings, distance="cosine")
            div_coeff, conf_interval = task_similarity.stats_of_distance_matrix(distance_matrix)
            print(f"{stem}: diversity coefficient={div_coeff:.6f}, ci={conf_interval:.6f}")
        else:
            print(f"{stem}: diversity coefficient unavailable (need >1 embedding)")


if __name__ == "__main__":
    main()
