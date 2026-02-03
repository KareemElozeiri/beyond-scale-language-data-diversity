import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import re
import sys
from pathlib import Path
from typing import Optional

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


def _parse_gpus(gpus: str) -> list[int]:
    if not gpus:
        return []
    ids: list[int] = []
    for part in gpus.split(","):
        part = part.strip()
        if not part:
            continue
        ids.append(int(part))
    return ids


def _maybe_cache_tokenized(
    tokenizer,
    texts: list[str],
    max_seq_length: int,
    max_cached_texts: int,
    seed: int,
) -> Dataset:
    if max_cached_texts and len(texts) > max_cached_texts:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(texts), size=max_cached_texts, replace=False)
        texts = [texts[i] for i in indices]
    return _tokenize_batch(tokenizer, texts, max_seq_length)


def _configure_torch_for_speed(tf32: bool) -> None:
    if torch.cuda.is_available() and tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def _load_model_and_tokenizer(pretrained: bool, cache_dir: str, device_id: Optional[int]):
    if pretrained:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=cache_dir or None)
        model = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir=cache_dir or None)
    else:
        config = AutoConfig.from_pretrained("gpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=cache_dir or None)
        model = GPT2LMHeadModel(config)

    if torch.cuda.is_available() and device_id is not None:
        device = torch.device(f"cuda:{device_id}")
    else:
        device = torch.device("cpu")

    model = model.to(device)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        assert tokenizer.pad_token_id == 50256

    return tokenizer, model, device


def _tokenize_batch(tokenizer, batch_texts: list[str], max_seq_length: int) -> Dataset:
    tokenized = Dataset.from_dict({"text": batch_texts}).map(
        lambda examples: tokenizer(
            examples["text"],
            padding="max_length",
            max_length=max_seq_length,
            truncation=True,
            return_tensors="pt",
        ),
        batched=True,
        remove_columns=["text"],
    )
    tokenized.set_format("torch")
    return tokenized


def _compute_embedding(
    batch_texts: list[str],
    tokenizer,
    model,
    max_seq_length: int,
    classifier_opts: dict,
    loader_opts: dict,
):
    tokenized = _tokenize_batch(tokenizer, batch_texts, max_seq_length)
    embedding, loss = Task2Vec(model, classifier_opts=classifier_opts, loader_opts=loader_opts).embed(tokenized)
    return embedding, loss


def _compute_embedding_from_tokenized(tokenized: Dataset, model, classifier_opts: dict, loader_opts: dict):
    embedding, loss = Task2Vec(model, classifier_opts=classifier_opts, loader_opts=loader_opts).embed(tokenized)
    return embedding, loss


def _worker_loop(device_id: int, job_queue: mp.JoinableQueue, result_queue: mp.Queue, worker_args: dict) -> None:
    try:
        torch.set_num_threads(1)
        torch.cuda.set_device(device_id)
        _configure_torch_for_speed(worker_args["tf32"])
        tokenizer, model, _device = _load_model_and_tokenizer(
            worker_args["pretrained"],
            worker_args["cache_dir"],
            device_id,
        )
        loader_opts = worker_args["loader_opts"]

        while True:
            job = job_queue.get()
            if job is None:
                job_queue.task_done()
                break
            label, task_num, batch_texts, seed = job
            try:
                classifier_opts = {
                    "finetune": worker_args["finetune"],
                    "seed": seed,
                    "epochs": worker_args["epochs"],
                    "task_batch_size": worker_args["task_batch_size"],
                }
                embedding, loss = _compute_embedding(
                    batch_texts,
                    tokenizer,
                    model,
                    worker_args["max_seq_length"],
                    classifier_opts,
                    loader_opts,
                )
                result_queue.put(("ok", label, task_num, embedding, loss))
            except Exception as exc:
                result_queue.put(("error", label, task_num, None, repr(exc)))
                job_queue.task_done()
                return
            job_queue.task_done()
    except Exception as exc:
        result_queue.put(("error", "__worker__", -1, None, repr(exc)))


def _save_outputs(
    output_dir: Path,
    stem: str,
    labels: list[str],
    label_tags: dict[str, str],
    label_embeddings: dict[str, list],
    label_losses: dict[str, list],
) -> None:
    all_embeddings: list = []
    all_losses: list = []
    all_labels: list[str] = []

    for label in labels:
        embeddings = label_embeddings.get(label, [])
        if not embeddings:
            continue
        losses = label_losses.get(label, [])

        label_tag = label_tags[label]
        label_prefix = f"{stem}__label_{label_tag}"
        np.save(output_dir / f"{label_prefix}_embeddings_{len(embeddings)}tasks.npy", embeddings)

        losses_to_save = [loss for loss in losses if loss is not None]
        if losses_to_save:
            np.save(output_dir / f"{label_prefix}_loss_{len(losses_to_save)}tasks.npy", losses_to_save)

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

        all_embeddings.extend(embeddings)
        all_labels.extend([label] * len(embeddings))
        if losses_to_save:
            all_losses.extend(losses_to_save)

    if all_embeddings:
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
    parser.add_argument("--gpus", default="", help="Comma-separated GPU ids to use for parallel embedding.")
    parser.add_argument("--loader_batch_size", type=int, default=None, help="DataLoader batch size for Task2Vec.")
    parser.add_argument("--loader_num_workers", type=int, default=0, help="DataLoader workers per process.")
    parser.add_argument("--pin_memory", action="store_true", help="Enable DataLoader pin_memory.")
    parser.add_argument("--prefetch_factor", type=int, default=None, help="DataLoader prefetch_factor (requires workers).")
    parser.add_argument("--persistent_workers", action="store_true", help="Use persistent DataLoader workers.")
    parser.add_argument("--tf32", action="store_true", help="Enable TF32 matmul/conv (faster on Ampere+).")
    parser.add_argument(
        "--pretokenize",
        action="store_true",
        help="Tokenize all texts once per label and sample from the cached tokens (single-GPU path only).",
    )
    parser.add_argument(
        "--max_cached_texts",
        type=int,
        default=0,
        help="Max texts to cache per label when using --pretokenize (0 = all).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    loader_batch_size = args.loader_batch_size
    if loader_batch_size is None:
        loader_batch_size = min(64, args.batch_size)

    loader_opts = {
        "batch_size": loader_batch_size,
        "num_workers": args.loader_num_workers,
        "pin_memory": args.pin_memory,
        "prefetch_factor": args.prefetch_factor,
        "persistent_workers": args.persistent_workers,
    }

    gpu_ids = _parse_gpus(args.gpus)
    use_parallel = len(gpu_ids) > 1 and torch.cuda.is_available()

    if not gpu_ids and torch.cuda.is_available():
        gpu_ids = [0]

    if use_parallel:
        ctx = mp.get_context("spawn")

        worker_args = {
            "pretrained": args.pretrained,
            "cache_dir": args.cache_dir,
            "finetune": args.finetune,
            "epochs": args.epochs,
            "task_batch_size": args.batch_size,
            "max_seq_length": args.max_seq_length,
            "loader_opts": loader_opts,
            "tf32": args.tf32,
        }

        for input_file in args.input_files:
            path = Path(input_file)
            texts_by_label = _load_texts_by_label(path, args.text_field, args.label_field)
            if not texts_by_label:
                raise ValueError(
                    f"No labeled texts found in {path} for label field '{args.label_field}'."
                )

            labels = sorted(texts_by_label.keys())
            label_tags = _build_label_tags(labels)
            label_embeddings: dict[str, list] = {label: [None] * args.num_tasks for label in labels}
            label_losses: dict[str, list] = {label: [None] * args.num_tasks for label in labels}

            job_queue: mp.JoinableQueue = ctx.JoinableQueue()
            result_queue: mp.Queue = ctx.Queue()

            processes: list[mp.Process] = []
            for device_id in gpu_ids:
                proc = ctx.Process(
                    target=_worker_loop,
                    args=(device_id, job_queue, result_queue, worker_args),
                )
                proc.start()
                processes.append(proc)

            total_jobs = 0
            for label in labels:
                label_texts = texts_by_label[label]
                if not label_texts:
                    continue
                for task_num in range(args.num_tasks):
                    seed = args.seed + task_num
                    rng = np.random.default_rng(seed)
                    batch_texts = _sample_texts(rng, label_texts, args.batch_size)
                    if not batch_texts:
                        continue
                    job_queue.put((label, task_num, batch_texts, seed))
                    total_jobs += 1

            for _ in processes:
                job_queue.put(None)

            error = None
            received = 0
            while received < total_jobs:
                status, label, task_num, embedding, loss = result_queue.get()
                if status == "error":
                    error = f"{label}:{task_num}: {loss}"
                    break
                label_embeddings[label][task_num] = embedding
                label_losses[label][task_num] = loss
                received += 1

            if error is not None:
                for proc in processes:
                    proc.terminate()
                for proc in processes:
                    proc.join()
                raise RuntimeError(f"Worker error: {error}")

            job_queue.join()
            for proc in processes:
                proc.join()

            label_embeddings_clean: dict[str, list] = {}
            label_losses_clean: dict[str, list] = {}
            for label in labels:
                embeddings = [e for e in label_embeddings[label] if e is not None]
                losses = [l for l in label_losses[label] if l is not None]
                label_embeddings_clean[label] = embeddings
                label_losses_clean[label] = losses

            _save_outputs(output_dir, path.stem, labels, label_tags, label_embeddings_clean, label_losses_clean)

    else:
        device_id = gpu_ids[0] if gpu_ids else None
        _configure_torch_for_speed(args.tf32)
        tokenizer, model, _device = _load_model_and_tokenizer(args.pretrained, args.cache_dir, device_id)

        for input_file in args.input_files:
            path = Path(input_file)
            texts_by_label = _load_texts_by_label(path, args.text_field, args.label_field)
            if not texts_by_label:
                raise ValueError(
                    f"No labeled texts found in {path} for label field '{args.label_field}'."
                )

            labels = sorted(texts_by_label.keys())
            label_tags = _build_label_tags(labels)
            label_embeddings: dict[str, list] = {}
            label_losses: dict[str, list] = {}

            for label in labels:
                label_texts = texts_by_label[label]
                if not label_texts:
                    continue
                token_cache = None
                if args.pretokenize:
                    token_cache = _maybe_cache_tokenized(
                        tokenizer,
                        label_texts,
                        args.max_seq_length,
                        args.max_cached_texts,
                        args.seed,
                    )
                embeddings: list = []
                losses: list = []
                for task_num in range(args.num_tasks):
                    seed = args.seed + task_num
                    rng = np.random.default_rng(seed)
                    classifier_opts = {
                        "finetune": args.finetune,
                        "seed": seed,
                        "epochs": args.epochs,
                        "task_batch_size": args.batch_size,
                    }
                    if token_cache is not None:
                        replace = len(token_cache) < args.batch_size
                        indices = rng.choice(len(token_cache), size=args.batch_size, replace=replace)
                        tokenized = token_cache.select(indices.tolist())
                        tokenized.set_format("torch")
                        embedding, loss = _compute_embedding_from_tokenized(
                            tokenized,
                            model,
                            classifier_opts,
                            loader_opts,
                        )
                    else:
                        batch_texts = _sample_texts(rng, label_texts, args.batch_size)
                        if not batch_texts:
                            break
                        embedding, loss = _compute_embedding(
                            batch_texts,
                            tokenizer,
                            model,
                            args.max_seq_length,
                            classifier_opts,
                            loader_opts,
                        )
                    embeddings.append(embedding)
                    losses.append(loss)

                label_embeddings[label] = embeddings
                label_losses[label] = losses

            _save_outputs(output_dir, path.stem, labels, label_tags, label_embeddings, label_losses)


if __name__ == "__main__":
    main()
