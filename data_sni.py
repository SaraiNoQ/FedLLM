from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Iterator, Optional, Sequence, Tuple

import os
import re
import random

import torch
from torch.utils.data import IterableDataset, DataLoader

from datasets import load_dataset

_DATASET_CACHE: Dict[Tuple[str, str, bool], object] = {}

def _load_dataset_safe(dataset_name: str, split: str, streaming: bool, seed: int, shuffle_buffer: int | None = None):
    local_path = dataset_name
    load_kwargs = {}
    if os.path.isdir(local_path):
        cache_id = os.path.abspath(local_path)
    else:
        if os.environ.get("HF_ENDPOINT", "") != "https://huggingface.co":
            os.environ["HF_ENDPOINT"] = "https://huggingface.co"
        load_kwargs["trust_remote_code"] = True
        cache_id = dataset_name
    cache_key = (cache_id, split, bool(streaming))
    base_ds = _DATASET_CACHE.get(cache_key)
    if base_ds is None:
        base_ds = load_dataset(local_path, split=split, streaming=streaming, **load_kwargs)
        _DATASET_CACHE[cache_key] = base_ds
    ds = base_ds
    if shuffle_buffer:
        try:
            ds = ds.shuffle(buffer_size=shuffle_buffer, seed=seed)
        except Exception:
            pass
    return ds

# -------- Prompt formatting --------
def build_prompt(definition: str, inputs: str) -> str:
    definition = (definition or "").strip()
    inputs = (inputs or "").strip()
    if inputs:
        return f"### Instruction:\n{definition}\n\n### Input:\n{inputs}\n\n### Response:\n"
    else:
        return f"### Instruction:\n{definition}\n\n### Response:\n"

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

# -------- Task selection --------
_CATEGORY_KEYWORDS = {
    "translation": ["translate", "translation", "convert to", "from english", "to english"],
    "reasoning": ["logical", "reason", "deduce", "inference", "entail", "contradict", "premise"],
    "code": ["sql", "python", "code", "program", "function", "bug", "compile"],
    "rewriting": ["paraphrase", "rewrite", "rephrase", "summarize", "simplify", "title"],
}

def choose_tasks_by_keyword_scan(
    dataset_name: str,
    split: str,
    categories: Sequence[str],
    tasks_per_category: int,
    seed: int = 42,
    max_scan_examples: int = 200_000,
) -> Dict[str, List[str]]:
    """Scan streaming dataset and pick unique task_names per category.

    This avoids enumerating all tasks (which can be huge).
    We match categories by searching keywords in `definition`.
    """
    rng = random.Random(seed)
    categories = list(categories)
    selected: Dict[str, List[str]] = {c: [] for c in categories}
    seen_tasks = set()

    ds = _load_dataset_safe(dataset_name, split=split, streaming=True, seed=seed, shuffle_buffer=20_000)

    def match_cat(defn: str, cat: str) -> bool:
        defn_n = normalize_text(defn)
        for kw in _CATEGORY_KEYWORDS.get(cat, [cat]):
            if kw in defn_n:
                return True
        return False

    scanned = 0
    for ex in ds:
        scanned += 1
        if scanned > max_scan_examples:
            break
        tname = ex.get("task_name", None)
        if not tname or tname in seen_tasks:
            continue
        defn = ex.get("definition", "")
        for c in categories:
            if len(selected[c]) >= tasks_per_category:
                continue
            if match_cat(defn, c):
                selected[c].append(tname)
                seen_tasks.add(tname)
        if all(len(selected[c]) >= tasks_per_category for c in categories):
            break

    # If some categories are empty, fall back to random tasks encountered
    if any(len(v) < tasks_per_category for v in selected.values()):
        # collect additional task names
        more = []
        ds2 = _load_dataset_safe(dataset_name, split=split, streaming=True, seed=seed)
        for ex in ds2:
            tname = ex.get("task_name")
            if tname and tname not in seen_tasks:
                more.append(tname)
            if len(more) > 10_000:
                break
        rng.shuffle(more)
        for c in categories:
            while len(selected[c]) < tasks_per_category and more:
                selected[c].append(more.pop())
    return selected

def make_client_task_map(
    num_clients: int,
    dataset_name: str,
    mode: str,
    categories: Sequence[str],
    tasks_per_category: int,
    seed: int = 42,
    explicit_json: Optional[Dict] = None,
) -> Dict[int, List[str]]:
    """Return mapping client_id -> list of task_names."""
    if mode == "explicit":
        assert explicit_json is not None, "explicit mode requires json"
        mapping = {}
        for k, v in explicit_json.items():
            mapping[int(k)] = v if isinstance(v, list) else [v]
        return mapping

    # keyword mode
    picked = choose_tasks_by_keyword_scan(
        dataset_name=dataset_name,
        split="train",
        categories=categories,
        tasks_per_category=tasks_per_category,
        seed=seed,
    )
    tasks_flat = []
    for c in categories:
        tasks_flat.extend(picked[c])
    # assign round-robin
    mapping = {}
    for i in range(num_clients):
        mapping[i] = [tasks_flat[i % len(tasks_flat)]]
    return mapping

# -------- Iterable datasets --------
@dataclass
class SNITextExample:
    task_name: str
    prompt: str
    target: str

class SNIIterable(IterableDataset):
    """Streaming iterable for a subset of tasks."""
    def __init__(
        self,
        dataset_name: str,
        split: str,
        task_names: Sequence[str],
        tokenizer,
        seq_len: int,
        seed: int = 42,
        max_examples: Optional[int] = None,
        shuffle_buffer: int = 10_000,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.task_names = set(task_names)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.seed = seed
        self.max_examples = max_examples
        self.shuffle_buffer = shuffle_buffer

    def _iter_text(self) -> Iterator[SNITextExample]:
        ds = _load_dataset_safe(self.dataset_name, split=self.split, streaming=True, seed=self.seed, shuffle_buffer=self.shuffle_buffer)
        count = 0
        for ex in ds:
            tname = ex.get("task_name", "")
            if tname not in self.task_names:
                continue
            definition = ex.get("definition", "")
            inputs = ex.get("inputs", "")
            targets = ex.get("targets", "")
            prompt = build_prompt(definition, inputs)
            target = (targets or "").strip()
            yield SNITextExample(task_name=tname, prompt=prompt, target=target)
            count += 1
            if self.max_examples is not None and count >= self.max_examples:
                break

    def __iter__(self):
        skipped = 0
        yielded = 0
        for te in self._iter_text():
            # Tokenize prompt+target for causal LM
            prompt_ids = self.tokenizer(te.prompt, add_special_tokens=False).input_ids
            target_ids = self.tokenizer(te.target, add_special_tokens=False).input_ids
            eos = [self.tokenizer.eos_token_id]

            input_ids = prompt_ids + target_ids + eos
            labels = [-100] * len(prompt_ids) + target_ids + eos
            if len(input_ids) > self.seq_len:
                input_ids = input_ids[: self.seq_len]
                labels = labels[: self.seq_len]

            if not any(l != -100 for l in labels):
                skipped += 1
                if skipped % 100 == 0:
                    print(f"[WARN] skipped {skipped} samples with empty labels in split={self.split}")
                continue
            yielded += 1
            if yielded % 100 == 0:
                print(f"[INFO] yielded {yielded} samples in split={self.split}")

            attn = [1] * len(input_ids)
            yield {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
                "prompt_text": te.prompt,
                "target_text": te.target,
                "task_name": te.task_name,
            }

def pad_collate(batch, pad_id: int):
    # batch: list of dict with tensors of varying length
    max_len = max(x["input_ids"].shape[0] for x in batch)
    input_ids, labels, attn = [], [], []
    prompts, targets, task_names = [], [], []
    for x in batch:
        L = x["input_ids"].shape[0]
        pad_len = max_len - L
        input_ids.append(torch.cat([x["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)]))
        labels.append(torch.cat([x["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))
        attn.append(torch.cat([x["attention_mask"], torch.zeros((pad_len,), dtype=torch.long)]))
        prompts.append(x["prompt_text"])
        targets.append(x["target_text"])
        task_names.append(x["task_name"])
    return {
        "input_ids": torch.stack(input_ids, dim=0),
        "labels": torch.stack(labels, dim=0),
        "attention_mask": torch.stack(attn, dim=0),
        "prompt_text": prompts,
        "target_text": targets,
        "task_name": task_names,
    }

def make_sni_client_loaders(
    tokenizer,
    dataset_name: str,
    client_task_map: Dict[int, List[str]],
    seq_len: int,
    micro_batch: int,
    seed: int = 42,
    train_max_examples: Optional[int] = None,
    eval_max_examples: Optional[int] = 512,
):
    """Return dicts: client_id -> (train_loader, val_loader, test_loader)."""
    loaders = {}
    pad_id = tokenizer.pad_token_id

    num_workers = min(4, os.cpu_count() or 0)
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0

    for cid, tasks in client_task_map.items():
        train_ds = SNIIterable(dataset_name, "train", tasks, tokenizer, seq_len, seed=seed+cid, max_examples=train_max_examples)
        val_ds = SNIIterable(dataset_name, "validation", tasks, tokenizer, seq_len, seed=seed+cid+10_000, max_examples=eval_max_examples, shuffle_buffer=2_000)
        test_ds = SNIIterable(dataset_name, "test", tasks, tokenizer, seq_len, seed=seed+cid+20_000, max_examples=eval_max_examples, shuffle_buffer=2_000)

        collate_fn = lambda b, pid=pad_id: pad_collate(b, pid)
        train_loader = DataLoader(
            train_ds,
            batch_size=micro_batch,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=micro_batch,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=micro_batch,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        loaders[cid] = (train_loader, val_loader, test_loader)
    return loaders
