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
# 10 Major Categories mapping logic
CAT_QA = "QA"              # QA / Question Answering / Reading Comprehension
CAT_CLS = "Classification" # Sentiment, NLI, Topic, Toxicity, Choice
CAT_EXT = "Extraction"     # NER, Tagging, Slot Filling, Span Extraction
CAT_SUM = "Summarization"  # Summarization
CAT_REW = "Rewriting"      # Paraphrase, Style Transfer, Simplification
CAT_TRN = "Translation"    # Translation
CAT_DIA = "Dialogue"       # Dialogue, Chat
CAT_REA = "Reasoning"      # Math, Code, Logic, Program
CAT_COR = "Correction"     # Grammar, Spelling, GEC
CAT_GEN = "Generation"     # Open generation, Story, Poem, Data-to-text

ALL_CATEGORIES = [CAT_QA, CAT_CLS, CAT_EXT, CAT_SUM, CAT_REW, CAT_TRN, CAT_DIA, CAT_REA, CAT_COR, CAT_GEN]

# Keywords priority list. Order matters within the list for specificity.
# Keys are the Category Names. Values are list of lowercase keywords to match in definition/task_name.
_CATEGORY_DEFINITIONS = {
    CAT_TRN: ["translation", "translate", "convert to", "from english", "to english", "german", "french", "spanish"],
    CAT_COR: ["correction", "grammar", "spelling", "punctuation", "gec", "correct the", "fix the", "error detection"],
    CAT_REA: ["math", "arithmetic", "algebra", "calculation", "reasoning", "logical", "deduce", "inference", "python", "code", "sql", "program", "compile", "puzzle"],
    CAT_SUM: ["summariz", "summary", "abstractive", "compression", "headline"],
    CAT_DIA: ["dialogue", "conversation", "chatbot", "response generation", "interact"],
    CAT_EXT: ["extract", "ner", "named entity", "entity", "tagging", "slot filling", "labeling", "span"],
    CAT_QA:  ["question answering", "answer generation", "reading comprehension", "answer the question", "qa", "mrc", "context"],
    CAT_CLS: ["classification", "classify", "sentiment", "nli", "entailment", "hypothesis", "premise", "stance", "topic", "category", "selection", "multiple choice", "choose", "toxicity"],
    CAT_REW: ["paraphrase", "rewrite", "rephrase", "simplify", "simplification", "style transfer", "expansion", "edit"],
    CAT_GEN: ["generate a story", "write a story", "poem", "title generation", "data-to-text", "creative writing", "generate a description"],
}

def classify_task_by_definition(defn: str, tname: str) -> str:
    """Classify a task into one of the 10 categories based on text matching."""
    text = normalize_text(defn + " " + tname)
    
    # Check specific categories first
    for cat, keywords in _CATEGORY_DEFINITIONS.items():
        for kw in keywords:
            if kw in text:
                return cat
    
    # Fallback heuristics for broader matches
    if "generate" in text:
        return CAT_GEN
    
    # Default fallback if nothing matches
    return "Misc"

def scan_and_pool_tasks(
    dataset_name: str,
    split: str,
    target_categories: Sequence[str],
    min_tasks_per_category: int,
    seed: int = 42,
    max_scan_examples: int = 150_000,
) -> Dict[str, List[str]]:
    """
    Scans the dataset and buckets unique task_names into the 10 categories.
    """
    rng = random.Random(seed)
    
    # Initialize pools
    pools: Dict[str, List[str]] = {c: [] for c in target_categories}
    pools["Misc"] = [] # Buffer for uncategorized
    
    seen_tasks = set()
    ds = _load_dataset_safe(dataset_name, split=split, streaming=True, seed=seed, shuffle_buffer=10_000)

    scanned_count = 0
    full_categories = 0
    
    for ex in ds:
        scanned_count += 1
        tname = ex.get("task_name")
        if not tname or tname in seen_tasks:
            continue

        defn = ex.get("definition", "")
        
        # Determine category
        cat = classify_task_by_definition(defn, tname)
        
        # Store if it's a target category (or Misc)
        if cat in pools:
            pools[cat].append(tname)
            seen_tasks.add(tname)
        elif "Misc" in pools:
             pools["Misc"].append(tname)
             seen_tasks.add(tname)

        # Check stopping condition
        # We stop if we have scanned too much OR we have enough for all requested categories
        if scanned_count > max_scan_examples:
            print(f"[Info] Max scan limit reached ({max_scan_examples}).")
            break
            
        # Optimization: Early exit if all pools are saturated (e.g. 2x the needed amount)
        saturated = True
        for c in target_categories:
            if len(pools[c]) < min_tasks_per_category:
                saturated = False
                break
        if saturated:
            print(f"[Info] All categories saturated with at least {min_tasks_per_category} tasks.")
            break

    # Log stats
    print(f"[Info] Scanned {scanned_count} examples. Found unique tasks per category:")
    for c in target_categories:
        print(f"  - {c}: {len(pools[c])}")
    
    return pools

def make_client_task_map(
    num_clients: int,
    dataset_name: str,
    mode: str,
    categories: Sequence[str], # Ignored in explicit, used to filter in keyword mode if needed
    tasks_per_category: int,
    seed: int = 42,
    explicit_json: Optional[Dict] = None,
) -> Dict[int, List[str]]:
    """
    Assigns each client a specific Task Category (from the 10 types),
    then assigns 'tasks_per_category' UNIQUE tasks to that client.
    """
    if mode == "explicit":
        assert explicit_json is not None, "explicit mode requires json"
        mapping = {}
        for k, v in explicit_json.items():
            mapping[int(k)] = v if isinstance(v, list) else [v]
        return mapping

    # 1. Assign a Category to each Client
    # If user provided specific categories in argument, use those cyclically. 
    # Otherwise use ALL_CATEGORIES.
    available_cats = list(categories) if categories else ALL_CATEGORIES
    client_cats = [available_cats[i % len(available_cats)] for i in range(num_clients)]
    
    # 2. Determine how many tasks we need to fetch total per category
    from collections import Counter
    cat_counts = Counter(client_cats)
    
    # We ask for a bit more than strictly needed to have variety
    needed_per_cat = max(cat_counts.values()) * tasks_per_category
    
    # 3. Scan dataset to build pools
    # We pass 'available_cats' to ensure we focus on finding those
    pools = scan_and_pool_tasks(
        dataset_name=dataset_name, 
        split="train", 
        target_categories=available_cats, 
        min_tasks_per_category=needed_per_cat, 
        seed=seed
    )
    
    # 4. Distribute tasks ensuring uniqueness
    mapping = {}
    global_used_tasks = set()
    
    # Helper to get unique tasks from a pool
    def pop_unique_from_pool(pool_list, n):
        selected = []
        remainder = []
        for t in pool_list:
            if len(selected) < n:
                if t not in global_used_tasks:
                    selected.append(t)
                    global_used_tasks.add(t)
                else:
                    remainder.append(t) # Already used by another client, put back
            else:
                remainder.append(t)
        return selected, remainder

    for i in range(num_clients):
        target_cat = client_cats[i]
        required_n = tasks_per_category
        
        my_tasks = []
        
        # A. Try to take from primary category
        found, left_over = pop_unique_from_pool(pools.get(target_cat, []), required_n)
        pools[target_cat] = left_over # Update pool
        my_tasks.extend(found)
        
        # B. If not enough, try to take from Misc
        if len(my_tasks) < required_n:
            needed = required_n - len(my_tasks)
            found, left_over = pop_unique_from_pool(pools.get("Misc", []), needed)
            pools["Misc"] = left_over
            my_tasks.extend(found)
            
        # C. If still not enough, steal from OTHER categories (as long as unique)
        if len(my_tasks) < required_n:
            other_cats = [c for c in available_cats if c != target_cat]
            random.shuffle(other_cats)
            for oc in other_cats:
                needed = required_n - len(my_tasks)
                if needed <= 0: break
                found, left_over = pop_unique_from_pool(pools.get(oc, []), needed)
                pools[oc] = left_over
                my_tasks.extend(found)
        
        # D. Final Resort: If the dataset scan yielded extremely few tasks total
        # We allow reusing tasks that OTHER clients have used (breaking uniqueness only for survival)
        if len(my_tasks) < required_n:
             all_found_tasks = list(global_used_tasks)
             needed = required_n - len(my_tasks)
             if all_found_tasks:
                 # Sample randomly from what we have
                 refill = [random.choice(all_found_tasks) for _ in range(needed)]
                 my_tasks.extend(refill)

        mapping[i] = my_tasks
        
    # Print Allocation Summary
    print("\n[Data Allocation Summary]")
    for cid, tasks in mapping.items():
        print(f"  Client {cid} ({client_cats[cid]}): {len(tasks)} tasks -> {tasks}")

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
