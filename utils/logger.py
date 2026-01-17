from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional
from datetime import datetime

@dataclass
class JsonlLogger:
    log_dir: str
    filename: str = "metrics.jsonl"

    def __post_init__(self):
        os.makedirs(self.log_dir, exist_ok=True)
        self.path = os.path.join(self.log_dir, self.filename)

    def log(self, record: Dict[str, Any]):
        record = dict(record)
        record["_ts"] = datetime.utcnow().isoformat()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def save_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
