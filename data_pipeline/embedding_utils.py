from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence, TypeVar

import pandas as pd
import torch
import torch.nn.functional as F


T = TypeVar("T")


def resolve_device(device_name: str | None) -> torch.device:
    if not device_name or device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def l2_normalize(features: torch.Tensor, enabled: bool) -> torch.Tensor:
    if not enabled:
        return features
    return F.normalize(features, dim=-1)


def batches(values: Sequence[T], batch_size: int) -> Iterable[Sequence[T]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def save_split_parquets(df: pd.DataFrame, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for split, split_df in df.groupby("split", sort=True):
        path = output_dir / f"{split}.parquet"
        split_df.to_parquet(path, index=False)
        saved.append(str(path))
    return saved


def write_manifest(output_dir: Path, manifest: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "manifest.json"
    manifest = {
        **manifest,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path

