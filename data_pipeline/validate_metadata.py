from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_pipeline.schema import VALID_SPLITS


REQUIRED_COLUMNS = {
    "dataset_name",
    "image_id",
    "caption_id",
    "caption_index",
    "caption",
    "file_name",
    "relative_image_path",
    "split",
    "source",
}


def validate_metadata_frame(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Metadata is empty")

    duplicate_caption_ids = df["caption_id"].duplicated().sum()
    if duplicate_caption_ids:
        raise ValueError(f"Found duplicate caption_id values: {duplicate_caption_ids}")

    invalid_splits = sorted(set(df["split"]) - VALID_SPLITS)
    if invalid_splits:
        raise ValueError(f"Found invalid split labels: {invalid_splits}")

    split_counts = df.groupby("image_id")["split"].nunique()
    leaking_images = split_counts[split_counts > 1]
    if len(leaking_images):
        examples = leaking_images.head(10).index.tolist()
        raise ValueError(f"Some image_ids appear in multiple splits: {examples}")

    empty_captions = df["caption"].isna().sum() + (df["caption"].astype(str).str.len() == 0).sum()
    if empty_captions:
        raise ValueError(f"Found empty captions: {empty_captions}")


def validate_image_paths(df: pd.DataFrame, raw_root: Path) -> None:
    missing_paths = []
    for relative_path in sorted(df["relative_image_path"].unique()):
        if not (raw_root / relative_path).exists():
            missing_paths.append(relative_path)
            if len(missing_paths) >= 10:
                break

    if missing_paths:
        raise ValueError(f"Missing local image files, examples: {missing_paths}")

