from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_pipeline.config import require_abs_path


SPLITS = ("train", "validation", "test")


def dataset_name(config: dict) -> str:
    return str(config.get("dataset_name", "dataset")).strip().replace("/", "_")


def output_root(config: dict) -> Path:
    return require_abs_path(config["local"]["output_root"], "local.output_root")


def dataset_root(config: dict) -> Path:
    root = output_root(config)
    name = dataset_name(config)
    if root.name == name:
        return root
    return root / name


def metadata_dir(config: dict) -> Path:
    return dataset_root(config) / "metadata"


def metadata_files(config: dict) -> list[Path]:
    directory = metadata_dir(config)
    split_files = sorted(
        path
        for path in directory.glob("*.parquet")
        if path.stem in SPLITS or any(path.name.startswith(f"{split}-") for split in SPLITS)
    )
    if split_files:
        return split_files

    legacy_path = output_root(config) / "metadata" / "metadata.parquet"
    if legacy_path.exists():
        return [legacy_path]

    single_path = directory / "metadata.parquet"
    return [single_path]


def read_metadata(config: dict) -> pd.DataFrame:
    paths = metadata_files(config)
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Metadata not found: {missing[0]}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def encoder_output_dir(config: dict, group_name: str, output_name: str) -> Path:
    return dataset_root(config) / group_name / output_name
