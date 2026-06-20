from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset
from tqdm import tqdm

from data_pipeline.schema import MetadataRow, normalize_split
from data_pipeline.splits import deterministic_split


def iter_hf_table(config: dict[str, Any]) -> Iterable[dict[str, Any]]:
    source = config["source"]
    dataset_name = source["hf_dataset"]
    split = source.get("split", "train")
    streaming = bool(source.get("streaming", True))
    dataset_config = source.get("hf_config")
    max_rows = source.get("max_rows")

    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=split, streaming=streaming)
    else:
        dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    for index, item in enumerate(dataset):
        if max_rows is not None and index >= int(max_rows):
            break
        yield item


def build_hf_table_metadata(config: dict[str, Any], raw_root: Path) -> list[MetadataRow]:
    dataset_name = config.get("dataset_name", "dataset")
    source_cfg = config["source"]
    metadata_cfg = config.get("metadata", {})

    image_uri_column = source_cfg.get("image_uri_column", source_cfg.get("image_url_column", "url"))
    caption_column = source_cfg.get("caption_column", "caption")
    id_column = source_cfg.get("id_column")
    split_column = source_cfg.get("split_column")
    source_name = source_cfg.get("hf_dataset", "hf_table")

    use_source_split = bool(metadata_cfg.get("use_source_split_if_available", True))
    ratios = metadata_cfg.get("fallback_split_ratios", {})
    seed = int(metadata_cfg.get("fallback_split_seed", 42))
    train_ratio = float(ratios.get("train", 1.0))
    validation_ratio = float(ratios.get("validation", 0.0))
    test_ratio = float(ratios.get("test", 0.0))

    rows: list[MetadataRow] = []
    for index, item in enumerate(tqdm(iter_hf_table(config), desc=f"Reading {dataset_name}")):
        caption = item.get(caption_column)
        image_uri = item.get(image_uri_column)
        if not caption or not image_uri:
            continue

        image_id = str(item.get(id_column) if id_column else item.get("id", index))
        file_name = str(item.get("file_name") or item.get("filename") or f"{image_id}.jpg")

        source_split = normalize_split(item.get(split_column)) if split_column and use_source_split else None
        split = source_split or deterministic_split(
            image_id=image_id,
            seed=seed,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
        )

        rows.append(
            MetadataRow(
                dataset_name=str(dataset_name),
                image_id=image_id,
                caption_id=f"{dataset_name}_{image_id}_0",
                caption_index=0,
                caption=str(caption),
                file_name=file_name,
                image_uri=str(image_uri),
                split=split,
                source=source_name,
            )
        )

    return rows
