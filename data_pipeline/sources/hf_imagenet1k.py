from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset, load_dataset_builder
from tqdm import tqdm

from data_pipeline.schema import MetadataRow, normalize_split
from data_pipeline.sources.hf_table import label_name_from_features


DEFAULT_DATASET = "ILSVRC/imagenet-1k"
DEFAULT_SPLIT = "validation"
DEFAULT_CAPTION_TEMPLATE = "a photo of a {label}"


def imagenet_split(config: dict[str, Any]) -> str:
    return str(config["source"].get("split", DEFAULT_SPLIT))


def imagenet_image_id(config: dict[str, Any], item: dict[str, Any], index: int) -> str:
    id_column = config["source"].get("id_column")
    raw_id = item.get(id_column) if id_column else item.get("id")
    if raw_id is not None:
        return str(raw_id)
    return f"{normalize_split(imagenet_split(config)) or imagenet_split(config)}_{index:08d}"


def iter_hf_imagenet1k(config: dict[str, Any]) -> Iterable[dict[str, Any]]:
    source = config["source"]
    dataset_name = source.get("hf_dataset", DEFAULT_DATASET)
    split = imagenet_split(config)
    streaming = bool(source.get("streaming", True))
    max_rows = source.get("max_rows")

    dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    for index, item in enumerate(dataset):
        if max_rows is not None and index >= int(max_rows):
            break
        yield item


def imagenet_features(config: dict[str, Any]):
    source = config["source"]
    return load_dataset_builder(source.get("hf_dataset", DEFAULT_DATASET)).info.features


def imagenet_caption(config: dict[str, Any], item: dict[str, Any], features=None) -> str | None:
    source = config["source"]
    label_column = source.get("label_column", "label")
    label_value = item.get(label_column)
    if label_value is None:
        return None

    label_name = label_name_from_features(features, label_column, label_value)
    template = source.get("caption_template", DEFAULT_CAPTION_TEMPLATE)
    return template.format(label=label_name, label_id=label_value)


def imagenet_image_uri(config: dict[str, Any], image_id: str, index: int) -> str:
    source = config["source"]
    template = source.get("image_uri_template", "hf://datasets/{hf_dataset}/{split}/{image_id}")
    return template.format(
        hf_dataset=source.get("hf_dataset", DEFAULT_DATASET),
        split=imagenet_split(config),
        image_id=image_id,
        index=index,
    )


def imagenet_metadata_dict(config: dict[str, Any], item: dict[str, Any], index: int, features=None) -> dict[str, Any] | None:
    dataset_name = str(config.get("dataset_name", "imagenet1k"))
    source = config["source"]
    hf_dataset = source.get("hf_dataset", DEFAULT_DATASET)
    split = normalize_split(source.get("split_override", imagenet_split(config))) or imagenet_split(config)
    image_id = imagenet_image_id(config, item, index)
    caption = imagenet_caption(config, item, features)
    if not caption:
        return None

    file_name = str(item.get("file_name") or item.get("filename") or f"{image_id}.jpg")
    return {
        "dataset_name": dataset_name,
        "image_id": image_id,
        "caption_id": f"{dataset_name}_{image_id}_0",
        "caption_index": 0,
        "caption": caption,
        "file_name": file_name,
        "image_uri": imagenet_image_uri(config, image_id, index),
        "split": split,
        "source": hf_dataset,
    }


def build_imagenet1k_metadata(config: dict[str, Any], raw_root: Path) -> list[MetadataRow]:
    del raw_root
    features = imagenet_features(config)
    rows: list[MetadataRow] = []

    for index, item in enumerate(tqdm(iter_hf_imagenet1k(config), desc="Reading ImageNet-1K")):
        metadata = imagenet_metadata_dict(config, item, index, features)
        if metadata is None:
            continue
        rows.append(MetadataRow(**metadata))

    return rows
