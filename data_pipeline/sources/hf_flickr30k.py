from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from data_pipeline.schema import MetadataRow, normalize_split
from data_pipeline.splits import deterministic_split


def iter_hf_flickr30k(config: dict[str, Any]) -> Iterable[dict[str, Any]]:
    source = config["source"]
    dataset_name = source.get("hf_dataset", "nlphuji/flickr30k")
    split = "test"
    streaming = bool(source.get("streaming", True))

    dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    yield from dataset


def build_flickr30k_metadata(config: dict[str, Any], raw_root: Path) -> list[MetadataRow]:
    dataset_name = config.get("dataset_name", "flickr30k")
    metadata_cfg = config.get("metadata", {})
    source_cfg = config["source"]

    save_raw_images = bool(metadata_cfg.get("save_raw_images", True))
    use_source_split = bool(metadata_cfg.get("use_source_split_if_available", True))

    ratios = metadata_cfg.get("fallback_split_ratios", {})
    seed = int(metadata_cfg.get("fallback_split_seed", 42))
    train_ratio = float(ratios.get("train", 0.90))
    validation_ratio = float(ratios.get("validation", 0.05))
    test_ratio = float(ratios.get("test", 0.05))

    image_dir = raw_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    rows: list[MetadataRow] = []

    for item in tqdm(iter_hf_flickr30k(config), desc="Reading Flickr30K"):
        image_id = str(item.get("img_id") or Path(str(item["filename"])).stem)
        file_name = str(item.get("filename") or f"{image_id}.jpg")
        relative_image_path = f"images/{file_name}"

        source_split = normalize_split(item.get("split")) if use_source_split else None
        split = source_split or deterministic_split(
            image_id=image_id,
            seed=seed,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
        )

        if save_raw_images:
            save_image_if_needed(item.get("image"), image_dir / file_name)

        captions = item.get("caption") or item.get("captions")
        sentids = item.get("sentids") or []
        if not isinstance(captions, list):
            raise ValueError(f"Expected captions list for image_id={image_id}")

        for caption_index, caption in enumerate(captions):
            raw_caption_id = sentids[caption_index] if caption_index < len(sentids) else None
            caption_id = (
                f"{dataset_name}_{raw_caption_id}"
                if raw_caption_id is not None
                else f"{dataset_name}_{image_id}_{caption_index}"
            )

            rows.append(
                MetadataRow(
                    dataset_name=dataset_name,
                    image_id=image_id,
                    caption_id=caption_id,
                    caption_index=caption_index,
                    caption=str(caption),
                    file_name=file_name,
                    relative_image_path=relative_image_path,
                    split=split,
                    source=source_cfg.get("hf_dataset", "nlphuji/flickr30k"),
                )
            )

    return rows


def save_image_if_needed(image_obj: Any, output_path: Path) -> None:
    if output_path.exists():
        return

    if image_obj is None:
        raise ValueError(f"Image object is missing for {output_path.name}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(image_obj, Image.Image):
        image_obj.convert("RGB").save(output_path, quality=95)
        return

    if isinstance(image_obj, dict):
        if image_obj.get("path"):
            Image.open(image_obj["path"]).convert("RGB").save(output_path, quality=95)
            return
        if image_obj.get("bytes"):
            from io import BytesIO

            Image.open(BytesIO(image_obj["bytes"])).convert("RGB").save(output_path, quality=95)
            return

    raise TypeError(f"Unsupported image object type for {output_path.name}: {type(image_obj)}")

