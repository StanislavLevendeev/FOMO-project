from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from data_pipeline.config import load_config
from data_pipeline.embedding_utils import l2_normalize, resolve_device, write_manifest
from data_pipeline.generate_image_embeddings import (
    image_obj_to_rgb,
    load_dino_processor,
    pool_vision_outputs,
)
from data_pipeline.paths import dataset_name, dataset_root, encoder_output_dir, metadata_dir
from data_pipeline.runtime import safe_exit_if_requested
from data_pipeline.schema import normalize_split
from data_pipeline.sources.hf_table import iter_hf_table
from data_pipeline.splits import deterministic_split


DEFAULT_TEXT_MODEL = "wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M"


def generate_streaming_features(config_path: str | Path) -> Path:
    config = load_config(config_path)
    if config.get("source", {}).get("kind") != "hf_table":
        raise ValueError("generate-streaming-features currently supports source.kind=hf_table")

    device = resolve_device(config.get("embeddings", {}).get("device", "auto"))
    normalize = bool(config.get("embeddings", {}).get("normalize", True))
    batch_size = resolve_streaming_batch_size(config)
    max_rows_per_file = int(
        config.get("streaming_features", {}).get(
            "max_rows_per_file",
            config.get("embeddings", {}).get(
                "max_rows_per_file",
                config.get("metadata", {}).get("max_rows_per_file", 100000),
            ),
        )
    )

    text_encoder = load_text_encoder(config, device)
    image_encoder = load_image_encoder(config, device)
    writer = StreamingShardWriter(config, max_rows_per_file)

    skipped = 0
    batch: list[PreparedItem] = []

    with torch.inference_mode():
        for index, item in enumerate(tqdm(iter_hf_table(config), desc=f"Streaming {dataset_name(config)}")):
            prepared = prepare_item(config, item, index)
            if prepared is None:
                skipped += 1
                continue

            try:
                prepared.image = image_obj_to_rgb(item.get(config["source"]["image_column"]))
            except (OSError, TypeError, ValueError) as exc:
                if not bool(config.get("embeddings", {}).get("image_encoder", {}).get("skip_failed_images", False)):
                    raise
                skipped += 1
                print(f"Skipping image_id={prepared.metadata['image_id']}: {exc}")
                continue

            batch.append(prepared)
            if len(batch) >= batch_size:
                encode_and_buffer(batch, text_encoder, image_encoder, writer, normalize)
                batch = []

        if batch:
            encode_and_buffer(batch, text_encoder, image_encoder, writer, normalize)

    writer.close(skipped=skipped)
    print(f"Saved streaming features: {dataset_root(config)}")
    return dataset_root(config)


def resolve_streaming_batch_size(config: dict) -> int:
    streaming_cfg = config.get("streaming_features", {})
    if streaming_cfg.get("batch_size"):
        return int(streaming_cfg["batch_size"])

    embeddings_cfg = config.get("embeddings", {})
    text_batch = int(embeddings_cfg.get("batch_size_text", 128))
    image_batch = int(embeddings_cfg.get("batch_size_image", 32))
    return max(1, min(text_batch, image_batch))


@dataclass
class PreparedItem:
    metadata: dict[str, Any]
    image: Any = None


@dataclass
class TextEncoder:
    kind: str
    name: str
    pretrained: str | None
    output_name: str
    model: Any
    tokenizer: Any
    device: torch.device
    max_length: int | None = None

    def encode(self, captions: list[str]) -> torch.Tensor:
        if self.kind in {"hf_clip_text", "tiny_clip"}:
            tokens = self.tokenizer(
                captions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)
            return self.model(**tokens).text_embeds

        if self.kind == "open_clip":
            tokens = self.tokenizer(captions).to(self.device)
            return self.model.encode_text(tokens)

        raise ValueError(f"Unsupported text encoder kind: {self.kind}")


@dataclass
class ImageEncoder:
    kind: str
    name: str
    pretrained: str | None
    output_name: str
    model: Any
    processor: Any
    device: torch.device
    pooling: str

    def encode(self, images: list[Any]) -> torch.Tensor:
        if self.kind in {"dinov2", "dinov3"}:
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            return pool_vision_outputs(outputs, self.pooling, self.kind)

        if self.kind == "open_clip":
            image_tensors = [self.processor(image) for image in images]
            image_batch = torch.stack(image_tensors).to(self.device)
            return self.model.encode_image(image_batch)

        raise ValueError(f"Unsupported image encoder kind: {self.kind}")


def load_text_encoder(config: dict, device: torch.device) -> TextEncoder:
    text_cfg = config.get("embeddings", {}).get("text_encoder", {})
    kind = str(text_cfg.get("kind", "hf_clip_text")).lower()

    if kind in {"hf_clip_text", "tiny_clip"}:
        from transformers import AutoTokenizer, CLIPTextModelWithProjection

        model_name = text_cfg.get("model_name", DEFAULT_TEXT_MODEL)
        output_name = text_cfg.get("output_name", model_name).replace("/", "_")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = CLIPTextModelWithProjection.from_pretrained(model_name).to(device)
        model.eval()
        max_length = int(text_cfg.get("max_length", model.config.max_position_embeddings))
        return TextEncoder(kind, model_name, None, output_name, model, tokenizer, device, max_length)

    if kind == "open_clip":
        import open_clip

        arch = text_cfg.get("arch", "ViT-B-32")
        pretrained = text_cfg.get("pretrained", "laion2b_s34b_b79k")
        output_name = text_cfg.get("output_name", f"openclip_{arch}_{pretrained}").replace("/", "_")
        model = open_clip.create_model(arch, pretrained=pretrained, device=device)
        model.eval()
        tokenizer = open_clip.get_tokenizer(arch)
        return TextEncoder(kind, arch, pretrained, output_name, model, tokenizer, device)

    raise ValueError(f"Unsupported text encoder kind: {kind}")


def load_image_encoder(config: dict, device: torch.device) -> ImageEncoder:
    image_cfg = config.get("embeddings", {}).get("image_encoder", {})
    kind = str(image_cfg.get("kind", "dinov2")).lower()

    if kind in {"dinov2", "dinov3"}:
        from transformers import AutoImageProcessor, AutoModel

        default_model = {
            "dinov2": "facebook/dinov2-base",
            "dinov3": "facebook/dinov3-vits16-pretrain-lvd1689m",
        }[kind]
        model_name = image_cfg.get("model_name", default_model)
        output_name = image_cfg.get("output_name", model_name.replace("/", "_"))
        token = image_cfg.get("hf_token")
        processor = load_dino_processor(AutoImageProcessor, model_name, token)
        model = AutoModel.from_pretrained(model_name, token=token).to(device)
        model.eval()
        pooling = image_cfg.get("pooling", "cls")
        return ImageEncoder(kind, model_name, None, output_name, model, processor, device, pooling)

    if kind == "open_clip":
        import open_clip

        model_name = image_cfg.get("model_name", "ViT-B-32")
        pretrained = image_cfg.get("pretrained", "laion2b_s34b_b79k")
        output_name = image_cfg.get("output_name", f"openclip_{model_name}_{pretrained}").replace("/", "_")
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
        model.eval()
        return ImageEncoder(kind, model_name, pretrained, output_name, model, preprocess, device, "")

    raise ValueError(f"Unsupported image encoder kind: {kind}")


def prepare_item(config: dict, item: dict[str, Any], index: int) -> PreparedItem | None:
    source_cfg = config["source"]
    metadata_cfg = config.get("metadata", {})
    dataset = dataset_name(config)

    caption = item.get(source_cfg.get("caption_column", "caption"))
    image_uri = item.get(source_cfg.get("image_uri_column", source_cfg.get("image_url_column", "url")))
    if not caption or not image_uri:
        return None

    id_column = source_cfg.get("id_column")
    image_id = str(item.get(id_column) if id_column else item.get("id", index))
    file_name = str(item.get("file_name") or item.get("filename") or f"{image_id}.jpg")

    split_column = source_cfg.get("split_column")
    source_split = (
        normalize_split(item.get(split_column))
        if split_column and bool(metadata_cfg.get("use_source_split_if_available", True))
        else None
    )
    split = source_split or deterministic_split(
        image_id=image_id,
        seed=int(metadata_cfg.get("fallback_split_seed", 42)),
        train_ratio=float(metadata_cfg.get("fallback_split_ratios", {}).get("train", 1.0)),
        validation_ratio=float(metadata_cfg.get("fallback_split_ratios", {}).get("validation", 0.0)),
        test_ratio=float(metadata_cfg.get("fallback_split_ratios", {}).get("test", 0.0)),
    )

    return PreparedItem(
        metadata={
            "dataset_name": dataset,
            "image_id": image_id,
            "caption_id": f"{dataset}_{image_id}_0",
            "caption_index": 0,
            "caption": str(caption),
            "file_name": file_name,
            "image_uri": str(image_uri),
            "split": split,
            "source": source_cfg.get("hf_dataset", "hf_table"),
        }
    )


def encode_and_buffer(
    batch: list[PreparedItem],
    text_encoder: TextEncoder,
    image_encoder: ImageEncoder,
    writer: "StreamingShardWriter",
    normalize: bool,
) -> None:
    captions = [item.metadata["caption"] for item in batch]
    images = [item.image for item in batch]

    text_features = l2_normalize(text_encoder.encode(captions).float(), normalize).cpu().numpy()
    image_features = l2_normalize(image_encoder.encode(images).float(), normalize).cpu().numpy()

    for item, text_embedding, image_embedding in zip(batch, text_features, image_features):
        writer.add(
            item.metadata,
            {
                "caption_id": item.metadata["caption_id"],
                "image_id": item.metadata["image_id"],
                "split": item.metadata["split"],
                "caption_index": int(item.metadata["caption_index"]),
                "text_encoder_kind": text_encoder.kind,
                "text_encoder_name": text_encoder.name,
                "text_encoder_pretrained": text_encoder.pretrained,
                "embedding_dim": int(text_embedding.shape[0]),
                "embedding": text_embedding.astype("float32").tolist(),
            },
            {
                "image_id": item.metadata["image_id"],
                "split": item.metadata["split"],
                "file_name": item.metadata["file_name"],
                "image_uri": item.metadata["image_uri"],
                "image_encoder_kind": image_encoder.kind,
                "image_encoder_name": image_encoder.name,
                "image_encoder_pretrained": image_encoder.pretrained,
                "embedding_set": image_encoder.output_name,
                "embedding_dim": int(image_embedding.shape[0]),
                "embedding": image_embedding.astype("float32").tolist(),
            },
        )


@dataclass
class StreamingShardWriter:
    config: dict
    max_rows_per_file: int
    metadata_buffers: dict[str, list[dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))
    text_buffers: dict[str, list[dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))
    image_buffers: dict[str, list[dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))
    shard_indices: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    text_encoder: TextEncoder | None = None
    image_encoder: ImageEncoder | None = None

    def __post_init__(self) -> None:
        self.metadata_dir = metadata_dir(self.config)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.text_dir = encoder_output_dir(
            self.config,
            "text_embeddings",
            self.config["embeddings"]["text_encoder"].get(
                "output_name",
                self.config["embeddings"]["text_encoder"].get("model_name", "text_encoder"),
            ).replace("/", "_"),
        )
        self.image_dir = encoder_output_dir(
            self.config,
            "image_embeddings",
            self.config["embeddings"]["image_encoder"].get(
                "output_name",
                self.config["embeddings"]["image_encoder"].get("model_name", "image_encoder"),
            ).replace("/", "_"),
        )
        self.text_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)

    def add(self, metadata_row: dict[str, Any], text_row: dict[str, Any], image_row: dict[str, Any]) -> None:
        split = str(metadata_row["split"])
        self.metadata_buffers[split].append(metadata_row)
        self.text_buffers[split].append(text_row)
        self.image_buffers[split].append(image_row)
        self.counts[split] += 1

        if len(self.metadata_buffers[split]) >= self.max_rows_per_file:
            self.flush_split(split)

    def flush_split(self, split: str) -> None:
        if not self.metadata_buffers[split]:
            return

        shard_index = self.shard_indices[split]
        name = f"{split}-{shard_index:05d}.parquet"
        pd.DataFrame(self.metadata_buffers[split]).to_parquet(self.metadata_dir / name, index=False)
        pd.DataFrame(self.text_buffers[split]).to_parquet(self.text_dir / name, index=False)
        pd.DataFrame(self.image_buffers[split]).to_parquet(self.image_dir / name, index=False)

        self.metadata_buffers[split].clear()
        self.text_buffers[split].clear()
        self.image_buffers[split].clear()
        self.shard_indices[split] += 1

    def close(self, skipped: int) -> None:
        for split in list(self.metadata_buffers):
            self.flush_split(split)

        info = {
            "dataset_name": dataset_name(self.config),
            "num_rows": int(sum(self.counts.values())),
            "split_counts_rows": dict(sorted(self.counts.items())),
            "skipped_rows": int(skipped),
            "streaming_fused": True,
        }
        (self.metadata_dir / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

        write_manifest(
            self.text_dir,
            {
                "embedding_type": "text",
                "streaming_fused": True,
                "num_rows": int(sum(self.counts.values())),
            },
        )
        write_manifest(
            self.image_dir,
            {
                "embedding_type": "image",
                "streaming_fused": True,
                "num_rows": int(sum(self.counts.values())),
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate metadata, text embeddings, and image embeddings in one streaming pass.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--safe-exit",
        action="store_true",
        help="Exit with os._exit(0) after successful completion to avoid native-library shutdown crashes on some clusters.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    generate_streaming_features(args.config)
    safe_exit_if_requested(args.safe_exit or bool(config.get("runtime", {}).get("safe_exit", False)))


if __name__ == "__main__":
    main()
