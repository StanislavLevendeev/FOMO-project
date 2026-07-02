from __future__ import annotations

import os
# Fix for Windows path too long errors when downloading from Hugging Face Hub
os.environ["HF_HUB_CACHE"] = "C:/hf_cache/hub"
os.environ["HF_DATASETS_CACHE"] = "C:/hf_cache/datasets"

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset


@dataclass
class TinyCLIPFeatureStore:
    repo_id: str
    split: str
    cache_dir: Path | None
    image_ids: list[str]
    caption_ids: list[str]
    image_embeddings: torch.Tensor
    text_embeddings: torch.Tensor
    captions: list[str] | None
    image_id_to_image_idx: dict[str, int]
    image_id_to_text_indices: dict[str, list[int]]
    caption_id_to_text_idx: dict[str, int]
    metadata_ds: Any | None = None
    image_id_to_captions: dict[str, list[str]] | None = None

    @classmethod
    def from_hub(
        cls,
        split: str,
        image_config: str,
        text_config: str,
        cache_dir: str | Path | None = None,
    ) -> "TinyCLIPFeatureStore":
        cache_dir_str = str(cache_dir) if cache_dir else None
        repo_id = "StanislavLev/tiny-clip-image-encoders-adapter"
        image_ds = load_dataset(
            repo_id,
            image_config,
            split=split,
            cache_dir=cache_dir_str,
        )

        text_ds = load_dataset(
            repo_id,
            text_config,
            split=split,
            cache_dir=cache_dir_str,
        )

        image_ids = list(image_ds["image_id"])
        caption_ids = list(text_ds["caption_id"])

        image_embeddings = torch.tensor(
            image_ds["embedding"], dtype=torch.float32
        )
        text_embeddings = torch.tensor(
            text_ds["embedding"], dtype=torch.float32
        )

        captions = (
            list(text_ds["caption"])
            if "caption" in text_ds.column_names
            else None
        )

        image_id_to_image_idx = {
            image_id: idx for idx, image_id in enumerate(image_ids)
        }

        image_id_to_text_indices: dict[str, list[int]] = {}
        for idx, image_id in enumerate(text_ds["image_id"]):
            image_id_to_text_indices.setdefault(image_id, []).append(idx)

        caption_id_to_text_idx = {
            caption_id: idx for idx, caption_id in enumerate(caption_ids)
        }

        return cls(
            image_ids=image_ids,
            caption_ids=caption_ids,
            image_embeddings=image_embeddings,
            text_embeddings=text_embeddings,
            captions=captions,
            image_id_to_image_idx=image_id_to_image_idx,
            image_id_to_text_indices=image_id_to_text_indices,
            caption_id_to_text_idx=caption_id_to_text_idx,
            repo_id=repo_id,
            split=split,
            cache_dir=Path(cache_dir) if cache_dir else None,
        )

    def get_image_embedding(self, image_id: str) -> torch.Tensor:
        return self.image_embeddings[self.image_id_to_image_idx[image_id]]

    def get_text_embeddings_for_image(self, image_id: str) -> torch.Tensor:
        indices = self.image_id_to_text_indices[image_id]
        return self.text_embeddings[indices]

    def _ensure_metadata_loaded(self):
        if self.metadata_ds is not None:
            return

        self.metadata_ds = load_dataset(
            self.repo_id,
            "metadata",
            split=self.split,
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
        )

        self.image_id_to_captions = {}
        for row in self.metadata_ds:
            self.image_id_to_captions.setdefault(row["image_id"], []).append(
                row["caption"]
            )

    def get_captions(self, image_id: str) -> list[str]:
        self._ensure_metadata_loaded()
        return self.image_id_to_captions.get(image_id, [])

class LAIONFeatureStore:
    DEFAULT_IMAGE_EMBEDDING_FOLDER = "dinov3_vits16_pretrain_lvd1689m"
    DEFAULT_TEXT_EMBEDDING_FOLDER = "tinyclip_vit_39m_16_text_19m_yfcc15m"

    def __init__(
        self,
        repo_id: str,
        split: str,
        cache_dir: Path | None,
        image_ds: Any,
        text_ds: Any,
        metadata_ds: Any,
        alignment_report: dict[str, Any],
        image_embedding_folder: str,
        text_embedding_folder: str,
    ):
        self.repo_id = repo_id
        self.split = split
        self.cache_dir = cache_dir
        self.image_embedding_folder = image_embedding_folder
        self.text_embedding_folder = text_embedding_folder
        self.alignment_report = alignment_report

        self.image_ds = image_ds
        self.text_ds = text_ds
        self.metadata_ds = metadata_ds

        print("Loading aligned embeddings into memory...")
        image_embeddings_np = image_ds.with_format("numpy", columns=["embedding"])[:]["embedding"]
        text_embeddings_np = text_ds.with_format("numpy", columns=["embedding"])[:]["embedding"]

        self.image_embeddings = torch.from_numpy(image_embeddings_np)
        self.text_embeddings = torch.from_numpy(text_embeddings_np)
        print(f"Embeddings loaded with metadata alignment: {alignment_report}")

    @staticmethod
    def _select_files(all_files: list[str], folder: str) -> list[str]:
        files = sorted([f for f in all_files if f.startswith(folder) and f.endswith(".parquet")])
        if not files:
            raise FileNotFoundError(f"No parquet files found under {folder!r}")
        return files

    @staticmethod
    def _alignment_order(source_keys: list[Any], target_keys: list[Any], label: str) -> tuple[list[int] | None, dict[str, Any]]:
        if len(source_keys) != len(target_keys):
            raise ValueError(
                f"{label}: source/target length mismatch: "
                f"{len(source_keys)} vs {len(target_keys)}"
            )

        first_mismatch = next(
            (idx for idx, (source, target) in enumerate(zip(source_keys, target_keys)) if source != target),
            None,
        )
        if first_mismatch is None:
            return None, {
                "label": label,
                "rows": len(source_keys),
                "reordered": False,
                "first_mismatch": None,
            }

        positions: dict[Any, deque[int]] = defaultdict(deque)
        for idx, key in enumerate(source_keys):
            positions[key].append(idx)

        order: list[int] = []
        missing: list[Any] = []
        for key in target_keys:
            if positions[key]:
                order.append(positions[key].popleft())
            else:
                missing.append(key)
                if len(missing) >= 10:
                    break

        unused = sum(len(values) for values in positions.values())
        if missing or unused:
            raise ValueError(
                f"{label}: cannot align to metadata order; "
                f"missing examples={missing[:5]}, unused source rows={unused}"
            )

        return order, {
            "label": label,
            "rows": len(source_keys),
            "reordered": True,
            "first_mismatch": int(first_mismatch),
        }

    @classmethod
    def from_hub(
        cls,
        repo_id: str = "StanislavLev/tiny-clip-image-encoders-adapter",
        cache_dir: str | Path | None = None,
        image_embedding_folder: str | None = None,
        text_embedding_folder: str | None = None,
    ) -> "LAIONFeatureStore":
        cache_dir_str = str(cache_dir) if cache_dir else None
        image_embedding_folder = image_embedding_folder or cls.DEFAULT_IMAGE_EMBEDDING_FOLDER
        text_embedding_folder = text_embedding_folder or cls.DEFAULT_TEXT_EMBEDDING_FOLDER

        print(f"Listing repo files for {repo_id}...")
        from huggingface_hub import list_repo_files
        all_files = list_repo_files(repo_id, repo_type="dataset")

        image_files = cls._select_files(
            all_files, f"laion1m/image_embeddings/{image_embedding_folder}/"
        )
        text_files = cls._select_files(
            all_files, f"laion1m/text_embeddings/{text_embedding_folder}/"
        )
        metadata_files = cls._select_files(all_files, "laion1m/metadata/")

        print(
            f"Loading LAION 1M with image={image_embedding_folder} "
            f"({len(image_files)} files), text={text_embedding_folder} "
            f"({len(text_files)} files), metadata ({len(metadata_files)} files)."
        )
        image_ds = load_dataset(
            repo_id,
            data_files=image_files,
            split="train",
            cache_dir=cache_dir_str,
        )
        text_ds = load_dataset(
            repo_id,
            data_files=text_files,
            split="train",
            cache_dir=cache_dir_str,
        )
        metadata_ds = load_dataset(
            repo_id,
            data_files=metadata_files,
            split="train",
            cache_dir=cache_dir_str,
        )

        image_order, image_report = cls._alignment_order(
            list(image_ds["image_uri"]),
            list(metadata_ds["image_uri"]),
            f"laion1m/{image_embedding_folder}/image",
        )
        if image_order is not None:
            image_ds = image_ds.select(image_order)

        text_order, text_report = cls._alignment_order(
            list(text_ds["caption_id"]),
            list(metadata_ds["caption_id"]),
            f"laion1m/{text_embedding_folder}/text",
        )
        if text_order is not None:
            text_ds = text_ds.select(text_order)

        return cls(
            repo_id=repo_id,
            split="train",
            cache_dir=Path(cache_dir) if cache_dir else None,
            image_ds=image_ds,
            text_ds=text_ds,
            metadata_ds=metadata_ds,
            alignment_report={
                "metadata_rows": len(metadata_ds),
                "image": image_report,
                "text": text_report,
            },
            image_embedding_folder=image_embedding_folder,
            text_embedding_folder=text_embedding_folder,
        )
