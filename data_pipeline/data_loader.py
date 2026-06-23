from __future__ import annotations

import os
# Fix for Windows path too long errors when downloading from Hugging Face Hub
os.environ["HF_HUB_CACHE"] = "C:/hf_cache/hub"
os.environ["HF_DATASETS_CACHE"] = "C:/hf_cache/datasets"

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
    def __init__(
        self,
        repo_id: str,
        split: str,
        cache_dir: Path | None,
        image_ds: Any,
        text_ds: Any,
    ):
        self.repo_id = repo_id
        self.split = split
        self.cache_dir = cache_dir
        
        self.image_ds = image_ds
        self.text_ds = text_ds
        
        print("Loading embeddings into memory...")
        # Extract only the "embedding" column to avoid allocating GBs of memory for string columns
        image_embeddings_np = image_ds.with_format("numpy", columns=["embedding"])[:]["embedding"]
        text_embeddings_np = text_ds.with_format("numpy", columns=["embedding"])[:]["embedding"]
        
        # Convert to contiguous PyTorch tensors (zero-copy from numpy)
        self.image_embeddings = torch.from_numpy(image_embeddings_np)
        self.text_embeddings = torch.from_numpy(text_embeddings_np)
        print("Embeddings loaded. Sequential 1-to-1 matching is ready.")

    @classmethod
    def from_hub(
        cls,
        repo_id: str = "StanislavLev/tiny-clip-image-encoders-adapter",
        cache_dir: str | Path | None = None,
    ) -> "LAIONFeatureStore":
        cache_dir_str = str(cache_dir) if cache_dir else None
        
        print(f"Listing repo files for {repo_id}...")
        from huggingface_hub import list_repo_files
        all_files = list_repo_files(repo_id, repo_type="dataset")
        
        # Sort files alphabetically to ensure aligned 1-to-1 order across images and texts
        image_files = sorted([
            f for f in all_files 
            if f.startswith("laion1m/image_embeddings/") and f.endswith(".parquet")
        ])
        text_files = sorted([
            f for f in all_files 
            if f.startswith("laion1m/text_embeddings/") and f.endswith(".parquet")
        ])
        
        print(f"Loading {len(image_files)} image files and {len(text_files)} text files from LAION 1M dataset...")
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

        return cls(
            repo_id=repo_id,
            split="train",
            cache_dir=Path(cache_dir) if cache_dir else None,
            image_ds=image_ds,
            text_ds=text_ds,
        )
