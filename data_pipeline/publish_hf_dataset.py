from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, upload_file

from data_pipeline.config import load_config, require_abs_path


SPLITS = ("train", "validation", "test")


def publish_hf_dataset(
    config_path: str | Path,
    repo_id: str | None = None,
    include: str = "all",
) -> None:
    config = load_config(config_path)
    output_root = require_abs_path(config["local"]["output_root"], "local.output_root")

    publish_cfg = config.get("publish", {})
    repo_id = repo_id or publish_cfg.get("hf_repo_id")
    if not repo_id or repo_id == "your-username-or-org/tinyclip-flickr30k-features":
        raise ValueError("Pass --repo-id or set publish.hf_repo_id in your local config")

    private = bool(publish_cfg.get("private", True))
    max_shard_size = publish_cfg.get("max_shard_size", "500MB")

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

    if include in {"all", "metadata"}:
        publish_metadata(output_root, repo_id, max_shard_size)

    if include in {"all", "text"}:
        publish_embedding_group(output_root / "text_embeddings", repo_id, "text_embeddings", max_shard_size)

    if include in {"all", "image"}:
        publish_embedding_group(output_root / "image_embeddings", repo_id, "image_embeddings", max_shard_size)


def publish_metadata(output_root: Path, repo_id: str, max_shard_size: str) -> None:
    metadata_path = output_root / "metadata" / "metadata.parquet"
    info_path = output_root / "metadata" / "dataset_info.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")

    df = pd.read_parquet(metadata_path)
    dataset = frame_to_dataset_dict(df)
    dataset.push_to_hub(
        repo_id,
        config_name="metadata",
        max_shard_size=max_shard_size,
        commit_message="Upload TinyCLIP metadata",
    )

    if info_path.exists():
        upload_file(
            path_or_fileobj=str(info_path),
            path_in_repo="manifests/metadata/dataset_info.json",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Upload metadata manifest",
        )

    print("Published config: metadata")


def publish_embedding_group(group_dir: Path, repo_id: str, group_name: str, max_shard_size: str) -> None:
    if not group_dir.exists():
        print(f"Skipping missing folder: {group_dir}")
        return

    for embedding_dir in sorted(path for path in group_dir.iterdir() if path.is_dir()):
        split_paths = {split: embedding_dir / f"{split}.parquet" for split in SPLITS}
        existing = {split: path for split, path in split_paths.items() if path.exists()}
        if not existing:
            continue

        dataset = parquet_files_to_dataset_dict(existing)
        config_name = safe_config_name(f"{group_name}__{embedding_dir.name}")
        dataset.push_to_hub(
            repo_id,
            config_name=config_name,
            max_shard_size=max_shard_size,
            commit_message=f"Upload {config_name}",
        )

        manifest_path = embedding_dir / "manifest.json"
        if manifest_path.exists():
            upload_file(
                path_or_fileobj=str(manifest_path),
                path_in_repo=f"manifests/{group_name}/{embedding_dir.name}/manifest.json",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Upload manifest for {config_name}",
            )

        print(f"Published config: {config_name}")


def frame_to_dataset_dict(df: pd.DataFrame) -> DatasetDict:
    split_map = {}
    for split in SPLITS:
        split_df = df[df["split"] == split].reset_index(drop=True)
        if len(split_df):
            split_map[split] = Dataset.from_pandas(split_df, preserve_index=False)
    if not split_map:
        raise ValueError("No train/validation/test rows found")
    return DatasetDict(split_map)


def parquet_files_to_dataset_dict(paths_by_split: dict[str, Path]) -> DatasetDict:
    split_map = {}
    for split, path in paths_by_split.items():
        df = pd.read_parquet(path)
        split_map[split] = Dataset.from_pandas(df, preserve_index=False)
    return DatasetDict(split_map)


def safe_config_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_").lower()


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish TinyCLIP metadata and embeddings to a HF Dataset repo.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo-id", default=None, help="Example: your-org/tinyclip-flickr30k-features")
    parser.add_argument("--include", choices=["all", "metadata", "text", "image"], default="all")
    args = parser.parse_args()

    publish_hf_dataset(config_path=args.config, repo_id=args.repo_id, include=args.include)


if __name__ == "__main__":
    main()
