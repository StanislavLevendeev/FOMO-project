from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from huggingface_hub import HfApi

from data_pipeline.config import load_config
from data_pipeline.paths import dataset_name, dataset_root, output_root


def publish_hf_dataset(
    config_path: str | Path,
    repo_id: str | None = None,
    include: str = "all",
) -> None:
    config = load_config(config_path)
    publish_cfg = config.get("publish", {})
    repo_id = repo_id or publish_cfg.get("hf_repo_id")
    if not repo_id or repo_id == "your-username-or-org/tinyclip-flickr30k-features":
        raise ValueError("Pass --repo-id or set publish.hf_repo_id in your local config")

    private = bool(publish_cfg.get("private", True))
    name = dataset_name(config)
    root = dataset_root(config)
    if not root.exists():
        raise FileNotFoundError(f"Dataset output folder not found: {root}")

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

    manifests_dir = write_shared_manifests(config)
    upload_dataset_tree(api, repo_id, root, name, include)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(manifests_dir),
        path_in_repo="manifests",
        commit_message=f"Upload manifests for {name}",
    )

    print(f"Published dataset folder: {name}")


def upload_dataset_tree(api: HfApi, repo_id: str, root: Path, name: str, include: str) -> None:
    if include == "all":
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(root),
            path_in_repo=name,
            ignore_patterns=["**/metadata.csv"],
            commit_message=f"Upload {name}",
        )
        return

    folder_by_include = {
        "metadata": root / "metadata",
        "text": root / "text_embeddings",
        "image": root / "image_embeddings",
    }
    folder = folder_by_include[include]
    if not folder.exists():
        print(f"Skipping missing folder: {folder}")
        return

    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(folder),
        path_in_repo=f"{name}/{folder.name}",
        ignore_patterns=["metadata.csv"],
        commit_message=f"Upload {name}/{folder.name}",
    )


def write_shared_manifests(config: dict) -> Path:
    root = output_root(config)
    manifests_dir = root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    dataset_entries = []
    text_encoders = []
    image_encoders = []
    for child in sorted(path for path in root.iterdir() if path.is_dir() and path.name != "manifests"):
        metadata_path = child / "metadata"
        if not metadata_path.exists():
            continue
        dataset_entries.append(
            {
                "name": child.name,
                "metadata": f"{child.name}/metadata",
                "dataset_info": f"{child.name}/metadata/dataset_info.json",
            }
        )
        text_encoders.extend(discover_encoder_dirs(child / "text_embeddings", child.name, "text_embeddings"))
        image_encoders.extend(discover_encoder_dirs(child / "image_embeddings", child.name, "image_embeddings"))

    datasets_manifest = {"datasets": dataset_entries}

    encoders_manifest = {
        "encoders": {
            "text": text_encoders,
            "image": image_encoders,
        }
    }

    (manifests_dir / "datasets.yaml").write_text(yaml.safe_dump(datasets_manifest, sort_keys=False), encoding="utf-8")
    (manifests_dir / "encoders.yaml").write_text(yaml.safe_dump(encoders_manifest, sort_keys=False), encoding="utf-8")
    return manifests_dir


def discover_encoder_dirs(root: Path, dataset: str, group: str) -> list[dict[str, str]]:
    if not root.exists():
        return []
    return [
        {
            "dataset": dataset,
            "name": path.name,
            "path": f"{dataset}/{group}/{path.name}",
            "manifest": f"{dataset}/{group}/{path.name}/manifest.json",
        }
        for path in sorted(root.iterdir())
        if path.is_dir()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish TinyCLIP feature files to a HF Dataset repo.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo-id", default=None, help="Example: your-org/tinyclip-features")
    parser.add_argument("--include", choices=["all", "metadata", "text", "image"], default="all")
    args = parser.parse_args()

    publish_hf_dataset(config_path=args.config, repo_id=args.repo_id, include=args.include)


if __name__ == "__main__":
    main()
