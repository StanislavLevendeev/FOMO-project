from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from data_pipeline.config import load_config, require_abs_path
from data_pipeline.embedding_utils import save_split_parquets
from data_pipeline.paths import dataset_name, dataset_root, metadata_dir
from data_pipeline.sources.hf_flickr30k import build_flickr30k_metadata
from data_pipeline.sources.hf_imagenet1k import build_imagenet1k_metadata
from data_pipeline.sources.hf_table import build_hf_table_metadata
from data_pipeline.validate_metadata import validate_image_paths, validate_metadata_frame


def build_metadata(config_path: str | Path) -> Path:
    config = load_config(config_path)

    raw_root_value = config.get("local", {}).get("raw_root")
    raw_root = require_abs_path(raw_root_value, "local.raw_root") if raw_root_value else dataset_root(config)
    output_dir = metadata_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_kind = config["source"]["kind"]
    if source_kind == "hf_flickr30k":
        rows = build_flickr30k_metadata(config=config, raw_root=raw_root)
    elif source_kind == "hf_imagenet1k":
        rows = build_imagenet1k_metadata(config=config, raw_root=raw_root)
    elif source_kind == "hf_table":
        rows = build_hf_table_metadata(config=config, raw_root=raw_root)
    else:
        raise ValueError(f"Unsupported source.kind: {source_kind}")
    
    df = pd.DataFrame([row.to_dict() for row in rows])

    validate_metadata_frame(df)

    if config.get("metadata", {}).get("save_raw_images", True):
        validate_image_paths(df, raw_root)

    shard_rows = config.get("metadata", {}).get("max_rows_per_file")
    saved_paths = save_split_parquets(
        df,
        output_dir,
        max_rows_per_file=int(shard_rows) if shard_rows else None,
    )
    csv_path = output_dir / "metadata.csv"
    info_path = output_dir / "dataset_info.json"

    df.to_csv(csv_path, index=False)

    info = {
        "dataset_name": dataset_name(config),
        "num_rows": int(len(df)),
        "num_images": int(df["image_id"].nunique()),
        "num_captions": int(df["caption_id"].nunique()),
        "split_counts_rows": df["split"].value_counts().sort_index().to_dict(),
        "split_counts_images": df.drop_duplicates("image_id")["split"].value_counts().sort_index().to_dict(),
        "raw_images_included_in_shared_repo": False,
        "files": [str(path) for path in saved_paths],
    }
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

    print(f"Saved metadata: {output_dir}")
    print(f"Saved metadata backup: {csv_path}")
    print(f"Saved dataset info: {info_path}")

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    build_metadata(args.config)


if __name__ == "__main__":
    main()

