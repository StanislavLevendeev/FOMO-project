from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from data_pipeline.config import load_config, require_abs_path
from data_pipeline.sources.hf_flickr30k import build_flickr30k_metadata
from data_pipeline.validate_metadata import validate_image_paths, validate_metadata_frame


def build_metadata(config_path: str | Path) -> Path:
    config = load_config(config_path)

    raw_root = require_abs_path(config["local"]["raw_root"], "local.raw_root")
    output_root = require_abs_path(config["local"]["output_root"], "local.output_root")
    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    source_kind = config["source"]["kind"]
    if source_kind == "hf_flickr30k":
        rows = build_flickr30k_metadata(config=config, raw_root=raw_root)
    else:
        raise ValueError(f"Unsupported source.kind: {source_kind}")
    
    df = pd.DataFrame([row.to_dict() for row in rows])

    validate_metadata_frame(df)

    if config.get("metadata", {}).get("save_raw_images", True):
        validate_image_paths(df, raw_root)

    parquet_path = metadata_dir / "metadata.parquet"
    csv_path = metadata_dir / "metadata.csv"
    info_path = metadata_dir / "dataset_info.json"

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    info = {
        "dataset_name": config.get("dataset_name", "flickr30k"),
        "num_rows": int(len(df)),
        "num_images": int(df["image_id"].nunique()),
        "num_captions": int(df["caption_id"].nunique()),
        "split_counts_rows": df["split"].value_counts().sort_index().to_dict(),
        "split_counts_images": df.drop_duplicates("image_id")["split"].value_counts().sort_index().to_dict(),
        "raw_images_included_in_shared_repo": False,
    }
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

    print(f"Saved metadata: {parquet_path}")
    print(f"Saved metadata backup: {csv_path}")
    print(f"Saved dataset info: {info_path}")

    return parquet_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    build_metadata(args.config)


if __name__ == "__main__":
    main()

