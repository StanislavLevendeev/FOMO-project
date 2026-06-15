from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def create_local_config(
    output_path: str | Path,
    raw_root: str | Path,
    output_root: str | Path,
    hf_dataset: str = "nlphuji/flickr30k",
) -> Path:
    output_path = Path(output_path).expanduser()
    raw_root = Path(raw_root).expanduser()
    output_root = Path(output_root).expanduser()

    if not raw_root.is_absolute():
        raise ValueError(f"--raw-root must be absolute: {raw_root}")
    if not output_root.is_absolute():
        raise ValueError(f"--output-root must be absolute: {output_root}")

    config = {
        "dataset_name": "flickr30k",
        "source": {
            "kind": "hf_flickr30k",
            "hf_dataset": hf_dataset,
            "streaming": True,
        },
        "local": {
            "raw_root": str(raw_root),
            "output_root": str(output_root),
        },
        "metadata": {
            "save_raw_images": True,
            "use_source_split_if_available": True,
            "fallback_split_seed": 42,
            "fallback_split_ratios": {
                "train": 0.90,
                "validation": 0.05,
                "test": 0.05,
            },
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a local Flickr30K pipeline config.")
    parser.add_argument("--output", default="configs/flickr30k.local.yaml")
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--hf-dataset", default="nlphuji/flickr30k")
    args = parser.parse_args()

    path = create_local_config(
        output_path=args.output,
        raw_root=args.raw_root,
        output_root=args.output_root,
        hf_dataset=args.hf_dataset,
    )
    print(f"Created local config: {path}")


if __name__ == "__main__":
    main()

