from __future__ import annotations

import argparse

from data_pipeline.config import load_config
from data_pipeline.publish_hf_dataset import publish_hf_dataset
from data_pipeline.build_metadata import build_metadata
from data_pipeline.generate_image_embeddings import generate_image_embeddings
from data_pipeline.generate_text_embeddings import generate_text_embeddings
from data_pipeline.runtime import safe_exit_if_requested


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TinyCLIP local data pipeline stages.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage",
        choices=["metadata", "text_embeddings", "image_embeddings", "all_local", "all"],
        default="metadata",
        help="Pipeline stage to run. Embedding and publish stages can be added later.",
    )
    parser.add_argument("--repo-id", default=None)
    parser.add_argument(
        "--safe-exit",
        action="store_true",
        help="Exit with os._exit(0) after successful completion to avoid native-library shutdown crashes on some clusters.",
    )
    args = parser.parse_args()
    config = load_config(args.config)

    if args.stage == "metadata":
        build_metadata(args.config)
    elif args.stage == "text_embeddings":
        generate_text_embeddings(args.config)
    elif args.stage == "image_embeddings":
        generate_image_embeddings(args.config)
    elif args.stage == "all_local":
        build_metadata(args.config)
        generate_text_embeddings(args.config)
        generate_image_embeddings(args.config)
    elif args.stage == "all":
        build_metadata(args.config)
        generate_text_embeddings(args.config)
        generate_image_embeddings(args.config)
        publish_hf_dataset(args.config, repo_id=args.repo_id)

    safe_exit_if_requested(args.safe_exit or bool(config.get("runtime", {}).get("safe_exit", False)))


if __name__ == "__main__":
    main()
