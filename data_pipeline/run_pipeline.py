from __future__ import annotations

import argparse

from data_pipeline.build_metadata import build_metadata
from data_pipeline.generate_image_embeddings import generate_image_embeddings
from data_pipeline.generate_text_embeddings import generate_text_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TinyCLIP local data pipeline stages.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage",
        choices=["metadata", "text_embeddings", "image_embeddings", "all_local"],
        default="metadata",
        help="Pipeline stage to run. Embedding and publish stages can be added later.",
    )
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
