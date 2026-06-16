from __future__ import annotations

import argparse

from data_pipeline.build_metadata import build_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local Flickr30K metadata for TinyCLIP.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    build_metadata(args.config)


if __name__ == "__main__":
    main()

