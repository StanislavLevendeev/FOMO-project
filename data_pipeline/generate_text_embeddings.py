from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from data_pipeline.config import load_config, require_abs_path
from data_pipeline.embedding_utils import batches, l2_normalize, resolve_device, save_split_parquets, write_manifest

DEFAULT_TEXT_MODEL = "wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M"


def generate_text_embeddings(config_path: str | Path) -> Path:
    config = load_config(config_path)
    output_root = require_abs_path(config["local"]["output_root"], "local.output_root")

    metadata_path = output_root / "metadata" / "metadata.parquet"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    embeddings_cfg = config.get("embeddings", {})
    text_cfg = embeddings_cfg.get("text_encoder", {})
    kind = str(text_cfg.get("kind", "hf_clip_text")).lower()
    if kind not in {"hf_clip_text", "tiny_clip"}:
        raise ValueError("Only embeddings.text_encoder.kind=hf_clip_text is implemented for now")

    batch_size = int(embeddings_cfg.get("batch_size_text", 128))
    normalize = bool(embeddings_cfg.get("normalize", True))
    device = resolve_device(embeddings_cfg.get("device", "auto"))

    model_name = text_cfg.get("model_name", DEFAULT_TEXT_MODEL)
    output_name = text_cfg.get("output_name", model_name).replace("/", "_")

    from transformers import AutoTokenizer, CLIPTextModelWithProjection

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = CLIPTextModelWithProjection.from_pretrained(model_name)
    max_length = int(text_cfg.get("max_length", model.config.max_position_embeddings))
    model = model.to(device)
    model.eval()

    metadata = pd.read_parquet(metadata_path)
    metadata = metadata.sort_values("caption_id").reset_index(drop=True)

    rows = []
    with torch.inference_mode():
        for batch_df in tqdm(list(batches(list(metadata.index), batch_size)), desc="Encoding text"):
            chunk = metadata.loc[list(batch_df)]
            tokens = tokenizer(
                chunk["caption"].astype(str).tolist(),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            features = model(**tokens).text_embeds
            features = l2_normalize(features.float(), normalize).cpu().numpy()

            for (_, row), embedding in zip(chunk.iterrows(), features):
                rows.append(
                    {
                        "caption_id": row["caption_id"],
                        "image_id": row["image_id"],
                        "split": row["split"],
                        "caption_index": int(row["caption_index"]),
                        "text_encoder_kind": kind,
                        "text_encoder_name": model_name,
                        "text_encoder_pretrained": None,
                        "embedding_dim": int(embedding.shape[0]),
                        "embedding": embedding.astype("float32").tolist(),
                    }
                )

    output_dir = output_root / "text_embeddings" / output_name
    embeddings = pd.DataFrame(rows)
    save_split_parquets(embeddings, output_dir)
    write_manifest(
        output_dir,
        {
            "embedding_type": "text",
            "metadata_path": str(metadata_path),
            "encoder_kind": kind,
            "model_name": model_name,
            "pretrained": None,
            "output_name": output_name,
            "normalized": normalize,
            "num_rows": int(len(embeddings)),
            "num_captions": int(embeddings["caption_id"].nunique()),
            "embedding_dim": int(embeddings["embedding_dim"].iloc[0]) if len(embeddings) else None,
        },
    )

    print(f"Saved text embeddings: {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate frozen text embeddings from metadata.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    generate_text_embeddings(args.config)


if __name__ == "__main__":
    main()
