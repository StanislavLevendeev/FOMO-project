from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from data_pipeline.config import load_config, require_abs_path
from data_pipeline.embedding_utils import batches, l2_normalize, resolve_device, save_split_parquets, write_manifest


def generate_image_embeddings(config_path: str | Path) -> Path:
    config = load_config(config_path)
    raw_root = require_abs_path(config["local"]["raw_root"], "local.raw_root")
    output_root = require_abs_path(config["local"]["output_root"], "local.output_root")

    metadata_path = output_root / "metadata" / "metadata.parquet"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    embeddings_cfg = config.get("embeddings", {})
    image_cfg = embeddings_cfg.get("image_encoder", {})
    kind = image_cfg.get("kind", "dinov2")

    if kind == "dinov2":
        return generate_dinov2_embeddings(config, metadata_path, raw_root, output_root)
    if kind == "open_clip":
        return generate_openclip_image_embeddings(config, metadata_path, raw_root, output_root)

    raise ValueError(f"Unsupported embeddings.image_encoder.kind: {kind}")


def generate_dinov2_embeddings(
    config: dict,
    metadata_path: Path,
    raw_root: Path,
    output_root: Path,
) -> Path:
    from transformers import AutoImageProcessor, AutoModel

    embeddings_cfg = config.get("embeddings", {})
    image_cfg = embeddings_cfg.get("image_encoder", {})

    batch_size = int(embeddings_cfg.get("batch_size_image", 32))
    normalize = bool(embeddings_cfg.get("normalize", True))
    device = resolve_device(embeddings_cfg.get("device", "auto"))

    model_name = image_cfg.get("model_name", "facebook/dinov2-base")
    output_name = image_cfg.get("output_name", model_name.replace("/", "_"))
    pooling = image_cfg.get("pooling", "cls")

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    images_df = unique_images(metadata_path)
    rows = []

    with torch.inference_mode():
        for batch_df in tqdm(list(batches(list(images_df.index), batch_size)), desc="Encoding DINOv2 images"):
            chunk = images_df.loc[list(batch_df)]
            pil_images = [load_rgb_image(resolve_local_image_path(row, raw_root)) for _, row in chunk.iterrows()]
            inputs = processor(images=pil_images, return_tensors="pt").to(device)
            outputs = model(**inputs)

            if pooling == "cls":
                features = outputs.last_hidden_state[:, 0]
            elif pooling == "mean_patch":
                features = outputs.last_hidden_state[:, 1:].mean(dim=1)
            else:
                raise ValueError(f"Unsupported DINOv2 pooling: {pooling}")

            features = l2_normalize(features.float(), normalize).cpu().numpy()

            for (_, row), embedding in zip(chunk.iterrows(), features):
                rows.append(image_embedding_row(row, embedding, "dinov2", model_name, None, output_name))

    output_dir = output_root / "image_embeddings" / output_name
    embeddings = pd.DataFrame(rows)
    save_split_parquets(embeddings, output_dir)
    write_manifest(
        output_dir,
        image_manifest(
            metadata_path=metadata_path,
            embeddings=embeddings,
            encoder_kind="dinov2",
            model_name=model_name,
            pretrained=None,
            output_name=output_name,
            normalized=normalize,
            extra={"pooling": pooling},
        ),
    )
    print(f"Saved image embeddings: {output_dir}")
    return output_dir


def generate_openclip_image_embeddings(
    config: dict,
    metadata_path: Path,
    raw_root: Path,
    output_root: Path,
) -> Path:
    import open_clip

    embeddings_cfg = config.get("embeddings", {})
    image_cfg = embeddings_cfg.get("image_encoder", {})

    batch_size = int(embeddings_cfg.get("batch_size_image", 32))
    normalize = bool(embeddings_cfg.get("normalize", True))
    device = resolve_device(embeddings_cfg.get("device", "auto"))

    model_name = image_cfg.get("model_name", "ViT-B-32")
    pretrained = image_cfg.get("pretrained", "laion2b_s34b_b79k")
    output_name = image_cfg.get("output_name", f"openclip_{model_name}_{pretrained}").replace("/", "_")

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    model.eval()

    images_df = unique_images(metadata_path)
    rows = []

    with torch.inference_mode():
        for batch_df in tqdm(list(batches(list(images_df.index), batch_size)), desc="Encoding OpenCLIP images"):
            chunk = images_df.loc[list(batch_df)]
            image_tensors = [preprocess(load_rgb_image(resolve_local_image_path(row, raw_root))) for _, row in chunk.iterrows()]
            image_batch = torch.stack(image_tensors).to(device)
            features = model.encode_image(image_batch)
            features = l2_normalize(features.float(), normalize).cpu().numpy()

            for (_, row), embedding in zip(chunk.iterrows(), features):
                rows.append(image_embedding_row(row, embedding, "open_clip", model_name, pretrained, output_name))

    output_dir = output_root / "image_embeddings" / output_name
    embeddings = pd.DataFrame(rows)
    save_split_parquets(embeddings, output_dir)
    write_manifest(
        output_dir,
        image_manifest(
            metadata_path=metadata_path,
            embeddings=embeddings,
            encoder_kind="open_clip",
            model_name=model_name,
            pretrained=pretrained,
            output_name=output_name,
            normalized=normalize,
            extra={},
        ),
    )
    print(f"Saved image embeddings: {output_dir}")
    return output_dir


def unique_images(metadata_path: Path) -> pd.DataFrame:
    metadata = pd.read_parquet(metadata_path)
    if "image_uri" not in metadata.columns and "relative_image_path" in metadata.columns:
        metadata["image_uri"] = metadata["relative_image_path"]
    columns = ["image_id", "split", "file_name", "image_uri"]
    return metadata[columns].drop_duplicates("image_id").sort_values("image_id").reset_index(drop=True)


def resolve_local_image_path(row: pd.Series, raw_root: Path) -> Path:
    image_uri = str(row.get("image_uri", ""))
    if image_uri and not image_uri.startswith(("http://", "https://", "zip://")):
        return raw_root / image_uri
    return raw_root / "images" / str(row["file_name"])


def load_rgb_image(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Missing local image: {path}")
    return Image.open(path).convert("RGB")


def image_embedding_row(
    row: pd.Series,
    embedding,
    encoder_kind: str,
    model_name: str,
    pretrained: str | None,
    output_name: str,
) -> dict:
    row_data = {
        "image_id": row["image_id"],
        "split": row["split"],
        "file_name": row["file_name"],
        "image_uri": row["image_uri"],
        "image_encoder_kind": encoder_kind,
        "image_encoder_name": model_name,
        "image_encoder_pretrained": pretrained,
        "embedding_set": output_name,
        "embedding_dim": int(embedding.shape[0]),
        "embedding": embedding.astype("float32").tolist(),
    }
    return row_data


def image_manifest(
    metadata_path: Path,
    embeddings: pd.DataFrame,
    encoder_kind: str,
    model_name: str,
    pretrained: str | None,
    output_name: str,
    normalized: bool,
    extra: dict,
) -> dict:
    return {
        "embedding_type": "image",
        "metadata_path": str(metadata_path),
        "encoder_kind": encoder_kind,
        "model_name": model_name,
        "pretrained": pretrained,
        "output_name": output_name,
        "normalized": normalized,
        "num_rows": int(len(embeddings)),
        "num_images": int(embeddings["image_id"].nunique()),
        "embedding_dim": int(embeddings["embedding_dim"].iloc[0]) if len(embeddings) else None,
        **extra,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate frozen image embeddings from local images.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    generate_image_embeddings(args.config)


if __name__ == "__main__":
    main()

