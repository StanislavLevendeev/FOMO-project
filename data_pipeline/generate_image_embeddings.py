from __future__ import annotations

import argparse
from io import BytesIO
from math import ceil
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from data_pipeline.config import load_config, require_abs_path
from data_pipeline.embedding_utils import batches, l2_normalize, resolve_device, save_split_parquets, write_manifest
from data_pipeline.paths import dataset_root, encoder_output_dir, metadata_files, read_metadata
from data_pipeline.runtime import safe_exit_if_requested


def generate_image_embeddings(config_path: str | Path) -> Path:
    config = load_config(config_path)
    raw_root_value = config.get("local", {}).get("raw_root")
    raw_root = require_abs_path(raw_root_value, "local.raw_root") if raw_root_value else dataset_root(config)

    embeddings_cfg = config.get("embeddings", {})
    image_cfg = embeddings_cfg.get("image_encoder", {})
    kind = str(image_cfg.get("kind", "dinov2")).lower()

    if kind in {"dinov2", "dinov3"}:
        return generate_dino_embeddings(config, raw_root, kind)
    if kind == "open_clip":
        return generate_openclip_image_embeddings(config, raw_root)

    raise ValueError(f"Unsupported embeddings.image_encoder.kind: {kind}")


def generate_dino_embeddings(
    config: dict,
    raw_root: Path,
    encoder_kind: str,
) -> Path:
    from transformers import AutoImageProcessor, AutoModel

    embeddings_cfg = config.get("embeddings", {})
    image_cfg = embeddings_cfg.get("image_encoder", {})

    batch_size = int(embeddings_cfg.get("batch_size_image", 32))
    normalize = bool(embeddings_cfg.get("normalize", True))
    device = resolve_device(embeddings_cfg.get("device", "auto"))

    default_model = {
        "dinov2": "facebook/dinov2-base",
        "dinov3": "facebook/dinov3-vits16-pretrain-lvd1689m",
    }[encoder_kind]
    model_name = image_cfg.get("model_name", default_model)
    output_name = image_cfg.get("output_name", model_name.replace("/", "_"))
    pooling = image_cfg.get("pooling", "cls")
    skip_failed_images = bool(image_cfg.get("skip_failed_images", False))

    token = image_cfg.get("hf_token")
    processor = load_dino_processor(AutoImageProcessor, model_name, token)
    model = AutoModel.from_pretrained(model_name, token=token).to(device)
    model.eval()

    images_df = unique_images(config)
    rows = []

    with torch.inference_mode():
        image_batches = iter_loaded_image_batches(config, images_df, raw_root, batch_size, skip_failed_images)
        for loaded in tqdm(image_batches, total=ceil(len(images_df) / batch_size), desc=f"Encoding {encoder_kind.upper()} images"):
            if not loaded:
                continue
            pil_images = [image for _, image in loaded]
            inputs = processor(images=pil_images, return_tensors="pt").to(device)
            outputs = model(**inputs)

            features = pool_vision_outputs(outputs, pooling, encoder_kind)

            features = l2_normalize(features.float(), normalize).cpu().numpy()

            for row, embedding in zip((row for row, _ in loaded), features):
                rows.append(image_embedding_row(row, embedding, encoder_kind, model_name, None, output_name))

    output_dir = encoder_output_dir(config, "image_embeddings", output_name)
    embeddings = pd.DataFrame(rows)
    shard_rows = image_cfg.get("max_rows_per_file") or embeddings_cfg.get("max_rows_per_file")
    save_split_parquets(
        embeddings,
        output_dir,
        max_rows_per_file=int(shard_rows) if shard_rows else None,
    )
    write_manifest(
        output_dir,
        image_manifest(
            metadata_files=metadata_files(config),
            embeddings=embeddings,
            encoder_kind=encoder_kind,
            model_name=model_name,
            pretrained=None,
            output_name=output_name,
            normalized=normalize,
            extra={"pooling": pooling},
        ),
    )
    print(f"Saved image embeddings: {output_dir}")
    return output_dir


def load_dino_processor(processor_class, model_name: str, token):
    try:
        return processor_class.from_pretrained(model_name, token=token)
    except OSError as exc:
        message = str(exc)
        if "403" in message or "gated" in message.lower() or "forbidden" in message.lower():
            raise OSError(
                f"Cannot access image processor for {model_name}. "
                "If this is a gated Hugging Face model, accept the model terms, then run "
                "`huggingface-cli login` with a token that has access to public gated repositories. "
                "For fine-grained tokens, enable access to public gated repos in the token settings."
            ) from exc
        raise


def pool_vision_outputs(outputs, pooling: str, encoder_kind: str) -> torch.Tensor:
    if pooling == "pooler" and getattr(outputs, "pooler_output", None) is not None:
        return outputs.pooler_output

    hidden_states = getattr(outputs, "last_hidden_state", None)
    if hidden_states is None and isinstance(outputs, (tuple, list)) and len(outputs):
        hidden_states = outputs[0]
    if hidden_states is None:
        raise ValueError(f"{encoder_kind} model output does not contain last_hidden_state")

    if pooling == "cls":
        return hidden_states[:, 0]
    if pooling == "mean_patch":
        return hidden_states[:, 1:].mean(dim=1)
    if pooling == "mean":
        return hidden_states.mean(dim=1)

    raise ValueError(f"Unsupported {encoder_kind} pooling: {pooling}")


def generate_openclip_image_embeddings(
    config: dict,
    raw_root: Path,
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
    skip_failed_images = bool(image_cfg.get("skip_failed_images", False))

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    model.eval()

    images_df = unique_images(config)
    rows = []

    with torch.inference_mode():
        image_batches = iter_loaded_image_batches(config, images_df, raw_root, batch_size, skip_failed_images)
        for loaded in tqdm(image_batches, total=ceil(len(images_df) / batch_size), desc="Encoding OpenCLIP images"):
            if not loaded:
                continue
            image_tensors = [preprocess(image) for _, image in loaded]
            image_batch = torch.stack(image_tensors).to(device)
            features = model.encode_image(image_batch)
            features = l2_normalize(features.float(), normalize).cpu().numpy()

            for row, embedding in zip((row for row, _ in loaded), features):
                rows.append(image_embedding_row(row, embedding, "open_clip", model_name, pretrained, output_name))

    output_dir = encoder_output_dir(config, "image_embeddings", output_name)
    embeddings = pd.DataFrame(rows)
    shard_rows = image_cfg.get("max_rows_per_file") or embeddings_cfg.get("max_rows_per_file")
    save_split_parquets(
        embeddings,
        output_dir,
        max_rows_per_file=int(shard_rows) if shard_rows else None,
    )
    write_manifest(
        output_dir,
        image_manifest(
            metadata_files=metadata_files(config),
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


def unique_images(config: dict) -> pd.DataFrame:
    metadata = read_metadata(config)
    if "image_uri" not in metadata.columns and "relative_image_path" in metadata.columns:
        metadata["image_uri"] = metadata["relative_image_path"]
    columns = ["image_id", "split", "file_name", "image_uri"]
    return metadata[columns].drop_duplicates("image_id").sort_values("image_id").reset_index(drop=True)


def iter_loaded_image_batches(
    config: dict,
    images_df: pd.DataFrame,
    raw_root: Path,
    batch_size: int,
    skip_failed_images: bool,
):
    source_cfg = config.get("source", {})
    image_cfg = config.get("embeddings", {}).get("image_encoder", {})
    image_column = source_cfg.get("image_column")
    use_source_column = bool(image_cfg.get("use_source_image_column", True))

    if source_cfg.get("kind") in {"hf_table", "hf_imagenet1k"} and use_source_column:
        yield from iter_source_image_batches(
            config,
            images_df,
            image_column or "image",
            batch_size,
            skip_failed_images,
        )
        return

    for batch_df in batches(list(images_df.index), batch_size):
        chunk = images_df.loc[list(batch_df)]
        yield load_image_batch(chunk, raw_root, skip_failed_images)


def iter_source_image_batches(
    config: dict,
    images_df: pd.DataFrame,
    image_column: str,
    batch_size: int,
    skip_failed_images: bool,
):
    source_cfg = config["source"]
    rows_by_image_id = {str(row["image_id"]): row for _, row in images_df.iterrows()}
    batch = []

    for index, item in enumerate(iter_embedding_source(config)):
        image_id = embedding_source_image_id(config, item, index)
        row = rows_by_image_id.get(image_id)
        if row is None:
            continue

        try:
            image = image_obj_to_rgb(item.get(image_column))
        except (OSError, TypeError, ValueError) as exc:
            if not skip_failed_images:
                raise
            print(f"Skipping image_id={image_id}: {exc}")
            continue

        batch.append((row, image))
        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def iter_embedding_source(config: dict):
    source_kind = config["source"].get("kind")
    if source_kind == "hf_table":
        from data_pipeline.sources.hf_table import iter_hf_table

        yield from iter_hf_table(config)
        return
    if source_kind == "hf_imagenet1k":
        from data_pipeline.sources.hf_imagenet1k import iter_hf_imagenet1k

        yield from iter_hf_imagenet1k(config)
        return
    raise ValueError(f"Unsupported source.kind for source image streaming: {source_kind}")


def embedding_source_image_id(config: dict, item: dict, index: int) -> str:
    source_kind = config["source"].get("kind")
    if source_kind == "hf_imagenet1k":
        from data_pipeline.sources.hf_imagenet1k import imagenet_image_id

        return imagenet_image_id(config, item, index)

    id_column = config["source"].get("id_column")
    raw_id = item.get(id_column) if id_column else item.get("id")
    return str(raw_id if raw_id is not None else index)


def load_image_batch(chunk: pd.DataFrame, raw_root: Path, skip_failed_images: bool) -> list[tuple[pd.Series, Image.Image]]:
    loaded = []
    for _, row in chunk.iterrows():
        try:
            loaded.append((row, load_rgb_image(row, raw_root)))
        except (OSError, URLError, TimeoutError) as exc:
            if not skip_failed_images:
                raise
            print(f"Skipping image_id={row['image_id']}: {exc}")
    return loaded


def load_rgb_image(row: pd.Series, raw_root: Path) -> Image.Image:
    image_uri = str(row.get("image_uri", ""))
    if image_uri and not image_uri.startswith(("http://", "https://", "zip://")):
        return load_local_rgb_image(raw_root / image_uri)
    if image_uri.startswith(("http://", "https://")):
        return load_remote_rgb_image(image_uri)
    return load_local_rgb_image(raw_root / "images" / str(row["file_name"]))


def load_local_rgb_image(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Missing local image: {path}")
    return Image.open(path).convert("RGB")


def load_remote_rgb_image(url: str) -> Image.Image:
    request = Request(url, headers={"User-Agent": "tinyclip-feature-pipeline/0.1"})
    with urlopen(request, timeout=20) as response:
        return Image.open(BytesIO(response.read())).convert("RGB")


def image_obj_to_rgb(image_obj) -> Image.Image:
    if image_obj is None:
        raise ValueError("source image column is empty")
    if isinstance(image_obj, Image.Image):
        return image_obj.convert("RGB")
    if isinstance(image_obj, dict):
        if image_obj.get("path"):
            return Image.open(image_obj["path"]).convert("RGB")
        if image_obj.get("bytes"):
            return Image.open(BytesIO(image_obj["bytes"])).convert("RGB")
    raise TypeError(f"Unsupported source image type: {type(image_obj)}")


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
    metadata_files: list[Path],
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
        "metadata_files": [str(path) for path in metadata_files],
        "encoder_kind": encoder_kind,
        "model_name": model_name,
        "pretrained": pretrained,
        "output_name": output_name,
        "normalized": normalized,
        "num_rows": int(len(embeddings)),
        "num_images": int(embeddings["image_id"].nunique()) if "image_id" in embeddings else 0,
        "embedding_dim": int(embeddings["embedding_dim"].iloc[0]) if len(embeddings) else None,
        **extra,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate frozen image embeddings from local images.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--safe-exit",
        action="store_true",
        help="Exit with os._exit(0) after successful completion to avoid native-library shutdown crashes on some clusters.",
    )
    args = parser.parse_args()
    generate_image_embeddings(args.config)
    config = load_config(args.config)
    safe_exit_if_requested(args.safe_exit or bool(config.get("runtime", {}).get("safe_exit", False)))


if __name__ == "__main__":
    main()

