from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml
from datasets import load_dataset

from adapter_training import (
    DeepResidualAdapter,
    ProjectionAdapter,
    ResidualAdapter,
    load_config,
)
from testing.testing_functions import flickr30k_retrieval_at_1


DEFAULT_REPO_ID = "StanislavLev/tiny-clip-image-encoders-adapter"
DEFAULT_IMAGE_FILE = (
    "imagenet1k/image_embeddings/dinov3_vits16_pretrain_lvd1689m/"
    "validation-00000.parquet"
)
DEFAULT_TEXT_FILE = (
    "imagenet1k/text_embeddings/tinyclip_vit_39m_16_text_19m_yfcc15m/"
    "validation-00000.parquet"
)
DEFAULT_METADATA_FILE = "imagenet1k/metadata/validation-00000.parquet"
DEFAULT_FLICKR_IMAGE_FILES = [
    "flickr30k/image_embeddings/dinov3_vits16_pretrain_lvd1689m/train.parquet",
    "flickr30k/image_embeddings/dinov3_vits16_pretrain_lvd1689m/validation.parquet",
    "flickr30k/image_embeddings/dinov3_vits16_pretrain_lvd1689m/test.parquet",
]
DEFAULT_FLICKR_TEXT_FILES = [
    "flickr30k/text_embeddings/tinyclip_vit_39m_16_text_19m_yfcc15m/train-00000.parquet",
    "flickr30k/text_embeddings/tinyclip_vit_39m_16_text_19m_yfcc15m/train-00001.parquet",
    "flickr30k/text_embeddings/tinyclip_vit_39m_16_text_19m_yfcc15m/validation.parquet",
    "flickr30k/text_embeddings/tinyclip_vit_39m_16_text_19m_yfcc15m/test.parquet",
]
DEFAULT_CLASSES = [
    "a photo of a tiger cat",
    "a photo of a golden retriever",
    "a photo of a airliner",
    "a photo of a sports car, sport car",
    "a photo of a school bus",
    "a photo of a espresso maker",
]


@dataclass
class AdapterSpec:
    label: str
    architecture: str
    checkpoint: Path
    config_path: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create ImageNet six-class CKA and t-SNE visualizations for raw "
            "DINOv3 embeddings and all discovered DINO-to-TinyCLIP adapters."
        )
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--image-file", default=DEFAULT_IMAGE_FILE)
    parser.add_argument("--text-file", default=DEFAULT_TEXT_FILE)
    parser.add_argument("--metadata-file", default=DEFAULT_METADATA_FILE)
    parser.add_argument(
        "--flickr-image-files",
        nargs="+",
        default=DEFAULT_FLICKR_IMAGE_FILES,
        help="Flickr30k image embedding parquet files for CKA/retrieval.",
    )
    parser.add_argument(
        "--flickr-text-files",
        nargs="+",
        default=DEFAULT_FLICKR_TEXT_FILES,
        help="Flickr30k text embedding parquet files for CKA/retrieval.",
    )
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--output-dir", default="visualizations/alignment")
    parser.add_argument(
        "--only-flickr",
        action="store_true",
        help="Only generate the Flickr30k CKA/retrieval plot.",
    )
    parser.add_argument(
        "--only-imagenet",
        action="store_true",
        help="Only generate the ImageNet CKA and t-SNE plots.",
    )
    parser.add_argument(
        "--only-tsne",
        action="store_true",
        help="Only regenerate the six-class ImageNet t-SNE plot.",
    )
    parser.add_argument(
        "--flickr-plot-name",
        default="flickr30k_cka_vs_text_retrieval.png",
        help="Filename for the Flickr30k CKA/retrieval plot.",
    )
    parser.add_argument("--retrieval-batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--samples-per-class", type=int, default=50)
    parser.add_argument("--tsne-fig-width", type=float, default=9.5)
    parser.add_argument("--tsne-panel-height", type=float, default=5.0)
    parser.add_argument("--tsne-title-font-size", type=float, default=24.0)
    parser.add_argument("--tsne-legend-font-size", type=float, default=16.0)
    parser.add_argument("--tsne-image-point-size", type=float, default=18.0)
    parser.add_argument("--tsne-text-point-size", type=float, default=130.0)
    parser.add_argument(
        "--tsne-cache-file",
        default=None,
        help="Cache file for fast six-class t-SNE redraws.",
    )
    parser.add_argument(
        "--refresh-tsne-cache",
        action="store_true",
        help="Rebuild the six-class t-SNE cache before plotting.",
    )
    parser.add_argument(
        "--tsne-highlight-top-k",
        type=int,
        default=12,
        help="Number of most aligned ImageNet classes to label in each t-SNE panel.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument(
        "--include-root-checkpoints",
        action="store_true",
        help="Also try root dino_to_clip_adapter*.pt files, not just models/.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASSES,
        help="Exact ImageNet caption strings to visualize.",
    )
    return parser.parse_args()


def require_plotting_dependencies():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
        from sklearn.manifold import TSNE  # noqa: F401
        from sklearn.preprocessing import StandardScaler  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing plotting dependencies. Install them with:\n"
            "  .\\.venv\\Scripts\\python.exe -m pip install matplotlib scikit-learn\n"
            f"Original error: {exc}"
        ) from exc


def build_adapter(config: dict, architecture: str) -> torch.nn.Module:
    arch_config = config["architectures"][architecture]
    input_dim = int(config["model"]["input_dim"])
    output_dim = int(config["model"]["output_dim"])

    if arch_config["type"] == "mlp":
        return ProjectionAdapter(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=arch_config["hidden_dims"],
            activation=arch_config.get("activation", "GELU"),
            use_layer_norm=arch_config.get("use_layer_norm", False),
            dropout=arch_config.get("dropout", 0.0),
        )
    if arch_config["type"] == "residual":
        return ResidualAdapter(
            input_dim=input_dim,
            output_dim=output_dim,
            bottleneck_dim=arch_config["bottleneck_dim"],
            activation=arch_config.get("activation", "ReLU"),
        )
    if arch_config["type"] == "deep_residual":
        return DeepResidualAdapter(
            input_dim=input_dim,
            hidden_dim=arch_config["hidden_dim"],
            output_dim=output_dim,
            num_blocks=arch_config.get("num_blocks", 6),
            dropout=arch_config.get("dropout", 0.1),
        )
    raise ValueError(f"Unknown architecture type: {arch_config['type']}")


def embedding_matrix(values: Iterable[object]) -> np.ndarray:
    return np.asarray(list(values), dtype=np.float32)


def l2_normalize_np(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-12)


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    numerator = np.linalg.norm(x.T @ y, ord="fro") ** 2
    denominator = np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(
        y.T @ y, ord="fro"
    )
    return float(numerator / denominator) if denominator > 0 else float("nan")


def tsne_2d(features: np.ndarray, perplexity: float, seed: int) -> np.ndarray:
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    scaled = StandardScaler().fit_transform(features)
    safe_perplexity = min(perplexity, max(2.0, (len(features) - 1) / 3))
    return TSNE(
        n_components=2,
        perplexity=safe_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(scaled)


def short_label(caption: str) -> str:
    prefix = "a photo of a "
    label = caption[len(prefix) :] if caption.startswith(prefix) else caption
    return label.split(",")[0]


def load_imagenet_six_class_data(args: argparse.Namespace):
    image_ds = load_dataset(args.repo_id, data_files=args.image_file, split="train")
    text_ds = load_dataset(args.repo_id, data_files=args.text_file, split="train")
    meta_ds = load_dataset(args.repo_id, data_files=args.metadata_file, split="train")

    captions = np.asarray(meta_ds["caption"], dtype=object)
    image_ids = np.asarray(meta_ds["image_id"], dtype=object)
    requested = list(args.classes)
    counts = Counter(captions.tolist())
    missing = [caption for caption in requested if counts[caption] == 0]
    if missing:
        raise ValueError(
            "Requested classes were not found:\n"
            + "\n".join(f"  {caption}" for caption in missing)
        )

    rng = np.random.default_rng(args.seed)
    selected_indices: list[int] = []
    selected_captions: list[str] = []
    for caption in requested:
        class_indices = np.flatnonzero(captions == caption)
        if len(class_indices) > args.samples_per_class:
            class_indices = rng.choice(
                class_indices, size=args.samples_per_class, replace=False
            )
        selected_indices.extend(sorted(class_indices.tolist()))
        selected_captions.extend([caption] * min(len(class_indices), args.samples_per_class))

    image_embeddings = embedding_matrix(image_ds.select(selected_indices)["embedding"])

    text_by_image_id = {
        image_id: embedding
        for image_id, embedding in zip(text_ds["image_id"], text_ds["embedding"])
    }
    selected_image_ids = image_ids[selected_indices]
    sample_text_embeddings = embedding_matrix(
        text_by_image_id[image_id] for image_id in selected_image_ids
    )

    prototype_embeddings = []
    for caption in requested:
        class_index = int(np.flatnonzero(captions == caption)[0])
        prototype_embeddings.append(text_by_image_id[image_ids[class_index]])

    return (
        image_embeddings,
        sample_text_embeddings,
        embedding_matrix(prototype_embeddings),
        selected_captions,
        requested,
    )


def load_imagenet_all_pairs(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, list[str]]:
    image_ds = load_dataset(args.repo_id, data_files=args.image_file, split="train")
    text_ds = load_dataset(args.repo_id, data_files=args.text_file, split="train")
    meta_ds = load_dataset(args.repo_id, data_files=args.metadata_file, split="train")

    image_ids = list(image_ds["image_id"])
    text_by_image_id = {
        image_id: embedding
        for image_id, embedding in zip(text_ds["image_id"], text_ds["embedding"])
    }
    caption_by_image_id = {
        image_id: caption
        for image_id, caption in zip(meta_ds["image_id"], meta_ds["caption"])
    }

    image_embeddings = embedding_matrix(image_ds["embedding"])
    text_embeddings = embedding_matrix(text_by_image_id[image_id] for image_id in image_ids)
    captions = [caption_by_image_id[image_id] for image_id in image_ids]
    return image_embeddings, text_embeddings, captions


def load_flickr30k_data(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    image_ds = load_dataset(
        args.repo_id,
        data_files=list(args.flickr_image_files),
        split="train",
    )
    text_ds = load_dataset(
        args.repo_id,
        data_files=list(args.flickr_text_files),
        split="train",
    )

    image_ids = [str(image_id) for image_id in image_ds["image_id"]]
    text_image_ids = [str(image_id) for image_id in text_ds["image_id"]]
    image_embeddings = embedding_matrix(image_ds["embedding"])
    text_embeddings = embedding_matrix(text_ds["embedding"])
    return image_embeddings, text_embeddings, image_ids, text_image_ids


def paired_flickr_image_embeddings(
    image_embeddings: np.ndarray,
    image_ids: list[str],
    text_image_ids: list[str],
) -> np.ndarray:
    image_by_id = {image_id: image_embeddings[index] for index, image_id in enumerate(image_ids)}
    return embedding_matrix(image_by_id[image_id] for image_id in text_image_ids)


def safe_label(text: str) -> str:
    text = text.replace("\\", "/").replace("arch-", "")
    text = re.sub(r"[^A-Za-z0-9_.=-]+", " ", text)
    return text.strip()


def architecture_from_name(path: Path) -> str | None:
    lowered = str(path).lower()
    if "simple_linear" in lowered or "simple-linear" in lowered:
        return "simple_linear"
    if "mlp_2layer" in lowered or "mlp_2_layer" in lowered:
        return "mlp_2layer"
    if "mlp_3layer" in lowered:
        return "mlp_3layer"
    if "residual_adapter" in lowered:
        return "residual_adapter"
    if "deep_residual" in lowered:
        return "deep_residual"
    return None


def architecture_from_config(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config.get("active_architecture")


def adapter_label(checkpoint: Path, architecture: str) -> str:
    run_name = checkpoint.parent.name
    pretty_arch = {
        "simple_linear": "Linear adapter",
        "mlp_2layer": "MLP 1-layer",
        "mlp_3layer": "MLP 2-layer",
        "residual_adapter": "Residual adapter",
        "deep_residual": "Deep residual",
    }.get(architecture, architecture)

    if run_name.startswith("arch-"):
        return pretty_arch
    if checkpoint.name.startswith("dino_to_clip_adapter"):
        return pretty_arch
    return safe_label(f"{pretty_arch} {checkpoint.stem}")


def discover_adapters(models_dir: Path, include_root: bool) -> list[AdapterSpec]:
    candidates: list[Path] = []
    if models_dir.exists():
        candidates.extend(models_dir.rglob("best.pt"))
        candidates.extend(models_dir.rglob("dino_to_clip_adapter*.pt"))
    if include_root:
        candidates.extend(Path(".").glob("dino_to_clip_adapter*.pt"))

    specs: list[AdapterSpec] = []
    seen_paths: set[Path] = set()
    used_labels: set[str] = set()
    for checkpoint in sorted(candidates, key=lambda path: str(path)):
        checkpoint = checkpoint.resolve()
        if checkpoint in seen_paths:
            continue
        seen_paths.add(checkpoint)

        config_path = checkpoint.parent / "config.yaml"
        architecture = architecture_from_name(checkpoint) or architecture_from_config(
            config_path
        )
        if architecture is None:
            continue

        label = adapter_label(checkpoint, architecture)
        base_label = label
        suffix = 2
        while label in used_labels:
            label = f"{base_label} #{suffix}"
            suffix += 1
        used_labels.add(label)

        specs.append(
            AdapterSpec(
                label=label,
                architecture=architecture,
                checkpoint=checkpoint,
                config_path=config_path if config_path.exists() else None,
            )
        )
    return specs


def project_embeddings(
    adapter: torch.nn.Module, image_embeddings: np.ndarray, batch_size: int = 4096
) -> np.ndarray:
    adapter.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(image_embeddings), batch_size):
            batch = torch.from_numpy(image_embeddings[start : start + batch_size])
            outputs.append(adapter(batch).cpu())
    return torch.cat(outputs, dim=0).numpy()


def load_adapter_outputs(
    base_config: dict,
    specs: list[AdapterSpec],
    image_embeddings: np.ndarray,
) -> tuple[dict[str, np.ndarray], list[AdapterSpec], list[str]]:
    outputs: dict[str, np.ndarray] = {}
    loaded_specs: list[AdapterSpec] = []
    skipped: list[str] = []

    for spec in specs:
        config = load_config(spec.config_path) if spec.config_path else base_config
        try:
            adapter = build_adapter(config, spec.architecture)
            state_dict = torch.load(spec.checkpoint, map_location="cpu")
            adapter.load_state_dict(state_dict)
        except Exception as exc:
            skipped.append(f"{spec.label}: {exc}")
            continue

        outputs[spec.label] = project_embeddings(adapter, image_embeddings)
        loaded_specs.append(spec)
    return outputs, loaded_specs, skipped


def class_colors(class_captions: list[str]):
    import matplotlib.pyplot as plt

    colors = plt.cm.tab10.colors
    return {
        caption: colors[index % len(colors)]
        for index, caption in enumerate(class_captions)
    }


def scatter_images(ax, xy, image_captions, class_captions, colors, point_size: float = 18.0):
    labels = np.asarray(image_captions, dtype=object)
    for caption in class_captions:
        mask = labels == caption
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=point_size,
            alpha=0.78,
            marker="o",
            color=colors[caption],
            label=short_label(caption),
            edgecolors="none",
        )


def scatter_text(ax, xy, class_captions, colors, point_size: float = 130.0):
    for index, caption in enumerate(class_captions):
        ax.scatter(
            xy[index, 0],
            xy[index, 1],
            s=point_size,
            alpha=0.95,
            marker="*",
            color=colors[caption],
            edgecolors="black",
            linewidths=0.45,
        )


def pad_to_same_dim(*arrays: np.ndarray) -> list[np.ndarray]:
    max_dim = max(array.shape[1] for array in arrays)
    padded = []
    for array in arrays:
        if array.shape[1] == max_dim:
            padded.append(array)
            continue
        pad_width = max_dim - array.shape[1]
        padded.append(np.pad(array, ((0, 0), (0, pad_width)), mode="constant"))
    return padded


def plot_cka(cka_scores: dict[str, float], output_path: Path):
    import matplotlib.pyplot as plt

    labels = list(cka_scores)
    values = [cka_scores[label] for label in labels]
    palette = ["#6b7280", "#4e79a7", "#59a14f", "#f28e2b", "#e15759", "#b07aa1"]
    colors = [palette[index % len(palette)] for index in range(len(labels))]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.3), 5))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0, max(1.0, max(values) * 1.15))
    ax.set_ylabel("Linear CKA vs TinyCLIP text")
    ax.set_title("ImageNet Validation Representational Alignment")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_flickr_cka_retrieval(
    metrics: dict[str, dict[str, float]],
    output_path: Path,
    title: str = "Flickr30k: CKA vs Text Retrieval",
):
    import matplotlib.pyplot as plt

    labels = list(metrics)
    palette = ["#6b7280", "#4e79a7", "#59a14f", "#f28e2b", "#e15759", "#b07aa1"]

    fig, ax = plt.subplots(figsize=(8, 6))
    for index, label in enumerate(labels):
        values = metrics[label]
        retrieval_percent = values["flickr30k_i2t_at_1"] * 100.0
        ax.scatter(
            values["cka"],
            retrieval_percent,
            s=110,
            color=palette[index % len(palette)],
            edgecolors="black",
            linewidths=0.5,
            label=label,
            zorder=3,
        )
        ax.annotate(
            label,
            (values["cka"], retrieval_percent),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlabel("Linear CKA vs TinyCLIP text on Flickr30k")
    ax.set_ylabel("Flickr30k image-to-text retrieval@1 (%)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def compute_flickr_metrics(
    args: argparse.Namespace,
    base_config: dict,
    specs: list[AdapterSpec],
) -> tuple[dict[str, dict[str, float]], list[AdapterSpec], list[str], int, int]:
    (
        flickr_image,
        flickr_text,
        flickr_image_ids,
        flickr_text_image_ids,
    ) = load_flickr30k_data(args)
    flickr_outputs, loaded_specs, skipped = load_adapter_outputs(
        base_config, specs, flickr_image
    )
    flickr_metrics = {}
    for label, projected_images in flickr_outputs.items():
        paired_images = paired_flickr_image_embeddings(
            projected_images, flickr_image_ids, flickr_text_image_ids
        )
        flickr_metrics[label] = {
            "cka": linear_cka(paired_images, flickr_text),
            **flickr30k_retrieval_at_1(
                image_features=torch.from_numpy(projected_images),
                text_features=torch.from_numpy(flickr_text),
                image_ids=flickr_image_ids,
                text_image_ids=flickr_text_image_ids,
                batch_size=args.retrieval_batch_size,
                device=args.device,
            ),
        }

    return (
        flickr_metrics,
        loaded_specs,
        skipped,
        len(flickr_image),
        len(flickr_text),
    )


def plot_tsne_grid(
    raw_image: np.ndarray,
    text_prototypes: np.ndarray,
    adapter_outputs: dict[str, np.ndarray],
    image_captions: list[str],
    class_captions: list[str],
    output_path: Path,
    perplexity: float,
    seed: int,
    fig_width: float,
    panel_height: float,
    title_font_size: float,
    legend_font_size: float,
    image_point_size: float,
    text_point_size: float,
    precomputed_xy: list[np.ndarray] | None = None,
) -> list[np.ndarray]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = class_colors(class_captions)
    panels = [("Before adapter projection", raw_image, text_prototypes, True)]
    panels.extend(
        ("After adapter projection", output, text_prototypes)
        for output in adapter_outputs.values()
    )

    cols = 1
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, panel_height * rows))
    axes = np.asarray(axes).reshape(-1)

    computed_xy: list[np.ndarray] = []

    for panel_index, (ax, panel) in enumerate(zip(axes, panels)):
        if len(panel) == 4:
            title, image_features, text_features, pad_raw = panel
        else:
            title, image_features, text_features = panel
            pad_raw = False

        if pad_raw:
            image_features, text_features = pad_to_same_dim(image_features, text_features)

        combined = np.vstack([image_features, text_features])
        if precomputed_xy is not None and panel_index < len(precomputed_xy):
            xy = precomputed_xy[panel_index]
        else:
            xy = tsne_2d(combined, perplexity=perplexity, seed=seed)
        computed_xy.append(xy)
        image_xy = xy[: len(image_features)]
        text_xy = xy[len(image_features) :]

        scatter_images(
            ax,
            image_xy,
            image_captions,
            class_captions,
            colors,
            point_size=image_point_size,
        )
        scatter_text(ax, text_xy, class_captions, colors, point_size=text_point_size)
        ax.set_title(title, fontsize=title_font_size, pad=12)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[len(panels) :]:
        ax.axis("off")

    class_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=colors[caption],
            markeredgecolor="none",
            markersize=max(7, legend_font_size * 0.7),
            label=short_label(caption),
        )
        for caption in class_captions
    ]
    shape_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#4b5563",
            markeredgecolor="none",
            markersize=max(7, legend_font_size * 0.7),
            label="Images",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor="#4b5563",
            markeredgecolor="black",
            markeredgewidth=0.45,
            markersize=max(10, legend_font_size * 1.0),
            label="Text",
        ),
    ]
    fig.legend(
        class_handles + shape_handles,
        [handle.get_label() for handle in class_handles + shape_handles],
        frameon=False,
        loc="center left",
        bbox_to_anchor=(0.79, 0.5),
        ncol=1,
        fontsize=legend_font_size,
        handletextpad=0.6,
        labelspacing=0.8,
    )
    fig.tight_layout(rect=(0, 0, 0.78, 1))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return computed_xy


def class_centroids(
    image_features: np.ndarray,
    text_features: np.ndarray,
    captions: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    labels = np.asarray(captions, dtype=object)
    class_captions = sorted(set(captions), key=short_label)
    image_centroids = []
    text_centroids = []
    for caption in class_captions:
        mask = labels == caption
        image_centroids.append(image_features[mask].mean(axis=0))
        text_centroids.append(text_features[mask].mean(axis=0))

    return (
        np.asarray(image_centroids, dtype=np.float32),
        np.asarray(text_centroids, dtype=np.float32),
        class_captions,
    )


def matching_cosine(image_features: np.ndarray, text_features: np.ndarray) -> np.ndarray:
    image_features = l2_normalize_np(image_features)
    text_features = l2_normalize_np(text_features)
    return np.sum(image_features * text_features, axis=1)


def plot_tsne_all_classes(
    raw_image: np.ndarray,
    text_features: np.ndarray,
    adapter_outputs: dict[str, np.ndarray],
    captions: list[str],
    output_path: Path,
    perplexity: float,
    seed: int,
    highlight_top_k: int,
    selected_adapter_labels: list[str] | None = None,
    highlight_reference_label: str | None = None,
):
    import matplotlib.pyplot as plt

    raw_image_centroids, text_centroids, class_captions = class_centroids(
        raw_image, text_features, captions
    )
    adapter_centroids = {
        label: class_centroids(output, text_features, captions)[0]
        for label, output in adapter_outputs.items()
    }
    if selected_adapter_labels is None:
        selected_adapter_labels = list(adapter_centroids)

    reference_label = highlight_reference_label or (
        selected_adapter_labels[0] if selected_adapter_labels else None
    )
    if reference_label in adapter_centroids:
        highlighted_scores = matching_cosine(
            adapter_centroids[reference_label], text_centroids
        )
    else:
        raw_for_similarity, text_for_similarity = pad_to_same_dim(
            raw_image_centroids, text_centroids
        )
        highlighted_scores = matching_cosine(raw_for_similarity, text_for_similarity)

    top_k = min(highlight_top_k, len(highlighted_scores))
    highlighted = np.argsort(highlighted_scores)[-top_k:][::-1]

    panels = [("Before adapter projection", raw_image_centroids, text_centroids, True)]
    for label in selected_adapter_labels:
        if label in adapter_centroids:
            panels.append(("After adapter projection", adapter_centroids[label], text_centroids, False))

    cols = 3
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.8 * cols, 5.25 * rows))
    axes = np.asarray(axes).reshape(-1)
    highlight_colors = plt.cm.tab20(np.linspace(0, 1, max(highlight_top_k, 1)))

    for ax, (title, image_centroids, text_centroids_panel, pad_raw) in zip(axes, panels):
        if pad_raw:
            image_plot, text_plot = pad_to_same_dim(image_centroids, text_centroids_panel)
        else:
            image_plot, text_plot = image_centroids, text_centroids_panel

        similarities = matching_cosine(image_plot, text_plot)

        combined = np.vstack([image_plot, text_plot])
        xy = tsne_2d(combined, perplexity=perplexity, seed=seed)
        image_xy = xy[: len(image_plot)]
        text_xy = xy[len(image_plot) :]

        ax.scatter(
            image_xy[:, 0],
            image_xy[:, 1],
            s=4,
            alpha=0.32,
            marker="o",
            color="#6b7280",
            edgecolors="none",
            label="Image class centroid",
        )
        ax.scatter(
            text_xy[:, 0],
            text_xy[:, 1],
            s=7,
            alpha=0.25,
            marker="^",
            color="#2563eb",
            edgecolors="none",
            label="Text class centroid",
        )

        for rank, class_index in enumerate(highlighted):
            color = highlight_colors[rank]
            ax.plot(
                [image_xy[class_index, 0], text_xy[class_index, 0]],
                [image_xy[class_index, 1], text_xy[class_index, 1]],
                color=color,
                alpha=0.72,
                linewidth=1.0,
                zorder=2,
            )
            ax.scatter(
                image_xy[class_index, 0],
                image_xy[class_index, 1],
                s=46,
                marker="o",
                color=color,
                edgecolors="black",
                linewidths=0.45,
                zorder=4,
            )
            ax.scatter(
                text_xy[class_index, 0],
                text_xy[class_index, 1],
                s=115,
                marker="*",
                color=color,
                edgecolors="black",
                linewidths=0.45,
                zorder=5,
            )
            ax.annotate(
                short_label(class_captions[class_index]),
                text_xy[class_index],
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7.2,
                color="#111827",
            )

        ax.set_title(
            f"{title}\nSame {top_k} highlighted classes",
            fontsize=11,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[len(panels) :]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:2], labels[:2], frameon=False, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def centroid_metrics(adapter_outputs, text_prototypes, image_captions, class_captions):
    labels = np.asarray(image_captions, dtype=object)
    text = l2_normalize_np(text_prototypes)
    metrics = {}
    for label, features in adapter_outputs.items():
        features = l2_normalize_np(features)
        centroids = np.asarray(
            [features[labels == caption].mean(axis=0) for caption in class_captions],
            dtype=np.float32,
        )
        similarities = l2_normalize_np(centroids) @ text.T
        metrics[label] = {
            "class_centroid_to_matching_text_mean_cosine": float(
                np.diag(similarities).mean()
            ),
            "class_centroid_to_text_top1": float(
                (similarities.argmax(axis=1) == np.arange(len(class_captions))).mean()
            ),
        }
    return metrics


def best_adapter_from_metrics(output_dir: Path, specs: list[AdapterSpec]) -> str | None:
    metrics_path = output_dir / "imagenet_alignment_metrics.yaml"
    if not metrics_path.exists():
        return None

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = yaml.safe_load(f) or {}

    available = {spec.label for spec in specs}
    stored_best = metrics.get("tsne_best_adapter")
    if stored_best in available:
        return stored_best

    cka_scores = metrics.get("cka", {})
    candidates = {
        label: score
        for label, score in cka_scores.items()
        if label in available and label != "Raw DINOv3"
    }
    if candidates:
        return max(candidates, key=candidates.get)

    return None


def default_tsne_cache_path(args: argparse.Namespace, output_dir: Path) -> Path:
    if args.tsne_cache_file:
        return Path(args.tsne_cache_file)
    return output_dir / "tsne_six_class_cache.npz"


def load_tsne_cache(
    cache_path: Path,
    args: argparse.Namespace,
    adapter_label: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    list[str],
    list[np.ndarray] | None,
] | None:
    if args.refresh_tsne_cache or not cache_path.exists():
        return None

    try:
        data = np.load(cache_path, allow_pickle=False)
        if str(data["adapter_label"]) != adapter_label:
            return None
        if int(data["samples_per_class"]) != int(args.samples_per_class):
            return None
        if list(data["requested_classes"].astype(str)) != list(args.classes):
            return None

        xy_cache = None
        if (
            "xy_before" in data
            and "xy_after" in data
            and "xy_seed" in data
            and "xy_perplexity" in data
            and int(data["xy_seed"]) == int(args.seed)
            and float(data["xy_perplexity"]) == float(args.perplexity)
        ):
            xy_cache = [
                data["xy_before"].astype(np.float32),
                data["xy_after"].astype(np.float32),
            ]

        return (
            data["raw_image"].astype(np.float32),
            data["text_prototypes"].astype(np.float32),
            data["adapter_output"].astype(np.float32),
            data["image_captions"].astype(str).tolist(),
            data["class_captions"].astype(str).tolist(),
            xy_cache,
        )
    except Exception:
        return None


def save_tsne_cache(
    cache_path: Path,
    args: argparse.Namespace,
    adapter_label: str,
    raw_image: np.ndarray,
    text_prototypes: np.ndarray,
    adapter_output: np.ndarray,
    image_captions: list[str],
    class_captions: list[str],
    xy_cache: list[np.ndarray] | None = None,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_items = {
        "adapter_label": np.asarray(adapter_label),
        "samples_per_class": np.asarray(args.samples_per_class),
        "requested_classes": np.asarray(args.classes, dtype=str),
        "raw_image": raw_image.astype(np.float32),
        "text_prototypes": text_prototypes.astype(np.float32),
        "adapter_output": adapter_output.astype(np.float32),
        "image_captions": np.asarray(image_captions, dtype=str),
        "class_captions": np.asarray(class_captions, dtype=str),
    }
    if xy_cache is not None and len(xy_cache) >= 2:
        cache_items.update(
            {
                "xy_before": xy_cache[0].astype(np.float32),
                "xy_after": xy_cache[1].astype(np.float32),
                "xy_seed": np.asarray(args.seed),
                "xy_perplexity": np.asarray(args.perplexity),
            }
        )
    np.savez_compressed(cache_path, **cache_items)


def main():
    args = parse_args()
    require_plotting_dependencies()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = discover_adapters(Path(args.models_dir), args.include_root_checkpoints)
    if not specs:
        raise FileNotFoundError(f"No adapter checkpoints found under {args.models_dir}")

    base_config = load_config(args.config)
    if args.only_flickr:
        (
            flickr_metrics,
            loaded_specs,
            skipped_flickr,
            flickr_image_count,
            flickr_text_count,
        ) = compute_flickr_metrics(args, base_config, specs)
        flickr_plot_path = output_dir / args.flickr_plot_name
        plot_flickr_cka_retrieval(
            flickr_metrics,
            flickr_plot_path,
            title="Flickr30k Test: CKA vs Text Retrieval",
        )

        print(f"Saved Flickr30k CKA/retrieval plot to: {flickr_plot_path}")
        print(
            "Flickr30k vectors: "
            f"{flickr_image_count} images, {flickr_text_count} captions"
        )
        print(f"Retrieval batch size: {args.retrieval_batch_size}")
        print("Adapters:")
        for spec in loaded_specs:
            values = flickr_metrics[spec.label]
            print(
                f"  - {spec.label}: CKA={values['cka']:.4f}, "
                f"I2T@1={values['flickr30k_i2t_at_1'] * 100:.2f}%, "
                f"T2I@1={values['flickr30k_t2i_at_1'] * 100:.2f}%"
            )
        if skipped_flickr:
            print("Skipped incompatible adapters:")
            for item in skipped_flickr:
                print(f"  - {item}")
        return

    if args.only_tsne:
        best_adapter_label = best_adapter_from_metrics(output_dir, specs) or specs[0].label
        cache_path = default_tsne_cache_path(args, output_dir)
        cached = load_tsne_cache(cache_path, args, best_adapter_label)

        if cached is None:
            (
                raw_image,
                sample_text,
                text_prototypes,
                image_captions,
                class_captions,
            ) = load_imagenet_six_class_data(args)
            selected_specs = [spec for spec in specs if spec.label == best_adapter_label]
            adapter_outputs, loaded_specs, skipped = load_adapter_outputs(
                base_config, selected_specs, raw_image
            )
            if not adapter_outputs:
                raise RuntimeError("No adapter checkpoint could be loaded for t-SNE.")

            best_adapter_label = loaded_specs[0].label
            adapter_output = adapter_outputs[best_adapter_label]
            xy_cache = None
        else:
            (
                raw_image,
                text_prototypes,
                adapter_output,
                image_captions,
                class_captions,
                xy_cache,
            ) = cached
            skipped = []

        computed_xy = plot_tsne_grid(
            raw_image=raw_image,
            text_prototypes=text_prototypes,
            adapter_outputs={best_adapter_label: adapter_output},
            image_captions=image_captions,
            class_captions=class_captions,
            output_path=output_dir / "tsne_imagenet_adapters.png",
            perplexity=args.perplexity,
            seed=args.seed,
            fig_width=args.tsne_fig_width,
            panel_height=args.tsne_panel_height,
            title_font_size=args.tsne_title_font_size,
            legend_font_size=args.tsne_legend_font_size,
            image_point_size=args.tsne_image_point_size,
            text_point_size=args.tsne_text_point_size,
            precomputed_xy=xy_cache,
        )
        if xy_cache is None:
            save_tsne_cache(
                cache_path,
                args,
                best_adapter_label,
                raw_image,
                text_prototypes,
                adapter_output,
                image_captions,
                class_captions,
                xy_cache=computed_xy,
            )
        print(f"Saved t-SNE plot to: {output_dir / 'tsne_imagenet_adapters.png'}")
        print(f"Best adapter shown: {best_adapter_label}")
        print(f"t-SNE cache: {cache_path}")
        if skipped:
            print("Skipped incompatible adapters:")
            for item in skipped:
                print(f"  - {item}")
        return

    (
        raw_image,
        sample_text,
        text_prototypes,
        image_captions,
        class_captions,
    ) = load_imagenet_six_class_data(args)

    adapter_outputs, loaded_specs, skipped = load_adapter_outputs(
        base_config, specs, raw_image
    )
    if not adapter_outputs:
        raise RuntimeError("No adapter checkpoints could be loaded.")

    raw_image_all, text_all, all_captions = load_imagenet_all_pairs(args)

    adapter_outputs_all, _, skipped_all = load_adapter_outputs(
        base_config, loaded_specs, raw_image_all
    )

    cka_scores = {"Raw DINOv3": linear_cka(raw_image_all, text_all)}
    cka_scores.update(
        {
            label: linear_cka(projected, text_all)
            for label, projected in adapter_outputs_all.items()
        }
    )
    best_adapter_label = max(adapter_outputs_all, key=lambda label: cka_scores[label])

    plot_cka(cka_scores, output_dir / "cka_imagenet_adapters.png")

    best_adapter_outputs = {best_adapter_label: adapter_outputs[best_adapter_label]}
    computed_xy = plot_tsne_grid(
        raw_image=raw_image,
        text_prototypes=text_prototypes,
        adapter_outputs=best_adapter_outputs,
        image_captions=image_captions,
        class_captions=class_captions,
        output_path=output_dir / "tsne_imagenet_adapters.png",
        perplexity=args.perplexity,
        seed=args.seed,
        fig_width=args.tsne_fig_width,
        panel_height=args.tsne_panel_height,
        title_font_size=args.tsne_title_font_size,
        legend_font_size=args.tsne_legend_font_size,
        image_point_size=args.tsne_image_point_size,
        text_point_size=args.tsne_text_point_size,
    )
    save_tsne_cache(
        default_tsne_cache_path(args, output_dir),
        args,
        best_adapter_label,
        raw_image,
        text_prototypes,
        best_adapter_outputs[best_adapter_label],
        image_captions,
        class_captions,
        xy_cache=computed_xy,
    )

    if args.only_imagenet:
        summary = {
            "image_points": int(len(raw_image)),
            "cka_pairs": int(len(raw_image_all)),
            "text_vectors_for_cka": int(len(text_all)),
            "image_points_in_tsne": int(len(raw_image)),
            "text_prototypes_in_tsne": int(len(text_prototypes)),
            "samples_per_class": args.samples_per_class,
            "tsne_classes": int(len(class_captions)),
            "tsne_best_adapter": best_adapter_label,
            "classes": [short_label(caption) for caption in class_captions],
            "cka": cka_scores,
            "adapters": [
                {
                    "label": spec.label,
                    "architecture": spec.architecture,
                    "checkpoint": str(spec.checkpoint),
                }
                for spec in loaded_specs
            ],
            "centroid_metrics": centroid_metrics(
                adapter_outputs, text_prototypes, image_captions, class_captions
            ),
            "skipped_adapters": skipped + skipped_all,
            "note": (
                "CKA is computed on all aligned ImageNet validation pairs. The "
                "t-SNE shows the same six selected ImageNet classes in the raw "
                "DINOv3 panel and in the best-adapter panel. Images are circles, "
                "TinyCLIP text prototypes are stars, and class identity is shown "
                "by the legend. The raw DINOv3 t-SNE panel zero-pads DINOv3 "
                "vectors from 384D to 512D only for visualization. The adapter "
                "panel jointly embeds projected image vectors and TinyCLIP text "
                "prototypes in their real shared 512D space."
            ),
        }
        with (output_dir / "imagenet_alignment_metrics.yaml").open(
            "w", encoding="utf-8"
        ) as f:
            yaml.safe_dump(summary, f, sort_keys=False)

        print(f"Saved CKA plot to: {output_dir / 'cka_imagenet_adapters.png'}")
        print(f"Saved t-SNE plot to: {output_dir / 'tsne_imagenet_adapters.png'}")
        print("Adapters:")
        for spec in loaded_specs:
            print(f"  - {spec.label}: {spec.checkpoint}")
        if skipped:
            print("Skipped incompatible adapters:")
            for item in skipped:
                print(f"  - {item}")
        return

    (
        flickr_metrics,
        _,
        skipped_flickr,
        flickr_image_count,
        flickr_text_count,
    ) = compute_flickr_metrics(args, base_config, loaded_specs)
    plot_flickr_cka_retrieval(
        flickr_metrics, output_dir / args.flickr_plot_name
    )

    summary = {
        "image_points": int(len(raw_image)),
        "cka_pairs": int(len(raw_image_all)),
        "text_vectors_for_cka": int(len(text_all)),
        "image_points_in_tsne": int(len(raw_image)),
        "text_prototypes_in_tsne": int(len(text_prototypes)),
        "samples_per_class": args.samples_per_class,
        "tsne_classes": int(len(class_captions)),
        "tsne_best_adapter": best_adapter_label,
        "classes": [short_label(caption) for caption in class_captions],
        "cka": cka_scores,
        "adapters": [
            {
                "label": spec.label,
                "architecture": spec.architecture,
                "checkpoint": str(spec.checkpoint),
            }
            for spec in loaded_specs
        ],
        "centroid_metrics": centroid_metrics(
            adapter_outputs, text_prototypes, image_captions, class_captions
        ),
        "flickr30k": {
            "image_vectors": int(flickr_image_count),
            "text_vectors": int(flickr_text_count),
            "retrieval_metric": "flickr30k_i2t_at_1",
            "cka_pairs": int(flickr_text_count),
            "metrics": flickr_metrics,
        },
        "skipped_adapters": skipped + skipped_all + skipped_flickr,
        "note": (
            "CKA is computed on all aligned ImageNet validation pairs. t-SNE shows "
            "the same six selected ImageNet classes in the raw DINOv3 panel and "
            "in the best-adapter panel. Images are circles, TinyCLIP text "
            "prototypes are stars, and class identity is shown by the legend. "
            "The raw DINOv3 t-SNE panel zero-pads DINOv3 vectors from 384D to "
            "512D only for visualization. The adapter panel jointly embeds "
            "projected image vectors and TinyCLIP text prototypes in their real "
            "shared 512D space. Flickr30k CKA repeats "
            "each image embedding once per caption so the CKA matrices are "
            "caption-aligned; Flickr30k retrieval is image-to-text@1 over all "
            "provided Flickr30k splits."
        ),
    }
    with (output_dir / "imagenet_alignment_metrics.yaml").open(
        "w", encoding="utf-8"
    ) as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    print(f"Saved CKA plot to: {output_dir / 'cka_imagenet_adapters.png'}")
    print(f"Saved t-SNE plot to: {output_dir / 'tsne_imagenet_adapters.png'}")
    print(
        "Saved Flickr30k CKA/retrieval plot to: "
        f"{output_dir / args.flickr_plot_name}"
    )
    print("Adapters:")
    for spec in loaded_specs:
        print(f"  - {spec.label}: {spec.checkpoint}")
    if skipped:
        print("Skipped incompatible adapters:")
        for item in skipped:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
