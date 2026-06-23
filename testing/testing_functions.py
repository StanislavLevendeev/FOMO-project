from __future__ import annotations
from collections.abc import Sequence
from typing import Any
import torch
import torch.nn.functional as F


def resolve_device(device: str | torch.device | None = "auto") -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def normalize_features(features: torch.Tensor | Sequence[Sequence[float]]) -> torch.Tensor:
    features = torch.as_tensor(features, dtype=torch.float32)
    return F.normalize(features, dim=-1)


def _ids_or_range(ids: Sequence[Any] | None, length: int, name: str) -> list[Any]:
    if ids is None:
        return list(range(length))

    ids = list(ids)
    if len(ids) != length:
        raise ValueError(f"{name} must have length {length}, got {len(ids)}")
    return ids


def _recall_at_1(
    query_features: torch.Tensor,
    target_features: torch.Tensor,
    query_ids: Sequence[Any],
    target_ids: Sequence[Any],
    *,
    batch_size: int = 1024,
    device: str | torch.device | None = "auto",
) -> float:
    device = resolve_device(device)
    query_features = normalize_features(query_features)
    target_features = normalize_features(target_features).to(device)

    correct = 0
    total = query_features.shape[0]

    with torch.inference_mode():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = query_features[start:end].to(device)
            top_indices = (batch @ target_features.T).argmax(dim=1).cpu().tolist()

            for query_id, target_index in zip(query_ids[start:end], top_indices):
                if query_id == target_ids[target_index]:
                    correct += 1

    return correct / max(total, 1)


def image_to_text_at_1(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    *,
    image_ids: Sequence[Any] | None = None,
    text_image_ids: Sequence[Any] | None = None,
    batch_size: int = 1024,
    device: str | torch.device | None = "auto",
) -> float:
    """Return I->T @1.

    For Flickr30k, pass one ``image_id`` per image feature and one
    ``text_image_id`` per caption feature. If ids are omitted, features are
    assumed to be one-to-one in the same order.
    """
    image_ids = _ids_or_range(image_ids, len(image_features), "image_ids")
    text_image_ids = _ids_or_range(text_image_ids, len(text_features), "text_image_ids")
    return _recall_at_1(
        image_features,
        text_features,
        image_ids,
        text_image_ids,
        batch_size=batch_size,
        device=device,
    )


def text_to_image_at_1(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    *,
    image_ids: Sequence[Any] | None = None,
    text_image_ids: Sequence[Any] | None = None,
    batch_size: int = 1024,
    device: str | torch.device | None = "auto",
) -> float:
    """Return T->I @1."""
    image_ids = _ids_or_range(image_ids, len(image_features), "image_ids")
    text_image_ids = _ids_or_range(text_image_ids, len(text_features), "text_image_ids")
    return _recall_at_1(
        text_features,
        image_features,
        text_image_ids,
        image_ids,
        batch_size=batch_size,
        device=device,
    )


def retrieval_at_1(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    *,
    image_ids: Sequence[Any] | None = None,
    text_image_ids: Sequence[Any] | None = None,
    batch_size: int = 1024,
    device: str | torch.device | None = "auto",
) -> dict[str, float]:
    """Return both I->T @1 and T->I @1 from feature vectors."""
    return {
        "image_to_text_at_1": image_to_text_at_1(
            image_features,
            text_features,
            image_ids=image_ids,
            text_image_ids=text_image_ids,
            batch_size=batch_size,
            device=device,
        ),
        "text_to_image_at_1": text_to_image_at_1(
            image_features,
            text_features,
            image_ids=image_ids,
            text_image_ids=text_image_ids,
            batch_size=batch_size,
            device=device,
        ),
    }


def flickr30k_retrieval_at_1(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    image_ids: Sequence[Any],
    text_image_ids: Sequence[Any],
    *,
    batch_size: int = 1024,
    device: str | torch.device | None = "auto",
) -> dict[str, float]:
    """Flickr30k-style retrieval with multiple captions per image."""
    scores = retrieval_at_1(
        image_features,
        text_features,
        image_ids=image_ids,
        text_image_ids=text_image_ids,
        batch_size=batch_size,
        device=device,
    )
    return {
        "flickr30k_i2t_at_1": scores["image_to_text_at_1"],
        "flickr30k_t2i_at_1": scores["text_to_image_at_1"],
    }


def _orient_class_features(
    class_features: torch.Tensor,
    *,
    feature_dim: int,
    min_num_classes: int,
) -> torch.Tensor:
    if class_features.ndim != 2:
        raise ValueError("class_features must be a 2D tensor")

    if class_features.shape[1] == feature_dim:
        oriented = class_features
    elif class_features.shape[0] == feature_dim:
        oriented = class_features.T
    else:
        raise ValueError(
            "class_features must have shape [num_classes, dim] or [dim, num_classes]"
        )

    if oriented.shape[0] < min_num_classes:
        raise ValueError(
            f"class_features has {oriented.shape[0]} classes, but labels require "
            f"at least {min_num_classes}"
        )
    return oriented


def topk_accuracy_from_features(
    image_features: torch.Tensor,
    class_features: torch.Tensor,
    labels: torch.Tensor | Sequence[int],
    *,
    topk: Sequence[int] = (1,),
    batch_size: int = 1024,
    device: str | torch.device | None = "auto",
) -> dict[str, float]:
    """Compute zero-shot top-k accuracy from image and class text features.

    ``labels`` must contain integer class ids matching the row order of
    ``class_features``.
    """
    device = resolve_device(device)
    labels = torch.as_tensor(labels, dtype=torch.long)
    image_features = torch.as_tensor(image_features, dtype=torch.float32)
    class_features = torch.as_tensor(class_features, dtype=torch.float32)
    class_features = _orient_class_features(
        class_features,
        feature_dim=image_features.shape[1],
        min_num_classes=int(labels.max().item()) + 1,
    )
    image_features = normalize_features(image_features)
    class_features = normalize_features(class_features).to(device)

    topk = tuple(sorted(set(int(k) for k in topk)))
    max_k = min(max(topk), class_features.shape[0])
    correct = {k: 0 for k in topk}
    total = image_features.shape[0]

    with torch.inference_mode():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_features = image_features[start:end].to(device)
            batch_labels = labels[start:end].to(device)
            predictions = (batch_features @ class_features.T).topk(max_k, dim=1).indices
            matches = predictions.eq(batch_labels[:, None])

            for k in topk:
                correct[k] += int(matches[:, : min(k, max_k)].any(dim=1).sum().item())

    return {f"top{k}": correct[k] / max(total, 1) for k in topk}


def imagenet_zero_shot_accuracy(
    image_features: torch.Tensor,
    class_features: torch.Tensor,
    labels: torch.Tensor | Sequence[int],
    *,
    batch_size: int = 1024,
    device: str | torch.device | None = "auto",
) -> float:
    """Return ImageNet zero-shot top-1 accuracy from feature vectors."""
    return topk_accuracy_from_features(
        image_features,
        class_features,
        labels,
        topk=(1,),
        batch_size=batch_size,
        device=device,
    )["top1"]
