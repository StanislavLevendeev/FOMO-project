from __future__ import annotations

import hashlib


def deterministic_split(
    image_id: str,
    seed: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> str:
    total = train_ratio + validation_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    key = f"{seed}:{image_id}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    value = int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)

    if value < train_ratio:
        return "train"
    if value < train_ratio + validation_ratio:
        return "validation"
    return "test"

