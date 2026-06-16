from __future__ import annotations

from dataclasses import asdict, dataclass


VALID_SPLITS = {"train", "validation", "test"}


@dataclass(frozen=True)
class MetadataRow:
    dataset_name: str
    image_id: str
    caption_id: str
    caption_index: int
    caption: str
    file_name: str
    relative_image_path: str
    split: str
    source: str

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_split(split: str | None) -> str | None:
    if split is None:
        return None

    value = str(split).strip().lower()
    aliases = {
        "val": "validation",
        "valid": "validation",
        "dev": "validation",
    }
    return aliases.get(value, value)

