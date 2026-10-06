from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import hashlib
import re
from typing import Iterable

from harting_anomaly.core.config import CONFIG


@dataclass(frozen=True)
class ImageRecord:
    """
    Metadata describing one source image.

    This is the application's internal representation of a dataset sample.
    """

    sample_id: str
    image_path: str

    category: str
    view: str

    piece_id: str
    camera_id: str
    session_id: str

    split: str
    label: str

    source_directory: str


VIEW_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"top", re.IGNORECASE), "top"),
    (re.compile(r"bottom", re.IGNORECASE), "bottom"),
    (re.compile(r"cross[\s_-]*section[\s_-]*1", re.IGNORECASE), "cross_section_1"),
    (re.compile(r"cross[\s_-]*section[\s_-]*2", re.IGNORECASE), "cross_section_2"),
)


def normalize_view(directory_name: str) -> str:
    """
    Convert raw viewpoint directory names into canonical view names.

    Examples:
        harting_xxx-Top
            -> top

        harting_xxx-bottom
            -> bottom

        harting_xxx-CrossSection-1
            -> cross_section_1
    """

    for pattern, normalized_name in VIEW_PATTERNS:
        if pattern.search(directory_name):
            return normalized_name

    return normalize_identifier(directory_name)


def normalize_identifier(value: str) -> str:
    """
    Convert an arbitrary product/directory name into a stable identifier.

    Example:
        harting_Baseline-10E-M-s
            -> harting_baseline_10e_m_s
    """

    normalized = value.strip().lower()
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)

    return normalized.strip("_")


def infer_category(product_directory: Path) -> str:
    """
    Return the canonical product category name.

    We currently use the raw top-level directory name.

    IMPORTANT:
    This intentionally does not attempt to guess product semantics.
    A separate product metadata/config layer can later define display names,
    PLC identifiers, barcodes, etc.
    """

    return normalize_identifier(product_directory.name)


def infer_piece_id(image_path: Path) -> str:
    """
    Best-effort physical-piece identifier.

    For Version 1 we do NOT assume a particular filename convention.

    Therefore:
        image_001.jpg -> image_001

    Later, this function will be replaced/configured according to the
    actual acquisition naming scheme so all views of the same physical
    piece receive the same piece_id.

    This distinction is important because dataset splitting should ultimately
    happen at physical-piece level, not random-image level.
    """

    return normalize_identifier(image_path.stem)


def stable_sample_id(
    category: str,
    view: str,
    image_path: Path,
) -> str:
    """
    Generate a deterministic sample ID.

    The ID remains stable across multiple scans as long as the relative
    source path remains stable.
    """

    relative_path = image_path.relative_to(CONFIG.paths.raw_data_root)

    key = f"{category}|{view}|{relative_path.as_posix()}"

    return hashlib.sha1(
        key.encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()[:16]


def iter_image_files(directory: Path) -> Iterable[Path]:
    """
    Recursively find supported images in a directory.
    """

    allowed = set(CONFIG.dataset.allowed_extensions)

    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in allowed:
            continue

        yield path


def scan_product_directory(product_directory: Path) -> list[ImageRecord]:
    """
    Scan one product directory.

    Expected raw structure:

        product/
            viewpoint_a/
            viewpoint_b/
            ...
    """

    category = infer_category(product_directory)

    records: list[ImageRecord] = []

    for view_directory in sorted(product_directory.iterdir()):
        if not view_directory.is_dir():
            continue

        view = normalize_view(view_directory.name)

        for image_path in iter_image_files(view_directory):
            sample_id = stable_sample_id(
                category=category,
                view=view,
                image_path=image_path,
            )

            record = ImageRecord(
                sample_id=sample_id,
                image_path=str(image_path.resolve()),
                category=category,
                view=view,
                piece_id=infer_piece_id(image_path),
                camera_id="",
                session_id="",
                split="unassigned",
                label="good",
                source_directory=str(view_directory.resolve()),
            )

            records.append(record)

    return records


def scan_dataset() -> list[ImageRecord]:
    """
    Scan the entire raw dataset.

    Expected structure:

        data/raw/
            product_a/
                view_1/
                view_2/

            product_b/
                view_1/
                view_2/
    """

    raw_root = CONFIG.paths.raw_data_root

    if not raw_root.exists():
        raise FileNotFoundError(
            f"Raw dataset directory does not exist: {raw_root}"
        )

    product_directories = sorted(
        path
        for path in raw_root.iterdir()
        if path.is_dir()
    )

    if not product_directories:
        raise RuntimeError(
            f"No product directories found inside: {raw_root}"
        )

    records: list[ImageRecord] = []

    for product_directory in product_directories:
        records.extend(
            scan_product_directory(product_directory)
        )

    return records


def write_catalog(
    records: list[ImageRecord],
    output_path: Path | None = None,
) -> Path:
    """
    Write the complete dataset catalog to CSV.
    """

    if output_path is None:
        output_path = CONFIG.paths.metadata_root / "samples.csv"

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        ImageRecord.__dataclass_fields__.keys()
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in records:
            writer.writerow(asdict(record))

    return output_path


def print_summary(records: list[ImageRecord]) -> None:
    """
    Print a compact human-readable dataset summary.
    """

    categories = sorted(
        {record.category for record in records}
    )

    views = sorted(
        {record.view for record in records}
    )

    print()
    print("=" * 72)
    print("HARTING DATASET CATALOG")
    print("=" * 72)

    print(f"Total images : {len(records)}")
    print(f"Categories   : {len(categories)}")
    print(f"Views        : {len(views)}")

    print()
    print("Categories:")

    for category in categories:
        category_records = [
            record
            for record in records
            if record.category == category
        ]

        category_views = {}

        for record in category_records:
            category_views[record.view] = (
                category_views.get(record.view, 0) + 1
            )

        print(f"  {category}")

        for view, count in sorted(category_views.items()):
            print(f"      {view:<20} {count}")

    print("=" * 72)
    print()


def main() -> None:
    """
    CLI entry point:

        harting-scan
    """

    CONFIG.paths.ensure_directories()

    records = scan_dataset()

    catalog_path = write_catalog(records)

    print_summary(records)

    print(f"Catalog written to:")
    print(f"  {catalog_path}")
    print()


if __name__ == "__main__":
    main()