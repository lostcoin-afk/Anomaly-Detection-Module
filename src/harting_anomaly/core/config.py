from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class PathConfig:
    """
    Central filesystem layout for the application.

    Nothing outside this module should hard-code paths such as:
        data/raw
        data/prepared
        models/patchcore

    This makes the project portable between:
        - developer laptop
        - workstation
        - Docker
        - Jetson
        - production server
    """

    project_root: Path

    @property
    def data_root(self) -> Path:
        return self.project_root / "data"

    @property
    def raw_data_root(self) -> Path:
        return self.data_root / "raw"

    @property
    def prepared_data_root(self) -> Path:
        return self.data_root / "prepared"

    @property
    def models_root(self) -> Path:
        return self.project_root / "models"

    @property
    def patchcore_models_root(self) -> Path:
        return self.models_root / "patchcore"

    @property
    def metadata_root(self) -> Path:
        return self.project_root / "metadata"

    @property
    def configs_root(self) -> Path:
        return self.project_root / "configs"

    @property
    def logs_root(self) -> Path:
        return self.project_root / "logs"

    @property
    def frontend_root(self) -> Path:
        return self.project_root / "frontend"

    def ensure_directories(self) -> None:
        """Create all application-owned directories."""
        directories = [
            self.data_root,
            self.raw_data_root,
            self.prepared_data_root,
            self.models_root,
            self.patchcore_models_root,
            self.metadata_root,
            self.configs_root,
            self.logs_root,
            self.frontend_root,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DatasetConfig:
    """
    Dataset behavior.

    These values are deliberately configuration rather than scattered
    throughout the data-processing code.
    """

    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15

    allowed_extensions: tuple[str, ...] = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
    )

    def validate(self) -> None:
        total = (
            self.train_ratio
            + self.validation_ratio
            + self.test_ratio
        )

        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                "Dataset split ratios must sum to 1.0. "
                f"Got {total:.6f}"
            )

        if any(
            ratio <= 0.0
            for ratio in (
                self.train_ratio,
                self.validation_ratio,
                self.test_ratio,
            )
        ):
            raise ValueError("Dataset split ratios must all be > 0.")

    def split_ratios(self) -> dict[str, float]:
        return {
            "train": self.train_ratio,
            "val": self.validation_ratio,
            "test": self.test_ratio,
        }


@dataclass(frozen=True)
class AppConfig:
    """
    Top-level application configuration.
    """

    paths: PathConfig
    dataset: DatasetConfig


def find_project_root() -> Path:
    """
    Resolve the project root.

    Priority:
        1. HARTING_ANOMALY_ROOT environment variable.
        2. Current working directory.

    The environment variable is useful inside Docker or production.
    """

    configured_root = os.getenv("HARTING_ANOMALY_ROOT")

    if configured_root:
        return Path(configured_root).expanduser().resolve()

    return Path.cwd().resolve()


def load_config() -> AppConfig:
    """
    Construct and validate application configuration.
    """

    paths = PathConfig(
        project_root=find_project_root(),
    )

    dataset = DatasetConfig()
    dataset.validate()

    return AppConfig(
        paths=paths,
        dataset=dataset,
    )


CONFIG = load_config()