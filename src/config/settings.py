"""Shared paths and preprocessing defaults for the BigEarthNet pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
SPLITS_DIR = BASE_DIR / "outputs" / "metadata"
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = 0
NORMALIZATION_MODE = "per_band"
PRESERVE_RESOLUTION = False
IMAGE_SIZE = (224, 224)
VV_BAND_SUFFIX = "_VV.tif"
VH_BAND_SUFFIX = "_VH.tif"


def setup_logging(name: str) -> logging.Logger:
    """Return a consistently configured module logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)
    return logger