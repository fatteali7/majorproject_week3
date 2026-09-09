"""Preprocessing for three-band Sentinel-2 patches."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image


def _find_band(folder: Path, band: str) -> Path:
    candidates = sorted(path for path in folder.rglob("*") if path.is_file() and band in path.stem)
    if not candidates:
        raise FileNotFoundError(f"Could not find Sentinel-2 band {band} under {folder}")
    return candidates[0]


def _read_band(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        array = np.load(path)
    else:
        with Image.open(path) as image:
            array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(f"Expected a single-band raster at {path}, got shape {array.shape}")
    return array.astype(np.float32, copy=False)


def preprocess_s2_patch(
    patch_folder: Path,
    target_size: tuple[int, int] = (224, 224),
    bands: Iterable[str] = ("B02", "B03", "B04"),
) -> torch.Tensor:
    arrays = [_read_band(_find_band(Path(patch_folder), band)) for band in bands]
    resized = [
        np.asarray(
            Image.fromarray(array).resize(
                (target_size[1], target_size[0]), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )
        for array in arrays
    ]
    output = np.stack(resized, axis=0)
    mean = output.mean(axis=(1, 2), keepdims=True)
    std = output.std(axis=(1, 2), keepdims=True)
    output = (output - mean) / np.maximum(std, 1e-6)
    return torch.from_numpy(output.astype(np.float32, copy=False))