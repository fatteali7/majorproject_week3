"""Preprocessing for two-band Sentinel-1 patches."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image


def _read_band(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        array = np.load(path)
    else:
        with Image.open(path) as image:
            array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(f"Expected a single-band raster at {path}, got shape {array.shape}")
    return array.astype(np.float32, copy=False)


def _resize(array: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(array)
    resized = image.resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def preprocess_sar_patch(
    vv_path: Path,
    vh_path: Path,
    target_size: Tuple[int, int] = (224, 224),
    preserve_res: bool = False,
    norm_mode: str = "per_band",
) -> torch.Tensor:
    bands = [_read_band(Path(vv_path)), _read_band(Path(vh_path))]
    if not preserve_res:
        bands = [_resize(band, target_size) for band in bands]
    output = np.stack(bands, axis=0)
    if norm_mode == "per_band":
        mean = output.mean(axis=(1, 2), keepdims=True)
        std = output.std(axis=(1, 2), keepdims=True)
        output = (output - mean) / np.maximum(std, 1e-6)
    return torch.from_numpy(output.astype(np.float32, copy=False))