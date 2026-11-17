"""Read offline Tex-LDM texture vectors with explicit image-ID checks."""
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


def load_texture_prior(path: str | Path, expected_image_id: str) -> Tensor:
    """Read NPZ containing image_id (scalar string) and z_t (float32[256]).

    The caller must use this same ID to select the enhanced image and label.
    This validates metadata, not whether Tex-LDM generated the contents.
    """
    with np.load(path, allow_pickle=False) as cache:
        image_id = cache["image_id"]
        if image_id.shape != () or str(image_id.item()) != expected_image_id:
            raise ValueError(f"Image/prior ID mismatch: {path}")
        z_t = cache["z_t"]
        if z_t.shape != (256,) or z_t.dtype != np.float32:
            raise ValueError(f"Expected float32[256] texture prior: {path}")
        if not np.isfinite(z_t).all():
            raise ValueError(f"Non-finite texture prior: {path}")
        return torch.from_numpy(z_t.copy())


def fixed_shuffle_indices(count: int, seed: int = 0) -> Tensor:
    """A seeded permutation with no fixed points, for an entire split.

    Compute once outside the data loader. Never shuffle across splits.
    This single-cycle derangement is an explicit reconstruction choice.
    """
    if count < 2:
        raise ValueError("Shuffled-prior control needs at least two images")
    order = torch.randperm(count, generator=torch.Generator().manual_seed(seed))
    mapping = torch.empty_like(order)
    mapping[order] = order.roll(1)
    return mapping
