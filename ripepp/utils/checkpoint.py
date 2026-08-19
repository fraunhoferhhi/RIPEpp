"""Utilities for saving and loading model checkpoints with configuration."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from omegaconf import DictConfig, OmegaConf

DEFAULT_CKPT = "ripe++.ckpt"

CKPT_URL_BASE = "https://datacloud.hhi.fraunhofer.de/public.php/dav/files/P6GTAKgejiaK9BC/"

CKPT_VARIANTS = {
    "default": DEFAULT_CKPT,  # ripe++.ckpt
    "aachen_day_night": "ripe++_tokyo_megadepth.ckpt",
    "scared": "ripe++_scared.ckpt",
}


def _checkpoint_cache_dir() -> Path:
    """Persistent per-user dir for downloaded checkpoints."""
    d = Path(torch.hub.get_dir()) / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _resolve_or_download(filename: str) -> Path:
    """Locate a checkpoint file, downloading it into the cache if needed.

    Lookup order:
        1. A repo-local ``weights/<filename>`` (the cloned-repo / dev workflow).
        2. A previously cached copy under :func:`_checkpoint_cache_dir`.
        3. Otherwise download from the server into the cache and return it.
    """
    repo_local = Path("weights") / filename  # dev / cloned-repo workflow
    if repo_local.exists():
        print(f"Using existing checkpoint: {repo_local}")
        return repo_local

    cached = _checkpoint_cache_dir() / filename
    if cached.exists():
        print(f"Using cached checkpoint: {cached}")
        return cached

    print(f"Checkpoint '{filename}' not found. Downloading to {cached} ...")
    torch.hub.download_url_to_file(f"{CKPT_URL_BASE}{filename}", str(cached))
    print("Done.")
    return cached


def resolve_variant_checkpoint(variant: str = "default") -> Path:
    """Return the local path to a named weight variant, downloading if absent.

    Args:
        variant: One of the keys in ``CKPT_VARIANTS`` (e.g. "default",
            "aachen_day_night", "scared").

    Returns:
        Path to the local checkpoint file (repo-local ``weights/`` if present,
        otherwise the per-user cache directory).

    Raises:
        ValueError: If ``variant`` is not a known variant name.
    """
    if variant not in CKPT_VARIANTS:
        raise ValueError(f"Unknown weight variant '{variant}'. Choose from {sorted(CKPT_VARIANTS)}.")
    return _resolve_or_download(CKPT_VARIANTS[variant])


def save_checkpoint(
    state_dict: Dict[str, Any],
    config: DictConfig,
    path: Path,
    step: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save model checkpoint with configuration.

    Args:
        state_dict: Model state dictionary
        config: Hydra configuration (OmegaConf DictConfig)
        path: Path to save checkpoint
        step: Training step number (optional)
        metadata: Additional metadata to save (optional)
    """
    checkpoint = {
        "state_dict": state_dict,
        "config": OmegaConf.to_container(config, resolve=True),
        "timestamp": datetime.now().isoformat(),
    }

    if step is not None:
        checkpoint["step"] = step

    if metadata is not None:
        checkpoint["metadata"] = metadata

    torch.save(checkpoint, path)


def load_checkpoint(path: Path, map_location="cpu"):
    """
    Load checkpoint with backward compatibility.

    Args:
        path: Path to checkpoint file
        map_location: Device to load checkpoint to

    Returns:
        dict with keys:
            - 'state_dict': Model weights
            - 'config': Configuration (None if old checkpoint)
            - 'step': Training step (None if not saved)
            - 'timestamp': Save timestamp (None if not saved)
            - 'metadata': Additional metadata (None if not saved)
    """
    checkpoint = torch.load(path, map_location=map_location)

    # Backward compatibility: old checkpoints are just state_dict
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        return {
            "state_dict": checkpoint,
            "config": None,
            "step": None,
            "timestamp": None,
            "metadata": None,
        }

    # Ensure all expected keys exist
    checkpoint.setdefault("config", None)
    checkpoint.setdefault("step", None)
    checkpoint.setdefault("timestamp", None)
    checkpoint.setdefault("metadata", None)

    return checkpoint


def load_config_from_checkpoint(path: Path) -> Optional[DictConfig]:
    """
    Load only the configuration from a checkpoint.

    Args:
        path: Path to checkpoint file

    Returns:
        OmegaConf DictConfig if config exists, None otherwise
    """
    checkpoint = load_checkpoint(path)

    if checkpoint["config"] is None:
        return None

    return OmegaConf.create(checkpoint["config"])


def validate_checkpoint_path(checkpoint_path):
    if isinstance(checkpoint_path, str):
        checkpoint_path = Path(checkpoint_path)

    if checkpoint_path is None:
        # Resolve the default checkpoint (repo-local weights/, then cache, then download).
        print("No checkpoint path given.")
        checkpoint_path = _resolve_or_download(DEFAULT_CKPT)
    else:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Error: Given checkpoint path {checkpoint_path} does not exist.")

    print(f"Using checkpoint: {checkpoint_path}")

    return checkpoint_path
