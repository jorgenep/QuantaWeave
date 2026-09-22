"""Centralized checkpoint loading: the project's one real trust boundary, in one place.

Every QuantaWeave checkpoint (model.pt, adapter.pt, sharded weights) is a plain ``torch.save``'d dict that mixes
tensors with ordinary Python objects (RNG state, config snapshots, optimizer state, run metadata) — ``torch.load``'s
default ``weights_only=True`` restricted unpickler rejects that, so every loader in this project needs
``weights_only=False``. That is equivalent to ``pickle.load``: a malicious checkpoint file can execute arbitrary
code the moment it is loaded, before any of this project's own validation runs.

That is a reasonable trade for a research pipeline where every checkpoint was produced by your own
``train_quantweave_moe.py`` / ``finetune_quantweave_moe.py`` run — but it means the same rule as unpickling any
other file: **never load a checkpoint you did not produce yourself or do not otherwise trust.** See SECURITY.md.

Every loader in ``src/`` goes through :func:`load_checkpoint` instead of calling ``torch.load`` directly, so this
docstring is the one place that trust boundary is documented and reasoned about, not duplicated fourteen times.
"""

from pathlib import Path
from typing import Optional, Union

import torch


def load_checkpoint(path: Union[str, Path], map_location: Optional[Union[str, torch.device]] = None) -> dict:
    """``torch.load`` with ``weights_only=False`` — see this module's docstring for what that means and why."""
    return torch.load(path, map_location=map_location, weights_only=False)
