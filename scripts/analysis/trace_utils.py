"""Archive discovery shared by the offline analysis commands."""

from __future__ import annotations

import glob
import os
import warnings
from collections.abc import Callable
from pickle import UnpicklingError
from zipfile import BadZipFile

import numpy as np


def newest_trace(path: str | None, suitable: Callable) -> str | None:
    """Resolve a file or select the newest suitable trace below a directory.

    An explicit file is returned for the caller to validate. During directory
    discovery, close each archive before opening the next and skip damaged or
    partially written files. This allows analysis alongside a live recorder.
    """
    if path and os.path.isfile(path):
        return path
    candidates = glob.glob(os.path.join(path or "logs", "**", "trace_*.npz"), recursive=True)
    matches = []
    for candidate in candidates:
        try:
            with np.load(candidate, allow_pickle=True) as archive:
                if suitable(archive):
                    matches.append(candidate)
        except (OSError, ValueError, EOFError, BadZipFile, UnpicklingError) as error:
            warnings.warn(f"Skipping unreadable trace {candidate}: {error}", stacklevel=2)
    return max(matches, key=os.path.getmtime) if matches else None
