"""Apple-Silicon allocator cleanup helpers.

The caller supplies the device that actually held temporary tensors.  This
module never imports PyTorch merely to clear an allocator cache.
"""
from __future__ import annotations

import sys


def clear_mps_cache(device) -> bool:
    """Clear cached MPS blocks only after a completed MPS operation.

    Returns whether an MPS cache-clear call completed. Cleanup must not mask
    a processing failure, so an unavailable or failing MPS runtime is logged
    and treated as a no-op.
    """
    if getattr(device, "type", str(device)).lower() != "mps":
        return False

    torch = sys.modules.get("torch")
    empty_cache = getattr(getattr(torch, "mps", None), "empty_cache", None)
    if not callable(empty_cache):
        return False
    try:
        empty_cache()
    except Exception as error:
        print(f"⚠️ [Mac] MPS cache cleanup failed "
              f"({type(error).__name__}: {error})")
        return False
    return True
