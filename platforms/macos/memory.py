"""macOS-only memory helpers.

This module stays dependency-light until the host-memory probe is called.
"""


def host_available_mb():
    """Read host-available unified memory in MiB, or return ``None``."""
    try:
        import psutil

        return int(psutil.virtual_memory().available // (1024 * 1024))
    except Exception:
        return None
