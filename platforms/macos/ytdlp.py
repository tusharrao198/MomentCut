"""Native macOS yt-dlp options kept outside the shared downloader."""
from __future__ import annotations

import os
import shutil


def javascript_runtime_options(environ=None, which=None):
    """Select a local Node runtime for yt-dlp when one is available.

    ``YTDLP_NODE_PATH`` is intentionally an absolute override for launchd and
    other service environments whose PATH does not include the interactive
    shell's Node installation.  An absent runtime leaves yt-dlp's portable
    defaults unchanged.
    """
    environ = os.environ if environ is None else environ
    which = shutil.which if which is None else which
    configured = str(environ.get("YTDLP_NODE_PATH") or "").strip()
    path = configured or which("node")
    if not path:
        return {}
    return {"js_runtimes": {"node": {"path": path}}}
