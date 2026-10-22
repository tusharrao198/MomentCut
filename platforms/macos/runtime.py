"""Native Apple-Silicon runtime commands.

Run from the OpenShorts checkout:

    ./.venv/bin/python -m platforms.macos.runtime backend
    ./.venv/bin/python -m platforms.macos.runtime dashboard
    ./.venv/bin/python -m platforms.macos.runtime status

The module starts only loopback services. It never changes the Windows/NVIDIA
Docker topology and does not start Ollama, Docker, the renderer, or Postiz.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from urllib.parse import urlparse
from dataclasses import dataclass
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LOOPBACK_HOST = "127.0.0.1"
DEFAULT_BACKEND_PORT = 8000
DEFAULT_DASHBOARD_PORT = 5175
DEFAULT_RENDERER_PORT = 3100
DEFAULT_OLLAMA_PORT = 11434


@dataclass(frozen=True)
class RuntimePorts:
    """Loopback ports used by the native Mac topology."""

    backend: int = DEFAULT_BACKEND_PORT
    dashboard: int = DEFAULT_DASHBOARD_PORT
    renderer: int = DEFAULT_RENDERER_PORT
    ollama: int = DEFAULT_OLLAMA_PORT


def _port(raw, name):
    try:
        value = int(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer port") from error
    if not 1 <= value <= 65535:
        raise ValueError(f"{name} must be between 1 and 65535")
    return value


def ports_from_args(args):
    """Return validated loopback ports from the parsed CLI arguments."""
    return RuntimePorts(
        backend=_port(args.backend_port, "--backend-port"),
        dashboard=_port(args.dashboard_port, "--dashboard-port"),
        renderer=_port(args.renderer_port, "--renderer-port"),
        ollama=_port(args.ollama_port, "--ollama-port"),
    )


def _load_repository_env():
    """Load the checkout's `.env` with the same non-overriding rule as app.py."""
    from dotenv import load_dotenv

    load_dotenv(REPOSITORY_ROOT / ".env")


def backend_environment(ports, environ=None):
    """Return backend environment with a local renderer default.

    This replaces a Docker-only renderer service name on a native host but
    does not start a renderer. The launcher is intentionally authoritative;
    use direct Uvicorn invocation for a custom remote renderer topology.
    """
    _load_repository_env()
    result = dict(os.environ if environ is None else environ)
    result["RENDER_SERVICE_URL"] = f"http://{LOOPBACK_HOST}:{ports.renderer}"
    return result


def backend_command(ports):
    """Build the native loopback Uvicorn command."""
    return [
        sys.executable,
        "-m",
        "uvicorn",
        "app:app",
        "--host",
        LOOPBACK_HOST,
        "--port",
        str(ports.backend),
    ]


def dashboard_environment(ports, environ=None):
    """Return native Vite targets for the local backend and optional renderer.

    Clear `VITE_API_URL` so the dashboard uses Vite's same-origin proxy rather
    than an inherited Docker or cloud endpoint. The launcher is a fixed native
    preset; run Vite directly when a custom API topology is intentional.
    """
    result = dict(os.environ if environ is None else environ)
    result["VITE_API_URL"] = ""
    result["VITE_PROXY_TARGET"] = f"http://{LOOPBACK_HOST}:{ports.backend}"
    result["VITE_RENDER_TARGET"] = f"http://{LOOPBACK_HOST}:{ports.renderer}"
    return result


def dashboard_command(ports):
    """Build the native, loopback-only Vite command."""
    return [
        "npm",
        "run",
        "dev",
        "--",
        "--host",
        LOOPBACK_HOST,
        "--port",
        str(ports.dashboard),
    ]


def _http(url, *, parse_json=False, headers=None):
    try:
        request = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(request, timeout=2) as response:
            body = response.read().decode("utf-8")
            result = {
                "available": 200 <= response.status < 300,
                "status": response.status,
            }
            if parse_json:
                result["body"] = json.loads(body) if body else None
            return result
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as error:
        return {"available": False, "error": f"{type(error).__name__}: {error}"}


def _local_ollama_configuration(ports):
    """Return (configured, model) when OpenShorts targets this Ollama."""
    _load_repository_env()
    base_url = (os.environ.get("LLM_BASE_URL") or "").strip()
    model = (os.environ.get("LLM_MODEL") or "").strip()
    try:
        parsed = urlparse(base_url)
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError:
        return False, ""
    configured = (
        parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        and port == ports.ollama
    )
    return configured, model if configured else ""


def _ollama_status(ports):
    """Return local Ollama reachability and its configured required model."""
    result = _http(
        f"http://{LOOPBACK_HOST}:{ports.ollama}/api/version", parse_json=True)
    configured, model = _local_ollama_configuration(ports)
    model_result = {"required": True, "name": model, "available": None}
    if result["available"] and configured and model:
        tags = _http(
            f"http://{LOOPBACK_HOST}:{ports.ollama}/api/tags", parse_json=True)
        names = {
            str(item.get("name") or "")
            for item in (tags.get("body") or {}).get("models", [])
            if isinstance(item, dict)
        }
        model_result["available"] = model in names
        if not tags["available"]:
            model_result["error"] = tags.get("error", "could not list local models")
    elif not configured:
        model_result["error"] = (
            "LLM_BASE_URL must target local Ollama on this Mac for the native profile"
        )
    elif not model:
        model_result["error"] = "LLM_MODEL must name the local Ollama model"
    result["configured"] = configured
    result["model"] = model_result
    return result


def status(ports):
    """Report native loopback service reachability without starting anything."""
    backend = _http(
        f"http://{LOOPBACK_HOST}:{ports.backend}/health/ready", parse_json=True)
    dashboard = _http(
        f"http://{LOOPBACK_HOST}:{ports.dashboard}/",
        headers={"Accept": "text/html"},
    )
    renderer = _http(
        f"http://{LOOPBACK_HOST}:{ports.renderer}/health", parse_json=True)
    ollama = _ollama_status(ports)
    return {
        "backend": backend,
        "dashboard": dashboard,
        "renderer": renderer,
        "ollama": ollama,
        # Renderer is optional: ordinary OpenShorts clip generation uses the
        # worker's FFmpeg path, while the dashboard and backend are required.
        "ready": bool(
            backend["available"]
            and dashboard["available"]
            and ollama["available"]
            and ollama["configured"]
            and ollama["model"]["available"] is True
        ),
    }


def _exec(command, *, cwd, env):
    os.chdir(cwd)
    os.execvpe(command[0], command, env)


def _run_backend(ports):
    _exec(
        backend_command(ports),
        cwd=REPOSITORY_ROOT,
        env=backend_environment(ports),
    )


def _run_dashboard(ports):
    if not shutil.which("npm"):
        raise RuntimeError("npm is unavailable; install Node.js before starting the dashboard")
    _exec(
        dashboard_command(ports),
        cwd=REPOSITORY_ROOT / "dashboard",
        env=dashboard_environment(ports),
    )


def parser():
    result = argparse.ArgumentParser(
        description="Run or inspect the native Apple-Silicon OpenShorts topology.")
    subcommands = result.add_subparsers(dest="command", required=True)
    for name in ("backend", "dashboard", "status"):
        command = subcommands.add_parser(name)
        command.add_argument("--backend-port", default=DEFAULT_BACKEND_PORT)
        command.add_argument("--dashboard-port", default=DEFAULT_DASHBOARD_PORT)
        command.add_argument("--renderer-port", default=DEFAULT_RENDERER_PORT)
        command.add_argument("--ollama-port", default=DEFAULT_OLLAMA_PORT)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        ports = ports_from_args(args)
        if args.command == "backend":
            _run_backend(ports)
        elif args.command == "dashboard":
            _run_dashboard(ports)
        else:
            report = status(ports)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if report["ready"] else 1
    except (RuntimeError, ValueError) as error:
        print(f"❌ [Mac runtime] {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
