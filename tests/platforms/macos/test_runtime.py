import json

from platforms.macos import runtime


def test_backend_command_is_native_loopback():
    ports = runtime.RuntimePorts(backend=8123)

    assert runtime.backend_command(ports) == [
        runtime.sys.executable, "-m", "uvicorn", "app:app",
        "--host", "127.0.0.1", "--port", "8123",
    ]


def test_dashboard_forces_its_loopback_targets():
    ports = runtime.RuntimePorts(backend=8123, renderer=3123)

    default = runtime.dashboard_environment(ports, {})
    assert default["VITE_PROXY_TARGET"] == "http://127.0.0.1:8123"
    assert default["VITE_RENDER_TARGET"] == "http://127.0.0.1:3123"

    inherited = runtime.dashboard_environment(ports, {
        "VITE_API_URL": "http://backend:8000",
        "VITE_PROXY_TARGET": "http://backend:8000",
        "VITE_RENDER_TARGET": "http://renderer:3100",
    })
    assert inherited["VITE_API_URL"] == ""
    assert inherited["VITE_PROXY_TARGET"] == "http://127.0.0.1:8123"
    assert inherited["VITE_RENDER_TARGET"] == "http://127.0.0.1:3123"


def test_backend_forces_native_renderer_target(monkeypatch):
    ports = runtime.RuntimePorts(renderer=3123)
    monkeypatch.setattr(runtime, "_load_repository_env", lambda: None)

    assert runtime.backend_environment(ports, {})["RENDER_SERVICE_URL"] == (
        "http://127.0.0.1:3123"
    )
    assert runtime.backend_environment(ports, {
        "RENDER_SERVICE_URL": "http://renderer:3100",
    })["RENDER_SERVICE_URL"] == "http://127.0.0.1:3123"


def test_status_requires_backend_dashboard_and_ollama_not_renderer(monkeypatch):
    responses = {
        "http://127.0.0.1:8000/health/ready": {"available": True},
        "http://127.0.0.1:5175/": {"available": True},
        "http://127.0.0.1:3100/health": {"available": False},
        "http://127.0.0.1:11434/api/version": {"available": True},
        "http://127.0.0.1:11434/api/tags": {
            "available": True,
            "body": {"models": [{"name": "qwen3:8b"}]},
        },
    }
    monkeypatch.setattr(runtime, "_http", lambda url, **_kwargs: responses[url])
    monkeypatch.setattr(runtime, "_local_ollama_configuration", lambda _ports: (True, "qwen3:8b"))

    report = runtime.status(runtime.RuntimePorts())

    assert report["ready"] is True
    assert report["renderer"]["available"] is False
    assert report["ollama"]["model"] == {
        "required": True, "name": "qwen3:8b", "available": True,
    }


def test_http_passes_headers_to_request(monkeypatch):
    captured = {}

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b""

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(runtime.urllib.request, "urlopen", fake_urlopen)

    assert runtime._http("http://127.0.0.1:5175/", headers={"Accept": "text/html"}) == {
        "available": True,
        "status": 200,
    }
    assert captured["request"].get_header("Accept") == "text/html"
    assert captured["timeout"] == 2


def test_status_requests_dashboard_as_html(monkeypatch):
    seen = []

    def fake_http(url, **kwargs):
        seen.append((url, kwargs))
        if url.endswith("/api/tags"):
            return {
                "available": True,
                "body": {"models": [{"name": "qwen3:8b"}]},
            }
        return {"available": True}

    monkeypatch.setattr(runtime, "_http", fake_http)
    monkeypatch.setattr(runtime, "_local_ollama_configuration", lambda _ports: (True, "qwen3:8b"))

    runtime.status(runtime.RuntimePorts())

    assert (
        "http://127.0.0.1:5175/",
        {"headers": {"Accept": "text/html"}},
    ) in seen


def test_local_ollama_configuration_rejects_another_llm_server(monkeypatch):
    monkeypatch.setattr(runtime, "_load_repository_env", lambda: None)
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:8080/v1")
    monkeypatch.setenv("LLM_MODEL", "qwen3:8b")

    assert runtime._local_ollama_configuration(runtime.RuntimePorts()) == (False, "")


def test_status_output_is_json_serializable(monkeypatch, capsys):
    monkeypatch.setattr(runtime, "status", lambda ports: {
        "backend": {"available": True},
        "dashboard": {"available": True},
        "renderer": {"available": False},
        "ollama": {"available": True},
        "ready": True,
    })

    assert runtime.main(["status"]) == 0
    assert json.loads(capsys.readouterr().out)["ready"] is True


def test_rejects_invalid_port():
    assert runtime.main(["status", "--backend-port", "0"]) == 2
