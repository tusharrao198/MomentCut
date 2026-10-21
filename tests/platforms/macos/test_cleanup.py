import sys
from types import SimpleNamespace

from platforms.macos import cleanup


def test_clear_mps_cache_only_for_a_completed_mps_operation(monkeypatch):
    cleared = []
    torch = SimpleNamespace(mps=SimpleNamespace(
        empty_cache=lambda: cleared.append(True)))
    monkeypatch.setitem(sys.modules, "torch", torch)

    assert cleanup.clear_mps_cache("cpu") is False
    assert cleared == []
    assert cleanup.clear_mps_cache(SimpleNamespace(type="mps")) is True
    assert cleared == [True]


def test_clear_mps_cache_never_masks_allocator_failure(monkeypatch, capsys):
    def fail():
        raise RuntimeError("Metal reset")

    torch = SimpleNamespace(mps=SimpleNamespace(empty_cache=fail))
    monkeypatch.setitem(sys.modules, "torch", torch)

    assert cleanup.clear_mps_cache("mps") is False
    assert "MPS cache cleanup failed (RuntimeError: Metal reset)" \
        in capsys.readouterr().out
