from platforms.macos import ytdlp


def test_explicit_node_path_is_passed_to_ytdlp():
    assert ytdlp.javascript_runtime_options(
        {"YTDLP_NODE_PATH": "/opt/node/bin/node"},
        which=lambda _name: None,
    ) == {"js_runtimes": {"node": {"path": "/opt/node/bin/node"}}}


def test_path_lookup_is_used_when_no_override_is_set():
    assert ytdlp.javascript_runtime_options(
        {}, which=lambda name: "/usr/local/bin/node" if name == "node" else None
    ) == {"js_runtimes": {"node": {"path": "/usr/local/bin/node"}}}


def test_missing_node_leaves_yt_dlp_defaults_unchanged():
    assert ytdlp.javascript_runtime_options({}, which=lambda _name: None) == {}
