import job_state


def _state():
    return {
        "schema_version": 1,
        "job_id": "terminal-job",
        "configuration_revision": 1,
        "execution_generation": 3,
        "status": "processing",
    }


def _artifact():
    return {
        "clip_index": 0,
        "relative_path": "clip.mp4",
        "size": 10,
        "sha256": "a" * 64,
        "duration_ms": 1000,
        "probe": {"playable": True},
        "clip_plan_identity": "b" * 64,
        "companions": [],
    }


def test_terminal_decision_and_outbox_are_one_atomic_state(tmp_path):
    job_state.write_job_state(str(tmp_path), _state())
    committed = job_state.commit_terminal_state(
        str(tmp_path),
        3,
        "partial",
        {"0": _artifact()},
        [1],
        {
            "actions": {"reservation": True, "webhook": True},
            "webhook_url": "https://example.test/hook",
        },
    )
    on_disk = job_state.load_job_state(str(tmp_path))
    assert on_disk == committed
    assert on_disk["status"] == "partial"
    assert on_disk["missing_clip_indices"] == [1]
    assert on_disk["terminal_actions"]["webhook"]["state"] == "pending"
    assert on_disk["terminal_action_context"]["webhook_url"].endswith("/hook")


def test_recovered_ambiguous_notification_can_enter_manual_review(tmp_path):
    job_state.write_job_state(str(tmp_path), _state())
    job_state.commit_terminal_state(
        str(tmp_path), 3, "failed", {}, [],
        {"actions": {"clips_ready_email": True}},
    )
    job_state.transition_terminal_action(
        str(tmp_path),
        "clips_ready_email",
        "pending",
        "dispatching",
    )
    recovered = job_state.transition_terminal_action(
        str(tmp_path),
        "clips_ready_email",
        "dispatching",
        "manual_review",
        {"last_error_code": "ambiguous_after_restart"},
    )
    assert recovered["terminal_actions"]["clips_ready_email"][
        "state"
    ] == "manual_review"


def test_autopilot_plan_durably_suppresses_ordinary_completion_email(
    tmp_path,
):
    job_state.write_job_state(str(tmp_path), _state())
    job_state.commit_terminal_state(
        str(tmp_path),
        3,
        "completed",
        {"0": _artifact()},
        [],
        {
            "actions": {
                "autopilot_finalize": True,
                "clips_ready_email": True,
            },
        },
    )
    updated = job_state.configure_autopilot_actions(str(tmp_path), {
        "is_autopilot": True,
        "email": None,
        "publish_items": [],
    })
    assert updated["terminal_actions"]["clips_ready_email"]["state"] == \
        "not_required"
