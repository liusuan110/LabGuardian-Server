from __future__ import annotations

from app.pipeline.vision.side_camera_gate import (
    SideCameraGateDetector,
    SideGateConfig,
    SideGateState,
)


class FakeRunner:
    def __init__(self, stats_payload: dict):
        self._stats_payload = stats_payload

    def stats(self) -> dict:
        return dict(self._stats_payload)


def _side_stats(*, diff: float, presence: float, running: bool = True) -> dict:
    return {
        "running": running,
        "latest_frame_ts": 1.0,
        "latest_diff_score": diff,
        "latest_presence_score": presence,
        "presence_detected": presence >= 4.0,
    }


def test_side_gate_waits_for_both_sides_to_be_present_and_still(monkeypatch) -> None:
    left = _side_stats(diff=0.1, presence=0.0)
    right = _side_stats(diff=0.1, presence=0.0)
    triggered: list[str] = []
    now = {"value": 0.0}

    monkeypatch.setattr(
        "app.pipeline.vision.side_camera_gate.time.time",
        lambda: now["value"],
    )

    detector = SideCameraGateDetector(
        runners={
            "left_front": FakeRunner(left),
            "right_front": FakeRunner(right),
        },
        config=SideGateConfig(stable_duration_s=2.5, max_motion_score=0.5, min_presence_score=4.0),
        on_ready=lambda: triggered.append("ready"),
    )

    detector._tick()
    assert detector.snapshot().state == SideGateState.WAITING
    assert triggered == []

    left.update(_side_stats(diff=0.1, presence=7.0))
    right.update(_side_stats(diff=0.1, presence=6.5))
    now["value"] = 1.0
    detector._tick()
    assert detector.snapshot().state == SideGateState.STABILIZING
    assert triggered == []

    now["value"] = 3.7
    detector._tick()
    assert detector.snapshot().state == SideGateState.READY
    assert triggered == []

    now["value"] = 3.8
    detector._tick()
    snap = detector.snapshot()
    assert snap.state == SideGateState.TRIGGERED
    assert snap.trigger_count == 1
    assert triggered == ["ready"]


def test_side_gate_resets_when_any_side_moves_again(monkeypatch) -> None:
    left = _side_stats(diff=0.1, presence=7.0)
    right = _side_stats(diff=0.1, presence=6.5)
    now = {"value": 0.0}

    monkeypatch.setattr(
        "app.pipeline.vision.side_camera_gate.time.time",
        lambda: now["value"],
    )

    detector = SideCameraGateDetector(
        runners={
            "left_front": FakeRunner(left),
            "right_front": FakeRunner(right),
        },
        config=SideGateConfig(stable_duration_s=2.5, max_motion_score=0.5, min_presence_score=4.0),
        on_ready=lambda: None,
    )

    detector._tick()
    assert detector.snapshot().state == SideGateState.STABILIZING

    now["value"] = 1.0
    right.update(_side_stats(diff=1.2, presence=6.5))
    detector._tick()
    snap = detector.snapshot()
    assert snap.state == SideGateState.WAITING
    assert snap.stable_since_ts is None
