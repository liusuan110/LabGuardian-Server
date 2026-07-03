from __future__ import annotations

import pytest


class _FakeTelemetry:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def start(self) -> None:
        self._events.append("telemetry:start")

    async def stop(self) -> None:
        self._events.append("telemetry:stop")


def test_lifespan_autostarts_and_stops_stream(monkeypatch) -> None:
    fastapi = pytest.importorskip("fastapi.testclient")
    TestClient = fastapi.TestClient
    from app import main

    events: list[str] = []
    payloads = []

    monkeypatch.setattr(
        main,
        "get_telemetry_service",
        lambda: _FakeTelemetry(events),
    )

    def _fake_start_stream_on_app_startup():
        payloads.append("startup")
        events.append("stream:start")

    def _fake_stop_stream():
        events.append("stream:stop")
        return {"stopped": True}

    monkeypatch.setattr(
        main.stream,
        "start_stream_on_app_startup",
        _fake_start_stream_on_app_startup,
    )
    monkeypatch.setattr(main.stream, "stop_stream", _fake_stop_stream)

    with TestClient(main.app) as client:
        response = client.get("/health")
        assert response.status_code == 200

    assert payloads == ["startup"]
    assert events == [
        "telemetry:start",
        "stream:start",
        "stream:stop",
        "telemetry:stop",
    ]
