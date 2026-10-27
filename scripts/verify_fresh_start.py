"""Check restart behavior using temporary data and mocked camera/model startup."""
from __future__ import annotations

import asyncio
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from backend.app.core import config
    from backend.app.storage.repository import CaptureRepository

    with tempfile.TemporaryDirectory(prefix="weightscale-restart-") as directory:
        root = Path(directory)
        database = root / "app.db"
        sessions = root / "sessions"
        repository = CaptureRepository(database, sessions)
        # Exercise each state that could have been saved before a power loss.
        for index, status in enumerate(("captured", "awaiting_confirmation", "processing", "complete", "failed", "rejected")):
            state = {"id": f"old-{index}", "captured_at": f"2026-10-23T10:00:0{index}", "status": status}
            repository.save(state)
        history = deepcopy(repository.list_recent())
        restarted = CaptureRepository(database, sessions)
        assert restarted.latest_in_run() is None
        assert restarted.list_recent() == history
        assert restarted.get("old-1")["status"] == "awaiting_confirmation"
        assert restarted.run_id != repository.run_id

        settings = config.SettingsStore(root / "settings.json")
        with patch.multiple(config, DATABASE_PATH=database, SESSIONS_DIR=sessions), patch.object(config, "SettingsStore", return_value=settings):
            from backend.app import main as backend

        async def exercise():
            with patch.object(backend.analysis, "preload_models"), patch.object(backend.analysis, "shutdown"), patch.object(backend.captures.packet_scanner, "preload"), patch.object(backend.camera, "start"), patch.object(backend.camera, "stop"), patch.object(backend.scale, "start"), patch.object(backend.scale, "stop"), patch.object(backend, "_public_capture", side_effect=lambda state: state):
                async with backend.lifespan(backend.app):
                    fresh = backend.live_state()
                    assert fresh["latest"] is None
                    run_id = fresh["run_id"]
                    new = {"id": "new", "captured_at": "2026-10-24T10:00:00", "status": "captured"}
                    backend.repository.save(new)
                    new["status"] = "awaiting_confirmation"
                    backend.repository.save(new)
                    assert backend.live_state()["latest"] == new
                    # Updating an older saved record must not replace the live capture.
                    old = backend.repository.get("old-1")
                    backend.repository.save(old)
                    assert backend.live_state()["latest"] == new
                    # WebSocket and HTTP must agree about the active run.
                    from fastapi import WebSocketDisconnect
                    socket = Mock()
                    async def accept():
                        pass
                    async def send(payload):
                        assert payload == backend.live_state()
                        raise WebSocketDisconnect()
                    socket.accept = accept
                    socket.send_json = send
                    await backend.live_socket(socket)
                async with backend.lifespan(backend.app):
                    assert backend.live_state()["latest"] is None
                    assert backend.live_state()["run_id"] != run_id
                    assert backend.repository.get("new") == new
                    assert backend.repository.list_recent()[1:] == history
        asyncio.run(exercise())
    print("PASS: fresh startup, pending/completed history preserved, current-run updates, HTTP/WebSocket consistency, repeated restart.")


if __name__ == "__main__":
    main()
