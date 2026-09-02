from __future__ import annotations

import re
import threading
from datetime import datetime
from typing import Any

import serial
import serial.tools.list_ports


WEIGHT_LINE = re.compile(rb"^\s*([-+]?\d+(?:\.\d+)?)\s*(?:g|gm)?\s*$", re.IGNORECASE)


class ScaleService:
    def __init__(self, settings_provider) -> None:
        self._settings_provider = settings_provider
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._weight: float | None = None
        self._reading_time: datetime | None = None
        self._raw = ""
        self._port = ""
        self._status = "Scale is starting"
        self._error = ""
        self._discarded = 0

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="scale-service", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=3)
        with self._lock:
            if self._thread is thread and (thread is None or not thread.is_alive()):
                self._thread = None

    def _select_port(self) -> str:
        configured = str(self._settings_provider()["scale"].get("port") or "").strip()
        ports = list(serial.tools.list_ports.comports())
        if configured and any(item.device.casefold() == configured.casefold() for item in ports):
            return configured
        for item in ports:
            if "CH340" in item.description.upper() or "1A86:7523" in item.hwid.upper():
                return item.device
        if ports:
            return ports[0].device
        raise RuntimeError("No serial scale port was found")

    @staticmethod
    def _parse(raw: bytes) -> float | None:
        if not raw or len(raw) > 64:
            return None
        if any(byte not in b"\t\r\n +-.0123456789gGmM" for byte in raw):
            return None
        match = WEIGHT_LINE.fullmatch(raw)
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                port = self._select_port()
                baud = int(self._settings_provider()["scale"]["baud_rate"])
                with serial.Serial(port, baud, timeout=0.3) as connection:
                    connection.reset_input_buffer()
                    with self._lock:
                        self._port = port
                        self._status = "Scale connected"
                        self._error = ""
                    while not self._stop.is_set():
                        raw = connection.readline()
                        if not raw:
                            continue
                        value = self._parse(raw)
                        if value is None:
                            with self._lock:
                                self._discarded += 1
                            if len(raw) > 1024:
                                connection.reset_input_buffer()
                            continue
                        now = datetime.now().astimezone()
                        with self._lock:
                            self._weight = value
                            self._reading_time = now
                            self._raw = raw.decode("ascii", errors="ignore").strip()
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    if not self._stop.is_set():
                        self._status = "Scale reconnecting"
                        self._error = str(exc)
                self._stop.wait(1)
        with self._lock:
            self._status = "Scale stopped"

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            age_ms = None
            if self._reading_time:
                age_ms = max(0, int((datetime.now().astimezone() - self._reading_time).total_seconds() * 1000))
            return {
                "connected": bool(self._port) and age_ms is not None and age_ms < 3000,
                "status": self._status,
                "error": self._error,
                "port": self._port,
                "weight_g": self._weight,
                "reading_at": self._reading_time.isoformat(timespec="milliseconds") if self._reading_time else None,
                "age_ms": age_ms,
                "raw": self._raw,
                "discarded_frames": self._discarded,
            }
