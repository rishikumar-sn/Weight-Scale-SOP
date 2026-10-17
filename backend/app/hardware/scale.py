from __future__ import annotations

import os
import re
import threading
import time
from datetime import datetime
from typing import Any

import serial
import serial.tools.list_ports


WEIGHT_LINE = re.compile(rb"^\s*([-+]?\d+(?:\.\d+)?)\s*(?:g|gm)?\s*$", re.IGNORECASE)


class ScaleService:
    READING_TIMEOUT_SECONDS = 3.0
    RECONNECT_INTERVAL_SECONDS = 0.5
    SERIAL_TIMEOUT_SECONDS = 0.2

    def __init__(self, settings_provider) -> None:
        self._settings_provider = settings_provider
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._weight: float | None = None
        self._reading_time: datetime | None = None
        self._reading_monotonic: float | None = None
        self._port_open = False
        self._raw = ""
        self._port = ""
        self._status = "Scale is starting"
        self._error = ""
        self._discarded = 0

    def _invalidate(self, status: str, error: str = "") -> None:
        with self._lock:
            self._port_open = False
            self._weight = None
            self._reading_time = None
            self._reading_monotonic = None
            self._raw = ""
            self._status = status
            self._error = error

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._invalidate("Scale is starting")
            self._thread = threading.Thread(target=self._run, name="scale-service", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            thread = self._thread
            self._invalidate("Scale stopped")
        if thread and thread.is_alive():
            thread.join(timeout=3)
        with self._lock:
            if self._thread is thread and (thread is None or not thread.is_alive()):
                self._thread = None

    def _select_port(self) -> str:
        configured = str(self._settings_provider()["scale"].get("port") or "").strip()
        # Persistent aliases follow this adapter across ttyUSB number changes.
        # Never switch an explicitly selected device to an unrelated USB port.
        if configured.startswith("/dev/") or (os.name == "nt" and configured):
            return configured
        ports = list(serial.tools.list_ports.comports())
        if configured and any(item.device.casefold() == configured.casefold() for item in ports):
            return configured
        for item in ports:
            if "CH340" in item.description.upper() or "1A86:7523" in item.hwid.upper():
                return item.device
        usb_ports = [
            item for item in ports
            if getattr(item, "vid", None) is not None
            or item.device.startswith(("/dev/ttyUSB", "/dev/ttyACM"))
        ]
        if len(usb_ports) == 1:
            return usb_ports[0].device
        if len(usb_ports) > 1:
            raise RuntimeError("Multiple USB serial devices found; select the scale port in settings")
        raise RuntimeError("USB scale not detected; waiting for connection")

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

    def _read_connection(self, connection, port: str) -> None:
        pending = bytearray()
        dropping = False
        last_valid = time.monotonic()
        while not self._stop.is_set():
            if port.startswith("/dev/") and not os.path.exists(port):
                raise serial.SerialException("USB scale was unplugged; waiting for reconnection")
            chunk = connection.read(min(512, connection.in_waiting or 1))
            if self._stop.is_set():
                return
            for byte in chunk:
                if byte in (10, 13):
                    if dropping:
                        dropping = False
                        pending.clear()
                        continue
                    if not pending:
                        continue
                    raw = bytes(pending)
                    pending.clear()
                    value = self._parse(raw)
                    if value is None:
                        with self._lock:
                            self._discarded += 1
                        continue
                    now = datetime.now().astimezone()
                    last_valid = time.monotonic()
                    with self._lock:
                        self._weight = value
                        self._reading_time = now
                        self._reading_monotonic = last_valid
                        self._raw = raw.decode("ascii", errors="ignore").strip()
                        self._status = "Scale connected"
                        self._error = ""
                elif not dropping:
                    if len(pending) >= 64:
                        pending.clear()
                        dropping = True
                        with self._lock:
                            self._discarded += 1
                    else:
                        pending.append(byte)
            if time.monotonic() - last_valid >= self.READING_TIMEOUT_SECONDS:
                raise serial.SerialException("No fresh scale readings; reopening USB connection")

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    port = self._select_port()
                    baud = int(self._settings_provider()["scale"]["baud_rate"])
                    with self._lock:
                        self._port = port
                    options = {"exclusive": True} if os.name != "nt" else {}
                    with serial.Serial(port, baud, timeout=self.SERIAL_TIMEOUT_SECONDS, **options) as connection:
                        connection.reset_input_buffer()
                        with self._lock:
                            self._port_open = True
                            self._status = "Scale waiting for data"
                            self._error = ""
                        self._read_connection(connection, port)
                except Exception as exc:  # noqa: BLE001
                    self._invalidate("Scale reconnecting", str(exc))
                    self._stop.wait(self.RECONNECT_INTERVAL_SECONDS)
                finally:
                    # An open port alone is not a live measurement. Invalidate
                    # before another connection can receive its first packet.
                    self._invalidate("Scale stopped" if self._stop.is_set() else "Scale reconnecting", self._error)
        finally:
            self._invalidate("Scale stopped")

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            age_ms = None
            if self._reading_monotonic is not None:
                age_ms = max(0, int((time.monotonic() - self._reading_monotonic) * 1000))
            fresh = self._port_open and age_ms is not None and age_ms < self.READING_TIMEOUT_SECONDS * 1000
            status = self._status
            if self._port_open and not fresh:
                status = "Scale waiting for data"
            return {
                "connected": fresh,
                "status": status,
                "error": self._error,
                "port": self._port,
                "weight_g": self._weight if fresh else None,
                "reading_at": self._reading_time.isoformat(timespec="milliseconds") if fresh and self._reading_time else None,
                "age_ms": age_ms,
                "raw": self._raw if fresh else "",
                "discarded_frames": self._discarded,
            }
