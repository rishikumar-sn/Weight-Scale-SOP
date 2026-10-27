from types import SimpleNamespace
from unittest.mock import patch

import pytest

from backend.app.hardware.scale import ScaleService


def port(device, description="Serial device", hwid="", **metadata):
    return SimpleNamespace(device=device, description=description, hwid=hwid, **metadata)


@pytest.mark.parametrize("configured", ["/dev/ttyUSB0", "/dev/serial/by-id/usb-scale"])
def test_linux_explicit_device_survives_disconnect(configured):
    service = ScaleService(lambda: {"scale": {"port": configured}})
    with patch("backend.app.hardware.scale.os", SimpleNamespace(name="posix")), patch(
        "serial.tools.list_ports.comports", return_value=[port("/dev/ttyUSB1")],
    ):
        assert service._select_port() == configured


def test_copied_windows_setting_finds_linux_ch340():
    service = ScaleService(lambda: {"scale": {"port": "COM12"}})
    with patch("backend.app.hardware.scale.os", SimpleNamespace(name="posix")), patch("serial.tools.list_ports.comports", return_value=[
        port("/dev/ttyS0"), port("/dev/ttyUSB0", hwid="USB VID:PID=1A86:7523"),
    ]):
        assert service._select_port() == "/dev/ttyUSB0"


def test_windows_explicit_port_still_takes_priority():
    service = ScaleService(lambda: {"scale": {"port": "COM12"}})
    with patch("backend.app.hardware.scale.os", SimpleNamespace(name="nt")), patch("serial.tools.list_ports.comports", return_value=[
        port("COM3", description="CH340"), port("COM12"),
    ]):
        assert service._select_port() == "COM12"


@pytest.mark.parametrize("configured", ["", "COM12", "/dev/ttyUSB0"])
def test_windows_discovers_scale_instead_of_missing_saved_port(configured):
    service = ScaleService(lambda: {"scale": {"port": configured}})
    with patch("backend.app.hardware.scale.os", SimpleNamespace(name="nt")), patch(
        "serial.tools.list_ports.comports", return_value=[
            port("COM1"), port("COM27", vid=0x1A86, pid=0x7523),
        ],
    ):
        assert service._select_port() == "COM27"


def test_windows_remembered_identity_wins_over_reused_com_number():
    scale = port("COM27", vid=0x1A86, pid=0x7523, serial_number="scale")
    service = ScaleService(lambda: {"scale": {
        "port": "COM12", "usb_identity": ScaleService._port_identity(scale),
    }})
    with patch("backend.app.hardware.scale.os", SimpleNamespace(name="nt")), patch(
        "serial.tools.list_ports.comports", return_value=[port("COM12"), scale],
    ):
        assert service._select_port() == "COM27"


def test_windows_retries_other_candidates_and_restarts_scan():
    service = ScaleService(lambda: {"scale": {"port": "COM12"}})
    with patch("backend.app.hardware.scale.os", SimpleNamespace(name="nt")), patch(
        "serial.tools.list_ports.comports", return_value=[
            port("COM12"), port("COM27", description="CH340"),
        ],
    ):
        assert service._select_port() == "COM12"
        service._failed_ports.add("COM12")
        assert service._select_port() == "COM27"
        service._failed_ports.add("COM27")
        assert service._select_port() == "COM12"


def test_windows_waits_for_device_then_discovers_new_port():
    service = ScaleService(lambda: {"scale": {"port": "COM12"}})
    with patch("backend.app.hardware.scale.os", SimpleNamespace(name="nt")), patch(
        "serial.tools.list_ports.comports", side_effect=[[], [port("COM8")]],
    ):
        with pytest.raises(RuntimeError, match="waiting for connection"):
            service._select_port()
        assert service._select_port() == "COM8"
