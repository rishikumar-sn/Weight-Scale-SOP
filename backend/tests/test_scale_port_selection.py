from types import SimpleNamespace
from unittest.mock import patch

import pytest

from backend.app.hardware.scale import ScaleService


def port(device, description="Serial device", hwid=""):
    return SimpleNamespace(device=device, description=description, hwid=hwid)


@pytest.mark.parametrize("configured", ["/dev/ttyUSB0", "/dev/serial/by-id/usb-scale"])
def test_linux_explicit_device_survives_disconnect(configured):
    service = ScaleService(lambda: {"scale": {"port": configured}})
    with patch("serial.tools.list_ports.comports", return_value=[port("/dev/ttyUSB1")]):
        assert service._select_port() == configured


def test_copied_windows_setting_finds_linux_ch340():
    service = ScaleService(lambda: {"scale": {"port": "COM12"}})
    with patch("backend.app.hardware.scale.os", SimpleNamespace(name="posix")), patch("serial.tools.list_ports.comports", return_value=[
        port("/dev/ttyS0"), port("/dev/ttyUSB0", hwid="USB VID:PID=1A86:7523"),
    ]):
        assert service._select_port() == "/dev/ttyUSB0"


def test_windows_explicit_port_still_takes_priority():
    service = ScaleService(lambda: {"scale": {"port": "COM12"}})
    with patch("serial.tools.list_ports.comports", return_value=[
        port("COM3", description="CH340"), port("COM12"),
    ]):
        assert service._select_port() == "COM12"
