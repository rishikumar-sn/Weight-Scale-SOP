from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import serial

from backend.app.core.config import SettingsStore
from backend.app.hardware.scale import ScaleService
from backend.app.services.capture_service import CaptureService


def usb(device, number="scale-123"):
    return SimpleNamespace(device=device, vid=0x1A86, pid=0x7523,
                           serial_number=number, description="USB serial", hwid="")


@pytest.mark.parametrize("platform", ["windows", "linux"])
def test_model_persistence_and_original_device_restored(tmp_path, platform):
    path = tmp_path / "settings.json"
    store = SettingsStore(path, platform=platform)
    original = store.get()["scale"]
    selected = store.select_scale("essae_jx500")["scale"]
    assert selected["baud_rate"] == 115200
    port = "COM19" if platform == "windows" else "/dev/ttyACM2"
    store.remember_scale_device("essae_jx500", port, "usb-identity")
    store = SettingsStore(path, platform=platform)
    assert store.get()["scale"]["model"] == "essae_jx500"
    restored = store.select_scale("embsys_wm")["scale"]
    assert restored["port"] == original["port"]
    assert restored["baud_rate"] == 9600
    selected = store.select_scale("essae_jx500")["scale"]
    assert selected["port"] == port
    assert selected["usb_identity"] == "usb-identity"


@pytest.mark.parametrize("device", ["COM24", "/dev/ttyACM3"])
def test_identity_follows_port_changes_and_wrong_devices_are_skipped(device):
    remembered = usb(device)
    unrelated = usb("COM11", "other-device")
    settings = {"scale": {"model": "essae_jx500", "port": "COM11",
                          "usb_identity": ScaleService._port_identity(remembered)}}
    service = ScaleService(lambda: settings)
    with patch("serial.tools.list_ports.comports", return_value=[unrelated, remembered]):
        assert service._select_port() == device
        service._failed_ports.add(device)
        assert service._select_port() == unrelated.device


def feed(service, chunks):
    service._model = "essae_jx500"
    service._port_open = True
    connection = Mock(in_waiting=512)
    def read(_):
        if chunks:
            return chunks.pop(0)
        raise serial.SerialException("unplugged")
    connection.read.side_effect = read
    with pytest.raises(serial.SerialException, match="unplugged"):
        service._read_connection(connection, "COM11")
    return service.snapshot()


def test_fragmented_debug_stream_displays_ct_without_exposing_grams():
    service = ScaleService(lambda: {"scale": {"model": "essae_jx500"}})
    snapshot = feed(service, [b"Booting decoder\r\nDEBUG bad frame\nWEI", b"GHT=97.645 g\nWEIGHT=", b"6.145 ct\r\n"])
    assert snapshot["connected"]
    assert snapshot["value"] == 6.145
    assert snapshot["unit"] == "ct"
    assert snapshot["weight_g"] is None
    assert snapshot["capture_allowed"] is False
    snapshot = feed(service, [b"debug WEIGHT=0.240 G\n"])
    assert snapshot["weight_g"] == 0.240
    assert snapshot["capture_allowed"] is True
    service.request_reconnect()
    assert service.snapshot()["value"] is None
    assert not service.snapshot()["connected"]


@pytest.mark.parametrize("method,args", [("capture", ()), ("capture_tare", ("pledge",)), ("capture_tare", ("release",))])
def test_capture_rejects_ct_before_saving(method, args):
    camera = Mock()
    camera.snapshot_frame.return_value = (SimpleNamespace(shape=(100, 100, 3)), {})
    scale = Mock()
    scale.snapshot.return_value = {"connected": True, "unit": "ct", "weight_g": 123, "age_ms": 0}
    repository = Mock()
    settings = Mock()
    settings.get.return_value = {"rois": {"apriltag": None}, "apriltag": {
        "id": 1, "width_mm": 10, "height_mm": 10,
    }}
    service = CaptureService(camera, scale, settings, repository)
    with patch("backend.app.services.capture_service.detect_apriltag", return_value={}), \
         patch("backend.app.services.capture_service.apriltag_roi_from_detection", return_value=None), \
         pytest.raises(RuntimeError, match="ct capture is not allowed"):
        getattr(service, method)(*args)
    repository.session_dir.assert_not_called()


@pytest.mark.parametrize("model,baud", [("embsys_wm", 9600), ("essae_jx500", 115200)])
def test_worker_reconnects_after_unplugging_at_a_new_port(model, baud):
    settings = {"scale": {"model": model, "port": "COM11", "baud_rate": baud}}
    service = ScaleService(lambda: settings)
    service.RECONNECT_INTERVAL_SECONDS = 0.001
    ports = [usb("COM11")]
    first, second = Mock(), Mock()
    first.__enter__ = Mock(return_value=first)
    first.__exit__ = Mock(return_value=False)
    second.__enter__ = Mock(return_value=second)
    second.__exit__ = Mock(return_value=False)
    first.in_waiting = second.in_waiting = 100
    def unplug(_):
        ports[:] = [usb("COM27")]
        raise serial.SerialException("unplugged")
    first.read.side_effect = unplug
    def finish(_):
        service._stop.set()
        return b""
    second.read.side_effect = finish
    with patch("backend.app.hardware.scale.os", SimpleNamespace(name="nt")), \
         patch("serial.tools.list_ports.comports", side_effect=lambda: list(ports)), \
         patch("serial.Serial", side_effect=[first, second]) as open_serial:
        service._run()
    assert [call.args[:2] for call in open_serial.call_args_list] == [("COM11", baud), ("COM27", baud)]
    first.__exit__.assert_called_once()
    assert not service.snapshot()["connected"]


def test_embsys_remembers_device_only_after_valid_reading():
    settings = {"scale": {"model": "embsys_wm", "port": "COM12"}}
    remember = Mock()
    service = ScaleService(lambda: settings, remember)
    service._failed_ports.add("COM12")
    connection = Mock(in_waiting=100)

    def read(_):
        if connection.read.call_count == 1:
            return b"invalid data\r\n"
        if connection.read.call_count == 2:
            remember.assert_not_called()
            return b"12.345 g\r\n"
        raise serial.SerialException("unplugged")

    connection.read.side_effect = read
    device = usb("COM27")
    with patch("backend.app.hardware.scale.os", SimpleNamespace(name="nt")), \
         patch("serial.tools.list_ports.comports", return_value=[device]), \
         pytest.raises(serial.SerialException, match="unplugged"):
        service._read_connection(connection, "COM27")
    remember.assert_called_once_with("embsys_wm", "COM27", ScaleService._port_identity(device))
    assert not service._failed_ports


@pytest.mark.parametrize("failure", ["busy", "silent", "invalid"])
def test_embsys_worker_skips_unusable_saved_port(failure):
    service = ScaleService(lambda: {"scale": {
        "model": "embsys_wm", "port": "COM12", "baud_rate": 9600,
    }})
    service.RECONNECT_INTERVAL_SECONDS = 0.001
    first, second = Mock(), Mock()
    for connection in (first, second):
        connection.__enter__ = Mock(return_value=connection)
        connection.__exit__ = Mock(return_value=False)
        connection.in_waiting = 100
    first.read.return_value = b"wrong device\r\n" if failure == "invalid" else b""

    def finish(_):
        service._stop.set()
        return b""

    second.read.side_effect = finish
    # Advance time on each read so silent/invalid devices reach the timeout.
    ticks = iter(range(100))
    with patch("backend.app.hardware.scale.os", SimpleNamespace(name="nt")), \
         patch("backend.app.hardware.scale.time.monotonic", side_effect=lambda: next(ticks)), \
         patch("serial.tools.list_ports.comports", return_value=[usb("COM12"), usb("COM27")]), \
         patch("serial.Serial", side_effect=[serial.SerialException("busy") if failure == "busy" else first, second]) as opened:
        service._run()
    assert [call.args[:2] for call in opened.call_args_list] == [("COM12", 9600), ("COM27", 9600)]
    if failure != "busy":
        first.__exit__.assert_called_once()


def test_apply_switches_running_reader_to_selected_model_and_baud():
    settings = {"scale": {"model": "embsys_wm", "port": "COM12", "baud_rate": 9600}}
    service = ScaleService(lambda: settings)
    observed = []

    def read(connection, port):
        observed.append((service._model, port))
        if len(observed) == 1:
            settings["scale"] = {"model": "essae_jx500", "port": "COM11", "baud_rate": 115200}
            service.request_reconnect()
        else:
            service._stop.set()

    connection = Mock()
    connection.__enter__ = Mock(return_value=connection)
    connection.__exit__ = Mock(return_value=False)
    with patch("serial.tools.list_ports.comports", return_value=[usb("COM11"), usb("COM12")]), \
         patch("serial.Serial", return_value=connection) as open_serial, \
         patch.object(service, "_read_connection", side_effect=read):
        service._run()
    assert observed == [("embsys_wm", "COM12"), ("essae_jx500", "COM11")]
    assert [call.args[:2] for call in open_serial.call_args_list] == [("COM12", 9600), ("COM11", 115200)]


def test_oversized_debug_frame_recovers_and_stale_readings_block_capture():
    service = ScaleService(lambda: {"scale": {"model": "essae_jx500"}})
    snapshot = feed(service, [b"x" * 9000 + b"\nWEIGHT=2.500 g\n"])
    assert snapshot["discarded_frames"] == 1
    assert snapshot["weight_g"] == 2.5
    service._reading_monotonic -= 4
    snapshot = service.snapshot()
    assert not snapshot["capture_allowed"]
    assert snapshot["value"] is None
    assert snapshot["weight_g"] is None
