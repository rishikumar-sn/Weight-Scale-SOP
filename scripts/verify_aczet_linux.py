"""Exercise Linux serial I/O and button-event pairing without physical hardware."""
from __future__ import annotations

import os
import pty
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
from unittest.mock import patch

from offline_runtime import configure


def wait_for(predicate, timeout=4):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError('Timed out waiting for serial event')


def main():
    configure()
    import numpy as np
    from backend.app.core.config import SettingsStore
    from backend.app.hardware.scale import ScaleService
    from backend.app.services.capture_service import CaptureService

    with tempfile.TemporaryDirectory() as temp:
        store = SettingsStore(Path(temp)/'settings.json', platform='linux')
        store.select_scale('accet_cg1003l')
        assert store.get()['scale']['baud_rate'] == 9600
        # Simulate renumbering: the physical identity outranks a stale port.
        store.update({'scale': {'model': 'embsys_wm', 'port': '/dev/ttyUSB0', 'usb_identity': '1027:24577:scale'}})
        wrong = SimpleNamespace(device='/dev/ttyUSB0', vid=1027, pid=24577, serial_number='other')
        right = SimpleNamespace(device='/dev/ttyUSB3', vid=1027, pid=24577, serial_number='scale')
        with patch('serial.tools.list_ports.comports', return_value=[wrong, right]):
            assert ScaleService(store.get)._select_port() == right.device
        print('PASS: numbered Linux ports follow saved USB identity', flush=True)

    masters, slaves = zip(pty.openpty(), pty.openpty())
    ports = [SimpleNamespace(device=os.ttyname(fd), vid=1027, pid=24577, serial_number=str(fd)) for fd in slaves]
    settings = {'scale': {'model': 'accet_cg1003l', 'port': 'COM99', 'baud_rate': 9600}}
    scale = ScaleService(lambda: settings)
    capture = CaptureService.__new__(CaptureService)
    capture.settings_store = SimpleNamespace(get=lambda: settings)
    capture._event_lock = RLock()
    capture._scale_events = OrderedDict()
    pixels = np.full((8, 8, 3), 42, dtype=np.uint8)
    calls = []

    def frame():
        calls.append(True)
        return pixels.copy(), {'connected': True}

    capture.camera = SimpleNamespace(snapshot_frame=frame)
    capture.scale = scale
    scale.on_reading = capture.record_scale_reading
    try:
        with patch('serial.tools.list_ports.comports', side_effect=lambda: list(ports)):
            scale.start()
            wait_for(lambda: ports[1].device in scale.snapshot()['port'])
            assert not calls, 'Silence must not capture a frame'
            os.write(masters[0], b'noise\r\n')
            os.write(masters[1], b'ST,GS,+ 12.')
            time.sleep(.05)
            os.write(masters[1], b'345 g\r\n')
            wait_for(lambda: scale.snapshot()['capture_event_id'] == 1)
            assert scale.snapshot()['weight_g'] == 12.345
            pixels[:] = 99
            frozen, _, reading = capture._capture_inputs(1)
            assert np.all(frozen == 42) and reading['weight_g'] == 12.345
            try:
                capture._capture_inputs(1)
            except RuntimeError:
                pass
            else:
                raise AssertionError('Duplicate event was accepted')
            time.sleep(3.2)
            assert scale.snapshot()['connected'] and scale.snapshot()['weight_g'] is None
            assert len(calls) == 1, 'Silence recaptured the image'
            os.write(masters[1], b'ST,NT,+ 1.234 ct\r\n')
            wait_for(lambda: scale.snapshot()['capture_event_id'] == 2)
            assert len(calls) == 1, 'Non-gram data captured pixels'
            try:
                capture._capture_inputs(2)
            except RuntimeError as exc:
                assert 'grams' in str(exc)
            else:
                raise AssertionError('Non-gram capture was accepted')
            # Reconnect to the remaining candidate and preserve its first packet.
            ports.pop(1)
            scale.request_reconnect()
            wait_for(lambda: scale.snapshot()['port'] == ports[0].device)
            os.write(masters[0], b'ST,GS,- 0.125 g\r\n')
            wait_for(lambda: scale.snapshot()['capture_event_id'] == 3)
            assert scale.snapshot()['weight_g'] == -.125
            assert len(calls) == 2
            print('PASS: split packets, first-button frame pairing, duplicate prevention, silent connection, grams-only capture and port rediscovery', flush=True)
    finally:
        scale.stop()
        for fd in (*masters, *slaves):
            os.close(fd)

    from backend.app.analysis.testbed_segmentation import TestbedSegmenter
    from backend.app.core.config import TESTBED_MODEL_PATHS
    segmenter = TestbedSegmenter(TESTBED_MODEL_PATHS['accet_cg1003l'], exclude_circle=False, outer_inset_ratio=.035)
    assert segmenter.circle_class_id is None
    print('PASS: ACZET testbed ONNX loads with circle exclusion disabled', flush=True)


if __name__ == '__main__':
    main()
