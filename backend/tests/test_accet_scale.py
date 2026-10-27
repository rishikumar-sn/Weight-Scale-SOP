from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest
import serial

from backend.app.core.config import SettingsStore, TESTBED_MODEL_PATHS
from backend.app.hardware.scale import ScaleService
from backend.app.services.capture_service import CaptureService
from backend.tests.test_testbed_segmentation import essae_segmenter
from backend.app.analysis.vision import (
    apriltag_roi_from_detection, build_jewellery_mask, detect_apriltag,
    marker_ignore_mask, separate_jewellery_items, translate_roi_to_crop,
)


@pytest.mark.parametrize('packet,value,unit,grams', [
    (b'US,GS,+   2.972g', 2.972, 'g', 2.972),
    (b'US,GS,-   0.538g', -.538, 'g', -.538),
    (b'ST,NT,+   0.321kg', .321, 'kg', None),
    (b'US,GS,+   300mg', 300, 'mg', None),
    (b'US,GS,+   2.5ct', 2.5, 'ct', None),
    (b'US,GS,+   120gsm', 120, 'gsm', None),
])
def test_fragmented_packets_preserve_units_and_freeze_each_reading(packet, value, unit, grams):
    service = ScaleService(lambda: {'scale': {'model': 'accet_cg1003l'}})
    service._model = 'accet_cg1003l'
    service._port_open = True
    service.on_reading = Mock()
    connection = Mock(in_waiting=100)
    connection.read.side_effect = [b'noise\r\n' + packet[:7], packet[7:] + b'\r\n', serial.SerialException('unplugged')]
    with pytest.raises(serial.SerialException):
        service._read_connection(connection, 'COM11')
    snapshot = service.snapshot()
    assert snapshot['value'] == value
    assert snapshot['unit'] == unit
    assert snapshot['weight_g'] == grams
    assert snapshot['capture_allowed'] == (grams is not None)
    assert snapshot['capture_event_id'] == 1
    service.on_reading.assert_called_once()
    service._reading_monotonic -= 60
    assert service.snapshot()['connected']
    assert service.snapshot()['weight_g'] is None


def test_silent_accet_stays_open_after_first_packet():
    service = ScaleService(lambda: {'scale': {'model': 'accet_cg1003l'}})
    service._model = 'accet_cg1003l'
    connection = Mock(in_waiting=100)
    connection.read.side_effect = [b'US,GS,+ 1.2g\n', b'', serial.SerialException('unplugged')]
    with patch('backend.app.hardware.scale.time.monotonic', side_effect=[0, 0, 100, 200]), pytest.raises(serial.SerialException, match='unplugged'):
        service._read_connection(connection, 'COM11')
    assert connection.read.call_count == 3


def test_auto_detection_listens_past_silent_wrong_port_and_closes_it():
    service = ScaleService(lambda: {'scale': {'model': 'accet_cg1003l', 'port': 'COM11'}})
    wrong, actual = Mock(), Mock()
    wrong.in_waiting = 0
    wrong.read.return_value = b''
    actual.in_waiting = 100
    actual.read.side_effect = [b'US,GS,+ ', b'2.972g\r\n']
    ports = [SimpleNamespace(device='COM11', vid=1), SimpleNamespace(device='COM27', vid=2)]
    with patch('serial.tools.list_ports.comports', return_value=ports), patch('serial.Serial', side_effect=[wrong, actual]) as opened:
        with service._accet_connection() as (connection, port, initial):
            assert connection is actual
            assert port == 'COM27'
            assert initial.startswith(b'US,GS,+ 2.972g\n')
            wrong.close.assert_called_once()
        actual.close.assert_called_once()
    assert [call.args[:2] for call in opened.call_args_list] == [('COM11', 9600), ('COM27', 9600)]


def capture_service(tmp_path):
    settings = SettingsStore(tmp_path / 'settings.json', platform='windows')
    settings.select_scale('accet_cg1003l')
    camera, scale, repository = Mock(), Mock(), Mock()
    camera.snapshot_frame.return_value = (np.zeros((100, 100, 3), np.uint8), {'sequence': 5})
    service = CaptureService(camera, scale, settings, repository)
    reading = {'reading_id': 1, 'model': 'accet_cg1003l', 'connected': True,
               'weight_g': 2.972, 'value': 2.972, 'unit': 'g', 'age_ms': 0}
    service.record_scale_reading(reading)
    return service


def test_capture_uses_event_frame_and_weight_once_and_blocks_manual_capture(tmp_path):
    service = capture_service(tmp_path)
    frame, camera, reading = service._capture_inputs(1)
    assert camera['sequence'] == 5
    assert reading['weight_g'] == 2.972
    service.camera.snapshot_frame.assert_called_once()
    service.scale.snapshot.assert_not_called()
    with pytest.raises(RuntimeError, match='already captured'):
        service._capture_inputs(1)
    with pytest.raises(RuntimeError, match='Press the ACCET'):
        service.capture()


@pytest.mark.parametrize('tare', [False, True])
def test_apriltag_is_mandatory_on_every_button_capture(tmp_path, tare):
    service = capture_service(tmp_path)
    with patch('backend.app.services.capture_service.detect_apriltag', side_effect=RuntimeError('AprilTag missing')), pytest.raises(RuntimeError):
        if tare:
            service.capture_tare('pledge', reading_id=1)
        else:
            service.capture(reading_id=1)
    service.repository.session_dir.assert_not_called()
    service.repository.save.assert_not_called()


def test_expired_event_and_non_gram_unit_cannot_be_saved(tmp_path):
    service = capture_service(tmp_path)
    with patch('backend.app.services.capture_service.time.monotonic', return_value=10**15), pytest.raises(RuntimeError, match='expired'):
        service._capture_inputs(1)
    service.record_scale_reading({'reading_id': 2, 'unit': 'gsm', 'weight_g': None})
    with pytest.raises(RuntimeError, match='Change the weight scale mode to grams'):
        service._capture_inputs(2)


@pytest.mark.parametrize('unit', ['kg', 'mg', 'ct', 'gsm', 'gm', 'oz', None])
@pytest.mark.parametrize('tare', [False, True])
def test_non_gram_packet_never_takes_photo_or_saves_capture(tmp_path, unit, tare):
    service = capture_service(tmp_path)
    service.camera.snapshot_frame.reset_mock()
    # Even an incorrectly populated weight_g must not bypass unit validation.
    service.record_scale_reading({'reading_id': 2, 'unit': unit, 'weight_g': 123})
    with pytest.raises(RuntimeError, match='Change the weight scale mode to grams'):
        if tare:
            service.capture_tare('pledge', reading_id=2)
        else:
            service.capture(reading_id=2)
    service.camera.snapshot_frame.assert_not_called()
    service.repository.save.assert_not_called()


def test_accet_testbed_does_not_require_or_remove_circle():
    segmenter = essae_segmenter(circle_score=0)
    segmenter.circle_class_id = None
    result = segmenter.segment(np.zeros((100, 100, 3), np.uint8))
    assert np.any(result.source_mask)
    assert result.hole_validation is None


def test_accet_settings_survive_switching_and_use_shared_model(tmp_path):
    store = SettingsStore(tmp_path / 'settings.json', platform='windows')
    settings = store.select_scale('accet_cg1003l')['scale']
    assert settings['port'] == 'COM11'
    assert settings['baud_rate'] == 9600
    store.remember_scale_device('accet_cg1003l', 'COM27', 'accet-usb')
    store.select_scale('essae_jx500')
    store = SettingsStore(tmp_path / 'settings.json', platform='windows')
    settings = store.select_scale('accet_cg1003l')['scale']
    assert settings['port'] == 'COM27'
    assert settings['usb_identity'] == 'accet-usb'
    assert settings['baud_rate'] == 9600
    assert TESTBED_MODEL_PATHS['accet_cg1003l'] == TESTBED_MODEL_PATHS['essae_jx500']


@pytest.mark.parametrize('scale', [0.5, 1])
def test_actual_accet_tray_rim_is_excluded_without_losing_necklace(tmp_path, scale):
    image = cv2.imread(str(Path(__file__).parent / 'fixtures' / 'accet_chain_tray.png'))
    if scale != 1:
        image = cv2.resize(image, None, fx=scale, fy=scale)
    tag = detect_apriltag(image, None, 1, 10, 10)
    service = capture_service(tmp_path)
    segmenter = service._get_testbed_segmenter()
    assert segmenter.circle_class_id is None

    def items_with_segmenter():
        tray = segmenter.segment(image, apriltag=tag)
        marker = translate_roi_to_crop(apriltag_roi_from_detection(image.shape, tag),
            tray.processing_roi, tray.image.shape[1], tray.image.shape[0])
        ignore = marker_ignore_mask(tray.image.shape, marker)
        working = tray.image.copy()
        working[ignore > 0] = 255
        ignore = cv2.bitwise_or(ignore, (tray.mask == 0).astype(np.uint8) * 255)
        _, mask = build_jewellery_mask(working, ignore)
        return separate_jewellery_items(working, mask)

    corrected = items_with_segmenter()
    assert len(corrected) == 1
    # Reproduce the former extra component from the rim with the old margin.
    segmenter.outer_inset_ratio = 0
    old = items_with_segmenter()
    assert len(old) >= len(corrected)
    if scale == 1:
        assert len(old) == 2
    necklace_area = max(item['area_px'] for item in old)
    assert corrected[0]['area_px'] == pytest.approx(necklace_area, rel=.01)
