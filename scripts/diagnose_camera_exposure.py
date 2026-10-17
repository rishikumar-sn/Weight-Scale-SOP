"""Controlled Linux UVC comparison; stop other camera apps before running.

Run with .venv/bin/python scripts/diagnose_camera_exposure.py. Controls and
stream format are restored in finally. Images are camera-decoded BGR, with no
application white balance, rotation, tone correction, or temporal filtering.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import time

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='/dev/video0')
    parser.add_argument('--output', type=Path, default=Path('data/camera-diagnostics/research-20261016'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    def ctl(*options):
        return subprocess.run(['v4l2-ctl', '-d', args.device, *options],
                              check=True, capture_output=True, text=True, timeout=5).stdout

    def controls():
        listing = ctl('--list-ctrls')
        return {m[0]: int(m[1]) for m in re.findall(
            r'^\s*(\w+)\s+0x[^\n]*?\bvalue=(-?\d+)', listing, re.M)}

    def set_controls(values):
        for key, value in values.items():
            ctl('-c', f'{key}={value}')

    before = controls()
    (args.output / 'device-before.txt').write_text(ctl('--all', '--list-ctrls-menus'))
    fmt = ctl('--get-fmt-video')
    width, height = map(int, re.search(r'Width/Height\s*:\s*(\d+)/(\d+)', fmt).groups())
    codec = re.search(r"Pixel Format\s*:\s*'([^']+)'", fmt)[1]
    fps = float(re.search(r'Frames per second:\s*([\d.]+)', ctl('--get-parm'))[1])
    cap = None
    report = {'before': before, 'stream_before': {'width': width, 'height': height, 'codec': codec, 'fps': fps},
              'cases': [], 'restoration_errors': []}
    auto = next(k for k in ('exposure_auto', 'auto_exposure') if k in before)
    shutter = next(k for k in ('exposure_absolute', 'exposure_time_absolute') if k in before)
    awb = next(k for k in ('white_balance_temperature_auto', 'white_balance_automatic') if k in before)
    priority = next(k for k in ('exposure_auto_priority', 'exposure_dynamic_framerate') if k in before)
    base = {auto: 1, priority: 0, 'power_line_frequency': 1, shutter: 100,
            'gain': 0, 'brightness': 128, 'contrast': 128, 'saturation': 128,
            'backlight_compensation': 0, awb: 1}
    cases = [('auto_60hz', {auto: 3, 'power_line_frequency': 2}),
             ('auto_50hz', {auto: 3}),
             ('manual_05ms', {shutter: 5}), ('manual_2ms', {shutter: 20}),
             ('manual_5ms', {shutter: 50}), ('manual_10ms', {}),
             ('manual_20ms', {shutter: 200}),
             ('manual_10ms_backlight1', {'backlight_compensation': 1}),
             ('manual_10ms_wb3340', {awb: 0, 'white_balance_temperature': 3340}),
             ('manual_10ms_wb4000', {awb: 0, 'white_balance_temperature': 4000}),
             ('manual_10ms_brightness96', {'brightness': 96}),
             ('manual_10ms_contrast96', {'contrast': 96})]
    try:
        cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError('Cannot open camera')
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)
        report['stream_test'] = {'width': cap.get(3), 'height': cap.get(4), 'fps': cap.get(5)}
        for name, overrides in cases:
            # Set all cases independently to avoid carrying previous tone/WB values.
            set_controls(base)
            set_controls(overrides)
            for _ in range(60):
                if not cap.grab():
                    raise RuntimeError('Camera stopped during settling')
            lumas, profiles, channels, clipped = [], [], [], []
            started = time.monotonic()
            for i in range(45):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError('Camera stopped during capture')
                small = cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
                lumas.append(float(gray.mean()))
                profiles.append(np.mean(gray, axis=1) / max(float(gray.mean()), 1))
                clipped.append(float(np.mean(np.max(frame, axis=2) >= 250) * 100))
                # Fixed sensor-coordinate patch: inspect JPEG before interpreting as neutral.
                channels.append(np.mean(frame[180:500, 600:1300], axis=(0, 1)).tolist())
                if i in (0, 22, 44):
                    cv2.imwrite(str(args.output / f'{name}_{i:02}.jpg'), frame)
            case = {'name': name, 'readback': controls(),
                    'fps': 45 / (time.monotonic() - started),
                    'mean_luma': float(np.mean(lumas)),
                    'luma_cv_percent': float(np.std(lumas) / max(np.mean(lumas), 1) * 100),
                    'moving_row_rms': float(np.sqrt(np.mean(np.diff(profiles, axis=0) ** 2))),
                    'any_channel_ge250_percent': float(np.mean(clipped)),
                    'patch_bgr': np.mean(channels, axis=0).tolist()}
            report['cases'].append(case)
            (args.output / 'report.json').write_text(json.dumps(report, indent=2))
            print(json.dumps(case), flush=True)
    finally:
        if cap is not None:
            cap.release()
        # Temporarily unlock manual-only controls, restore their values, then
        # restore original auto modes. Do not silently ignore restoration errors.
        restore = [(auto, 1), (awb, 0)]
        restore += [(k, v) for k, v in before.items() if k not in (auto, awb)]
        restore += [(awb, before[awb]), (auto, before[auto])]
        for key, value in restore:
            try:
                set_controls({key: value})
            except Exception as exc:
                report['restoration_errors'].append(f'{key}: {exc}')
        try:
            ctl(f'--set-fmt-video=width={width},height={height},pixelformat={codec}')
            ctl(f'--set-parm={fps}')
        except Exception as exc:
            report['restoration_errors'].append(str(exc))
        report['after'] = controls()
        (args.output / 'report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
