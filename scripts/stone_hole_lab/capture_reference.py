"""Capture an empty tray without starting the app or changing its settings."""
import json
import argparse
from pathlib import Path
import sys
import time

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from backend.app.hardware.camera import CameraService


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', choices=['empty_tray', 'white_stone_control'], default='empty_tray')
    args = parser.parse_args()
    out = ROOT / 'data/audits/stone-hole-lab' / ('reference' if args.scene == 'empty_tray' else 'control')
    out.mkdir(parents=True, exist_ok=True)
    settings = json.loads((ROOT / 'data/settings/app.json').read_text())
    profile = settings.get('hardware_profiles', {}).get('windows', {})
    settings.update({k: v for k, v in profile.items() if k in ('camera', 'scale')})
    camera = CameraService(lambda: settings)
    camera.start()
    try:
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            try:
                frame, metadata = camera.snapshot_frame()
                if not metadata.get('anti_flicker', {}).get('ready', False):
                    time.sleep(0.2)
                    continue
                break
            except RuntimeError:
                time.sleep(0.2)
        else:
            raise RuntimeError(f'Camera did not stabilize: {camera.status()}')
        if not cv2.imwrite(str(out / (args.scene + '.png')), frame):
            raise RuntimeError('Reference write failed')
        metadata['settings'] = settings
        metadata['scene'] = args.scene
        metadata['scene_confirmed_by_user'] = True
        (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
        print(json.dumps({'path': str(out / (args.scene + '.png')),
                          'exposure': metadata['exposure'],
                          'anti_flicker': metadata['anti_flicker']}, indent=2))
    finally:
        camera.stop()


if __name__ == '__main__':
    main()
