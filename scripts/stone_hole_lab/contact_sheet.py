"""Visual review sheet of archived white detections (predictions, not truth)."""
from pathlib import Path
import json
import sys
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/audits/stone-hole-lab'


def main():
    tiles = []
    for p in sorted((ROOT / 'data/sessions').glob('*/results/stones/item_*/result.json')):
        report = json.loads(p.read_text())
        whites = [r for j in report.get('jewels', []) for r in j.get('regions', [])
                  if r.get('color') == 'White/Colorless']
        if len(whites) < 3:
            continue
        raw = cv2.imread(str(p.parents[3] / 'items' / (p.parent.name + '_raw.png')))
        if raw is None:
            continue
        for j in report['jewels']:
            gx, gy, _, _ = j['bbox_global']
            for r in j['regions']:
                if r['color'] == 'White/Colorless':
                    x, y, w, h = r['bbox']
                    cv2.rectangle(raw, (gx+x, gy+y), (gx+x+w, gy+y+h), (0, 0, 255), 1)
        scale = min(310/raw.shape[1], 300/raw.shape[0])
        resized = cv2.resize(raw, None, fx=scale, fy=scale)
        tile = np.full((340, 320, 3), 255, 'uint8')
        tile[30:30+resized.shape[0], :resized.shape[1]] = resized
        cv2.putText(tile, p.parents[3].name[:15]+' '+p.parent.name, (3, 16), cv2.FONT_HERSHEY_SIMPLEX, .38, (0, 0, 0), 1)
        tiles.append(tile)
    for page in range(0, len(tiles), 16):
        chunk = tiles[page:page+16]
        chunk += [np.full_like(tiles[0], 255)] * (16-len(chunk))
        sheet = np.vstack([np.hstack(chunk[i:i+4]) for i in range(0, 16, 4)])
        cv2.imwrite(str(OUT / f'white_review_{page//16:02d}.jpg'), sheet)
    print(len(tiles), 'items on review sheets')


if __name__ == '__main__':
    main()
