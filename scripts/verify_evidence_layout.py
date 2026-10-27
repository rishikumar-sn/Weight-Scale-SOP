"""Check typography bounds, dense trays, and unchanged evidence-photo pixels."""
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from PIL import Image, ImageDraw
from backend.app.analysis import annotations as a


def main():
    output = ROOT.parent / 'testing' / 'evidence-layout'
    output.mkdir(parents=True, exist_ok=True)
    original_text = ImageDraw.ImageDraw.text
    calls = []

    def checked_text(draw, xy, text, *args, **kwargs):
        box = draw.textbbox(xy, text, font=kwargs.get('font'), anchor=kwargs.get('anchor'))
        assert box[0] >= 0 and box[1] >= 0 and box[2] <= 1536 and box[3] <= 1024, (text, box)
        calls.append((str(text), box))
        return original_text(draw, xy, text, *args, **kwargs)

    # Nonuniform pixels expose any change in contain-fit, centering, or resampling.
    image = np.random.default_rng(42).integers(0, 256, (650, 900, 3), dtype=np.uint8)
    source_copy = image.copy()
    with patch.object(ImageDraw.ImageDraw, 'text', checked_text):
        for count in (0, 1, 6, 15, 20):
            items = [(i, 'Earrings / Nosepin' if i % 2 else 'Chain / Necklace') for i in range(1, count + 1)]
            fields = [('JEWELLERY TYPES / %s ITEMS' % count, 'Unused') if count > 1 else ('JEWELLERY TYPE', 'Chain / Necklace'),
                      ('TOTAL CAPTURED WEIGHT', '50.54 g'), ('DATE & TIME', '23-10-2026 12:30:18 PM'), ('IMAGE CONTENT', 'Jewellery present')]
            calls.clear()
            result = a.add_side_panel(image, 'CAPTURE CONFIRMATION', fields, presence_ok=True, jewellery_items=items if count > 1 else None)
            assert result.shape == (1024, 1536, 3)
            if count > 1:
                for index, _ in items:
                    assert any(text == f'J{index}' for text, _ in calls)
            # Table lines must not overlap the weight/content sections below them.
            boxes = [box for text, box in calls if text.startswith('J') and text[1:].isdigit()]
            weight_top = next(box[1] for text, box in calls if text == 'TOTAL CAPTURED WEIGHT')
            assert all(box[3] < weight_top for box in boxes)
            cv2.imwrite(str(output / f'confirmation-{count}.png'), result)
        for title, fields in [
            ('BEAD ANALYSIS', [('DETECTED','16'), ('TINY','2'), ('SMALL','1'), ('LARGE','13')]),
            ('STONE ANALYSIS', [('RESULT','Medium stones'), ('STONE SURFACE AREA','71.37 mm²')]),
            ('STONE ANALYSIS', [('RESULT','High risk'), ('STONE SURFACE AREA','71.37 mm²'), ('SURFACE REFLECTION','Surface reflection detected; possible stones are present. Please review the marked regions.')]),
            ('DIMENSION ANALYSIS', [('OUTER DIAMETER (OD)','71.87 mm'), ('INNER DIAMETER (ID)','65.71 mm')]),
        ]:
            result = a.add_side_panel(image, title, fields)
            result = a.identify_result_image(result, 'J4 / Bangle')
            # Reproduce the original photo transform, then compare all photo pixels.
            factor = min(2., 840 * .80 / 900, 856 * .80 / 650)
            photo = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)).resize((round(900*factor),round(650*factor)),Image.Resampling.LANCZOS)
            expected = cv2.cvtColor(np.asarray(photo),cv2.COLOR_RGB2BGR)
            x, y = 12+(840-photo.width)//2, 156+(856-photo.height)//2
            assert np.array_equal(result[y:y+photo.height,x:x+photo.width], expected)
            cv2.imwrite(str(output / f'{title.lower().replace(" ", "-")}-{len(fields)}.png'),result)
        # Long types are wrapped and retained, including the last jewel.
        dense = [(i,'Earrings / Nosepin / Custom jewellery description') for i in range(1,21)]
        a.add_side_panel(image,'CAPTURE REVIEW',[('JEWELLERY TYPES / 20 ITEMS',''),('CAPTURED WEIGHT','50.54 g'),('IMAGE CONTENT','Jewellery present')],jewellery_items=dense)
    assert np.array_equal(image,source_copy)
    print('Passed: fixed resolution, 0/1/6/15/20 jewels, wrapped labels, text bounds, photo pixels, unchanged source.')


if __name__ == '__main__':
    main()
