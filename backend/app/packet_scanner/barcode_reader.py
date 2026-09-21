import cv2
import numpy as np
import zxingcpp

from .config import ALLOWED_DIGIT_LENGTHS, BARCODE_REGION_RATIO


def region(image, ratio):
    h = image.shape[0]
    return image[int(h * ratio[0]):max(int(h * ratio[0]) + 1, int(h * ratio[1]))]


def read_barcode(image):
    upper = region(image, BARCODE_REGION_RATIO)
    variants = [("original", image), ("upper", upper),
                ("2x", cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)),
                ("upper_2x", cv2.resize(upper, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))]
    gray = cv2.cvtColor(upper, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    variants.extend([
        ("gray", gray), ("clahe", clahe),
        ("otsu", cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
        ("adaptive", cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)),
        ("sharp", cv2.filter2D(gray, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))),
    ])
    for name, candidate in variants:
        for result in zxingcpp.read_barcodes(candidate, formats=zxingcpp.BarcodeFormat.Code128,
                                               try_rotate=True, try_downscale=True, try_invert=True):
            value = result.text.strip()
            if value.isascii() and value.isdigit() and len(value) in ALLOWED_DIGIT_LENGTHS:
                return value, "Code128", candidate, name
    # Small camera labels often have only 1-2 pixels per narrow bar. A nearest-neighbor
    # upscale preserves those bar edges; fixed threshold handles low contrast on white labels.
    for name, crop in (("full", image), ("upper", upper)):
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
        for factor in (2, 3, 4):
            for kind, base in (("gray", gray), ("clahe", clahe)):
                candidate = cv2.resize(base, None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)
                for binarizer in (zxingcpp.Binarizer.LocalAverage, zxingcpp.Binarizer.FixedThreshold):
                    for result in zxingcpp.read_barcodes(candidate, formats=zxingcpp.BarcodeFormat.Code128,
                                                         try_rotate=True, try_downscale=False, try_invert=True,
                                                         binarizer=binarizer):
                        value = result.text.strip()
                        if value.isascii() and value.isdigit() and len(value) in ALLOWED_DIGIT_LENGTHS:
                            return value, "Code128", candidate, f"{name}_{kind}_{factor}x_{binarizer}"
    return "", "", upper, "upper"
