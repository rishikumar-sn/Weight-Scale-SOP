import logging
import re

import cv2
import numpy as np

from .config import ALLOWED_DIGIT_LENGTHS, OCR_REGION_RATIO


class OCRReader:
    def __init__(self):
        from paddleocr import PaddleOCR
        # PaddleOCR 3.x uses predict(); 2.x uses ocr(). Keep one engine per worker.
        try:
            self.engine = PaddleOCR(lang="en", use_textline_orientation=False,
                                    use_doc_orientation_classify=False, use_doc_unwarping=False,
                                    enable_mkldnn=False)
        except TypeError:
            self.engine = PaddleOCR(lang="en", use_angle_cls=False, show_log=False)

    def _items(self, image):
        if hasattr(self.engine, "predict"):
            results = self.engine.predict(image)
            for result in results:
                data = result.json if hasattr(result, "json") else result
                if isinstance(data, dict):
                    data = data.get("res", data)
                    for text, score in zip(data.get("rec_texts", []), data.get("rec_scores", [])):
                        yield str(text), float(score)
        else:
            results = self.engine.ocr(image, cls=False) or []
            for group in results:
                for line in group or []:
                    if len(line) >= 2 and isinstance(line[1], (list, tuple)):
                        yield str(line[1][0]), float(line[1][1])

    def quick_match(self, image, barcode):
        """Accept the original crop only when it confidently confirms the barcode."""
        if not barcode:
            return None
        h = image.shape[0]
        crop = image[int(h * OCR_REGION_RATIO[0]):int(h * OCR_REGION_RATIO[1])]
        if crop.size == 0:
            return None
        try:
            readings = []
            for text, score in self._items(crop):
                digits = "".join(re.findall(r"\d+", text))
                if digits.isascii() and len(digits) in ALLOWED_DIGIT_LENGTHS:
                    readings.append((score, digits))
        except Exception:
            logging.exception("Quick OCR check failed")
            return None
        confidence, digits = max(readings, default=(0.0, ""))
        return (digits, confidence, crop, crop, "original") if digits == barcode and confidence >= 0.90 else None

    def read(self, image):
        h = image.shape[0]
        crop = image[int(h * OCR_REGION_RATIO[0]):int(h * OCR_REGION_RATIO[1])]
        if crop.size == 0:
            return "", 0.0, crop, crop, "none"
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        variants = [("original", crop), ("gray", gray),
                    ("clahe", cv2.createCLAHE(2.0, (8, 8)).apply(gray)),
                    ("2x", cv2.resize(gray, None, fx=2, fy=2)),
                    ("3x", cv2.resize(gray, None, fx=3, fy=3)),
                    ("otsu", cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
                    ("adaptive", cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)),
                    ("sharp", cv2.filter2D(gray, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])))]
        best = ("", 0.0, crop, "original")
        for name, candidate in variants:
            try:
                supplied = cv2.cvtColor(candidate, cv2.COLOR_GRAY2BGR) if candidate.ndim == 2 else candidate
                items = list(self._items(supplied))
            except Exception:
                logging.exception("OCR variant %s failed", name)
                continue
            for text, score in items:
                digits = "".join(re.findall(r"\d+", text))
                if digits.isascii() and len(digits) in ALLOWED_DIGIT_LENGTHS and score > best[1]:
                    best = (digits, score, candidate, name)
            if best[1] >= 0.90:
                break
        return best[0], best[1], crop, best[2], best[3]
