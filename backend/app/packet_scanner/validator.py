from .config import ALLOWED_DIGIT_LENGTHS


def validate(barcode, ocr):
    valid = lambda value: bool(value and value.isascii() and value.isdigit() and len(value) in ALLOWED_DIGIT_LENGTHS)
    barcode = barcode if valid(barcode) else ""
    ocr = ocr if valid(ocr) else ""
    if barcode and ocr:
        return "MATCH" if barcode == ocr else "MISMATCH"
    if barcode:
        return "BARCODE_ONLY"
    if ocr:
        return "OCR_ONLY"
    return "NOT_FOUND"
