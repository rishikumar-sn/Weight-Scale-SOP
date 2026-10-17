from .identifier import packet_identifier


def validate(barcode, ocr):
    barcode = packet_identifier(barcode) if barcode else ""
    ocr = packet_identifier(ocr) if ocr else ""
    if barcode and ocr:
        return "MATCH" if barcode == ocr else "MISMATCH"
    if barcode:
        return "BARCODE_ONLY"
    if ocr:
        return "OCR_ONLY"
    return "NOT_FOUND"
