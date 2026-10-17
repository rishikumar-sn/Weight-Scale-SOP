import unittest
from unittest.mock import Mock

import numpy as np
import zxingcpp

from backend.app.packet_scanner.barcode_reader import read_barcode
from backend.app.packet_scanner.identifier import packet_identifier
from backend.app.packet_scanner.ocr_reader import OCRReader
from backend.app.packet_scanner.validator import validate


class PacketIdentifierTests(unittest.TestCase):
    def test_legacy_and_new_identifiers_keep_all_characters(self):
        for value in ("41233285", "0123456", "EKM9005L-00001659", "ABC1234/00000987",
                      "XYZ123400001659", "BR12_00001659", "BR12.00001659"):
            with self.subTest(value=value):
                self.assertEqual(packet_identifier(value), value)
                self.assertEqual(validate(value, value), "MATCH")
        self.assertEqual(packet_identifier(" EKM9005L – 00001659 "), "EKM9005L-00001659")

    def test_invalid_text_does_not_turn_into_a_digit_suffix(self):
        for value in ("Muthoot Finance", "Packet No: 41233285", "123456", "123456789",
                      "１２３４５６７８", "BR12--00001659", "A" * 33 + "1"):
            with self.subTest(value=value):
                self.assertEqual(packet_identifier(value), "")

    def test_prefix_separator_and_ambiguous_characters_are_compared(self):
        for other in ("ABC9005L-00001659", "EKM9005L-00001658", "EKM9OO5L-00001659",
                      "EKM9005L00001659", "EKM9005L/00001659", "ekm9005l-00001659"):
            with self.subTest(other=other):
                self.assertEqual(validate("EKM9005L-00001659", other), "MISMATCH")
        self.assertEqual(validate("EKM9005L-00001659", "00001659"), "MISMATCH")
        self.assertEqual(validate("EKM9005L-00001659", ""), "BARCODE_ONLY")
        self.assertEqual(validate("", "EKM9005L-00001659"), "OCR_ONLY")

    def test_real_code128_decoder_accepts_both_packet_formats(self):
        for value in ("41233285", "EKM9005L-00001659", "ABC1234/00000987"):
            with self.subTest(value=value):
                gray = np.asarray(zxingcpp.write_barcode(zxingcpp.BarcodeFormat.Code128, value, height=80))
                image = np.repeat(gray[:, :, None], 3, axis=2)
                self.assertEqual(read_barcode(image)[:2], (value, "Code128"))

    def test_ocr_keeps_complete_identifier_in_quick_and_full_paths(self):
        reader = OCRReader.__new__(OCRReader)
        reader._items = Mock(side_effect=lambda _: iter([("EKM9005L - 00001659", .99)]))
        image = np.full((140, 600, 3), 255, dtype=np.uint8)
        self.assertEqual(reader.quick_match(image, "EKM9005L-00001659")[:2],
                         ("EKM9005L-00001659", .99))
        self.assertEqual(reader.read(image)[:2], ("EKM9005L-00001659", .99))
        self.assertIsNone(reader.quick_match(image, "00001659"))


if __name__ == "__main__":
    unittest.main()
