from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image as PillowImage, ImageDraw, ImageFont, ImageOps

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from ..core.config import PROJECT_ROOT
from ..domain.weights import weight_summary, weight_rows


class ReportService:
    def __init__(self) -> None:
        self.company_logo = PROJECT_ROOT / "assets" / "branding" / "embsys_logo.png"
        self.client_logo = PROJECT_ROOT / "assets" / "branding" / "logo.jpg"

    @staticmethod
    def _result_font(size: int, bold: bool = False):
        candidates = (
            Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        )
        for candidate in candidates:
            if candidate.is_file():
                return ImageFont.truetype(str(candidate), size=size)
        return ImageFont.load_default()

    @staticmethod
    def _fit_text(draw: ImageDraw.ImageDraw, text: str, max_width: int, preferred_size: int, bold: bool = False):
        size = preferred_size
        while size > 18:
            font = ReportService._result_font(size, bold)
            if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
                return font
            size -= 2
        return ReportService._result_font(size, bold)

    def generate_tare_result_image(self, state: dict[str, Any]) -> PillowImage.Image:
        """Place packet verification beside the untouched tare photograph."""
        original_path = Path(state["paths"]["original"])
        with PillowImage.open(original_path) as source:
            capture = ImageOps.exif_transpose(source).convert("RGB")
        image_width, image_height = capture.size
        panel_width = max(480, min(780, round(image_width * 0.54)))
        output = PillowImage.new("RGB", (image_width + panel_width, image_height), "#F3F6F9")
        output.paste(capture, (0, 0))
        draw = ImageDraw.Draw(output)
        scale = min(panel_width / 580, image_height / 1920)

        def px(value: float) -> int:
            return max(1, round(value * scale))

        packet = (state.get("result") or {}).get("packet") or {}
        for detection in packet.get("detections") or []:
            points = detection.get("points") or []
            if len(points) == 4:
                polygon = [tuple(point) for point in points]
                draw.line(polygon + polygon[:1], fill="#51C8A6", width=px(5))

        left = image_width
        x = left + px(42)
        right = image_width + panel_width - px(42)
        width = right - x
        navy, gold, muted, ink = "#14233B", "#C79A43", "#617087", "#182235"
        draw.rectangle((left, 0, image_width + panel_width, px(240)), fill=navy)
        draw.rectangle((left, px(232), image_width + panel_width, px(240)), fill=gold)
        mode = str(state.get("tare_mode") or "").upper()
        draw.text((x, px(48)), f"{mode} TARE CAPTURE", fill="#C9D4E3", font=self._result_font(px(26), True))
        draw.text((x, px(102)), "Packet Summary", fill="white", font=self._fit_text(draw, "Packet Summary", width, px(50), True))

        def card(top: int, height: int, label: str, value: str, color: str = ink) -> int:
            draw.rounded_rectangle((x, top, right, top + height), radius=px(18), fill="white", outline="#DCE3EA", width=px(2))
            draw.rectangle((x, top + px(12), x + px(8), top + height - px(12)), fill=gold)
            draw.text((x + px(27), top + px(24)), label, fill=muted, font=self._result_font(px(23), True))
            draw.text((x + px(27), top + px(72)), value, fill=color,
                      font=self._fit_text(draw, value, width - px(55), px(48), True))
            return top + height + px(22)

        top = px(290)
        number = str(packet.get("packet_number") or "Unavailable")
        top = card(top, px(164), "PACKET NUMBER (OCR)", number)
        status = str(packet.get("status") or "NOT_FOUND")
        status_text = {
            "MATCH": "BARCODE MATCHED",
            "MISMATCH": "BARCODE MISMATCH",
            "BARCODE_ONLY": "BARCODE ONLY",
            "OCR_ONLY": "BARCODE NOT READ",
            "NOT_FOUND": "PACKET NOT READ",
        }.get(status, "NOT VERIFIED")
        status_color = "#167D5A" if status == "MATCH" else "#B24C37"
        top = card(top, px(164), "BARCODE CHECK", status_text, status_color)
        barcode = str(packet.get("barcode") or "Unavailable")
        top = card(top, px(164), "BARCODE DIGITS", barcode)
        weight = state.get("weight_g")
        weight_text = f"{float(weight):.2f} g" if weight is not None else "Unavailable"
        top = card(top, px(164), "TARE WEIGHT", weight_text)
        try:
            captured = datetime.fromisoformat(str(state.get("captured_at")))
            date_text, time_text = captured.strftime("%d-%m-%Y"), captured.strftime("%I:%M:%S %p")
        except ValueError:
            date_text, time_text = "Unavailable", "Unavailable"
        top = card(top, px(164), "CAPTURE DATE", date_text)
        card(top, px(164), "CAPTURE TIME", time_text)
        return output

    def generate_result_image(self, state: dict[str, Any]) -> io.BytesIO:
        """Compose the untouched capture and its summary into one PNG image."""
        original_path = Path(str((state.get("paths") or {}).get("original") or ""))
        if not original_path.is_file():
            raise FileNotFoundError("The original captured image is unavailable")

        with PillowImage.open(original_path) as source:
            capture = ImageOps.exif_transpose(source).convert("RGB")

        image_width, image_height = capture.size
        panel_width = max(480, min(780, round(image_width * 0.54)))
        navy = "#14233B"
        navy_soft = "#203653"
        gold = "#C79A43"
        ink = "#182235"
        muted = "#617087"
        paper = "#F3F6F9"
        card = "#FFFFFF"
        line_color = "#DCE3EA"
        output = PillowImage.new("RGB", (image_width + panel_width, image_height), paper)
        output.paste(capture, (0, 0))
        draw = ImageDraw.Draw(output)
        scale = min(panel_width / 580, image_height / 1920)

        def px(value: float) -> int:
            return max(1, round(value * scale))

        panel_left = image_width
        panel_right = image_width + panel_width
        header_height = px(250)
        draw.rectangle((panel_left, 0, panel_right, header_height), fill=navy)
        draw.rectangle((panel_left, header_height - px(8), panel_right, header_height), fill=gold)

        result = state.get("result") or {}
        classification = state.get("classification") or {}
        items = result.get("items") or []
        count = int(result.get("count") or classification.get("count") or len(items) or 1)
        labels = [str(item.get("label") or "-") for item in items]
        if not labels:
            labels = list(classification.get("confirmed_labels") or classification.get("labels") or [])
        if not labels:
            labels = [str(classification.get("confirmed_label") or classification.get("predicted_label") or "-")]

        # Recreate the confirmed per-jewel boxes on the untouched capture. The
        # detector coordinates are relative to the processing ROI, so translate
        # each one back into the full camera frame before drawing it.
        processing_roi = (state.get("rois") or {}).get("processing") or {}
        offset_x = int(processing_roi.get("x", 0))
        offset_y = int(processing_roi.get("y", 0))
        box_scale = max(0.7, min(image_width, image_height) / 900)
        box_width = max(2, round(2 * box_scale))
        box_color = "#62BFAE"
        tag_color = "#14233B"
        tag_accent = "#C79A43"
        tag_padding_x = max(14, round(18 * box_scale))
        tag_padding_y = max(8, round(10 * box_scale))
        for position, item in enumerate(classification.get("items") or [], start=1):
            bbox = item.get("bbox") or {}
            x1 = max(0, int(bbox.get("x", 0)) + offset_x)
            y1 = max(0, int(bbox.get("y", 0)) + offset_y)
            x2 = min(image_width - 1, x1 + int(bbox.get("w", 0)))
            y2 = min(image_height - 1, y1 + int(bbox.get("h", 0)))
            if x2 <= x1 or y2 <= y1:
                continue
            draw.rectangle((x1, y1, x2, y2), outline=box_color, width=box_width)
            item_index = int(item.get("index") or position)
            tag_text = f"J{item_index}"
            tag_font = self._fit_text(
                draw,
                tag_text,
                max(60, min(x2 - x1, image_width - x1) - (tag_padding_x * 2)),
                max(22, round(27 * box_scale)),
                bold=True,
            )
            text_box = draw.textbbox((0, 0), tag_text, font=tag_font)
            text_width = text_box[2] - text_box[0]
            text_height = text_box[3] - text_box[1]
            tag_height = text_height + (tag_padding_y * 2)
            tag_width = min(image_width - x1, text_width + (tag_padding_x * 2) + box_width)
            tag_top = max(0, y1 - tag_height)
            if tag_top == y1:
                tag_top = y1
            draw.rectangle((x1, tag_top, x1 + tag_width, tag_top + tag_height), fill=tag_color)
            draw.rectangle((x1, tag_top, x1 + box_width + 3, tag_top + tag_height), fill=tag_accent)
            draw.text(
                (x1 + tag_padding_x + box_width, tag_top + tag_padding_y - text_box[1]),
                tag_text,
                fill="white",
                font=tag_font,
            )

        captured_at = str(state.get("captured_at") or "")
        try:
            captured = datetime.fromisoformat(captured_at)
            date_text = captured.strftime("%d-%m-%Y")
            time_text = captured.strftime("%I:%M:%S %p")
        except ValueError:
            date_text, time_text = captured_at or "-", "-"
        weight_text = f"{float(state['weight_g']):.2f} g" if state.get("weight_g") is not None else "Unavailable"

        padding = max(px(44), round(panel_width * 0.07))
        x = panel_left + padding
        available_width = panel_width - (padding * 2)
        eyebrow_font = self._result_font(px(25), bold=True)
        title_font = self._result_font(px(48), bold=True)
        section_font = self._result_font(px(24), bold=True)
        label_font = self._result_font(px(22), bold=True)

        draw.text((x, px(54)), "JEWELLERY CAPTURE", fill="#C9D4E3", font=eyebrow_font)
        draw.text((x, px(100)), "Capture Summary", fill="white", font=title_font)

        content_top = header_height + px(46)
        gap = px(22)
        stat_height = px(235)
        count_width = round(available_width * 0.31)
        weight_left = x + count_width + gap
        weight_width = available_width - count_width - gap

        draw.rounded_rectangle(
            (x, content_top, x + count_width, content_top + stat_height),
            radius=px(20), fill=navy_soft,
        )
        count_label_font = self._fit_text(draw, "JEWEL COUNT", count_width - px(34), px(22), bold=True)
        draw.text((x + px(17), content_top + px(29)), "JEWEL COUNT", fill="#C9D4E3", font=count_label_font)
        count_font = self._fit_text(draw, str(count), count_width - px(48), px(86), bold=True)
        draw.text((x + px(24), content_top + px(89)), str(count), fill="white", font=count_font)

        draw.rounded_rectangle(
            (weight_left, content_top, weight_left + weight_width, content_top + stat_height),
            radius=px(20), fill=card, outline=line_color, width=px(2),
        )
        draw.rectangle(
            (weight_left, content_top, weight_left + px(9), content_top + stat_height),
            fill=gold,
        )
        draw.text((weight_left + px(30), content_top + px(29)), "CAPTURED WEIGHT", fill=muted, font=label_font)
        fitted_weight = self._fit_text(draw, weight_text, weight_width - px(60), px(66), bold=True)
        draw.text((weight_left + px(30), content_top + px(91)), weight_text, fill=ink, font=fitted_weight)

        meta_top = content_top + stat_height + gap
        meta_height = px(205)
        draw.rounded_rectangle(
            (x, meta_top, x + available_width, meta_top + meta_height),
            radius=px(20), fill=card, outline=line_color, width=px(2),
        )
        meta_mid = x + available_width // 2
        draw.line((meta_mid, meta_top + px(26), meta_mid, meta_top + meta_height - px(26)), fill=line_color, width=px(2))
        draw.text((x + px(28), meta_top + px(28)), "CAPTURE DATE", fill=muted, font=label_font)
        date_font = self._fit_text(draw, date_text, available_width // 2 - px(56), px(32), bold=True)
        draw.text((x + px(28), meta_top + px(91)), date_text, fill=ink, font=date_font)
        draw.text((meta_mid + px(28), meta_top + px(28)), "CAPTURE TIME", fill=muted, font=label_font)
        time_font = self._fit_text(draw, time_text, available_width // 2 - px(56), px(32), bold=True)
        draw.text((meta_mid + px(28), meta_top + px(91)), time_text, fill=ink, font=time_font)

        weights_top = meta_top + meta_height + px(24)
        stone_analysis_applies = any(
            (item.get("route") or {}).get("stones") or item.get("stones")
            for item in (items or [result])
        )
        rows = weight_rows(weight_summary(state)) if count == 1 and stone_analysis_applies else []
        row_height = px(62)
        column_offsets = [0, int(available_width * 0.40), int(available_width * 0.66), available_width]
        for row_index, row in enumerate(rows):
            top = weights_top + row_index * row_height
            for column, value in enumerate(row):
                left, right = x + column_offsets[column], x + column_offsets[column + 1]
                draw.rectangle((left, top, right, top + row_height),
                               fill=navy if row_index == 0 else card, outline=line_color)
                font = self._fit_text(draw, value, right - left - px(16), px(22), bold=row_index == 0)
                draw.text((left + px(8), top + px(16)), value,
                          fill="white" if row_index == 0 else ink, font=font)
        types_top = weights_top + len(rows) * row_height + px(30)
        draw.text((x, types_top), "JEWELLERY DETAILS", fill=navy, font=section_font)
        draw.rectangle((x, types_top + px(45), x + px(72), types_top + px(51)), fill=gold)
        item_top = types_top + px(82)
        item_height = px(104)
        item_gap = px(14)
        for index, label in enumerate(labels, start=1):
            top = item_top + (index - 1) * (item_height + item_gap)
            logo_zone_top = image_height - px(305)
            if top + item_height > logo_zone_top - px(20):
                break
            draw.rounded_rectangle(
                (x, top, x + available_width, top + item_height),
                radius=px(16), fill=card, outline=line_color, width=px(2),
            )
            badge_size = px(58)
            badge_x = x + px(22)
            badge_y = top + (item_height - badge_size) // 2
            draw.rounded_rectangle(
                (badge_x, badge_y, badge_x + badge_size, badge_y + badge_size),
                radius=px(14), fill=navy,
            )
            jewel_reference = f"J{index}"
            number_font = self._fit_text(draw, jewel_reference, badge_size - px(10), px(25), bold=True)
            number_box = draw.textbbox((0, 0), jewel_reference, font=number_font)
            draw.text(
                (badge_x + (badge_size - (number_box[2] - number_box[0])) // 2, badge_y + px(10)),
                jewel_reference, fill="white", font=number_font,
            )
            type_font = self._fit_text(draw, label, available_width - px(125), px(34), bold=True)
            beads_item = (items[index - 1].get("beads") or {}) if index <= len(items) else {}
            text_y = top + (item_height - type_font.size) // 2 - px(4) - (px(16) if beads_item else 0)
            draw.text((badge_x + badge_size + px(22), text_y), label, fill=ink, font=type_font)
            if beads_item:
                bead_count = beads_item.get("count", len(beads_item.get("detections") or []))
                draw.text((badge_x + badge_size + px(22), top + px(65)), f"Bead count: {bead_count}",
                          fill=muted, font=self._result_font(px(22)))

        if self.client_logo.is_file():
            with PillowImage.open(self.client_logo) as logo_source:
                logo = ImageOps.exif_transpose(logo_source).convert("RGB")
                logo.thumbnail((available_width, px(150)), PillowImage.Resampling.LANCZOS)
                logo_x = image_width + (panel_width - logo.width) // 2
                logo_y = image_height - px(67) - logo.height
                divider_y = image_height - px(250)
                draw.line((x, divider_y, x + available_width, divider_y), fill=line_color, width=px(2))
                output.paste(logo, (logo_x, logo_y))

        buffer = io.BytesIO()
        output.save(buffer, format="PNG", optimize=True, compress_level=9)
        buffer.seek(0)
        return buffer

    @staticmethod
    def _image(path: str | Path | None, width: float = 165 * mm, height: float = 105 * mm):
        if not path or not Path(path).is_file():
            return None
        image = Image(str(path))
        image._restrictSize(width, height)
        image.hAlign = "CENTER"
        return image

    def generate(self, state: dict[str, Any]) -> io.BytesIO:
        buffer = io.BytesIO()
        document = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=16 * mm,
            leftMargin=16 * mm,
            topMargin=14 * mm,
            bottomMargin=14 * mm,
            title="Jewellery Capture Report",
            pageCompression=1,
        )
        styles = getSampleStyleSheet()
        title = ParagraphStyle(
            "TitleCenter",
            parent=styles["Title"],
            alignment=TA_CENTER,
            textColor=colors.HexColor("#172033"),
            fontSize=18,
            spaceAfter=8,
        )
        heading = ParagraphStyle(
            "Section",
            parent=styles["Heading2"],
            textColor=colors.HexColor("#9A6816"),
            fontSize=12,
            spaceBefore=7,
            spaceAfter=5,
        )
        body = styles["BodyText"]
        body.leading = 16
        story: list[Any] = []
        logo_cells = []
        for path in (self.company_logo, self.client_logo):
            logo = self._image(path, 42 * mm, 16 * mm)
            logo_cells.append(logo or "")
        logo_table = Table([logo_cells], colWidths=[84 * mm, 84 * mm])
        logo_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (0, 0), "LEFT"),
            ("ALIGN", (1, 0), (1, 0), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.extend([logo_table, Spacer(1, 4 * mm), Paragraph("Jewellery Capture Report", title)])

        classification = state.get("classification") or {}
        label = classification.get("confirmed_label") or classification.get("predicted_label") or "-"
        captured_on = str(state.get("captured_at") or "-")
        try:
            captured_on = datetime.fromisoformat(captured_on).strftime("%d-%m-%Y %I:%M:%S %p")
        except ValueError:
            pass
        details = [
            ["Captured on", captured_on],
            ["Jewel count", str(int(classification.get("count") or 1))],
            ["Jewel types", str(label)],
            ["Weight", f"{float(state['weight_g']):.2f} g" if state.get("weight_g") is not None else "Unavailable"],
        ]
        table = Table(details, colWidths=[45 * mm, 120 * mm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F6EBD5")),
            ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#6F4B12")),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D8C7A7")),
            ("PADDING", (0, 0), (-1, -1), 7),
        ]))
        story.extend([Paragraph("Capture details", heading), table])

        result = state.get("result") or {}
        if any(
            (item.get("route") or {}).get("stones") or item.get("stones")
            for item in (result.get("items") or [result])
        ):
            weights = weight_summary(state)
            weights_table = Table(weight_rows(weights), colWidths=[60 * mm, 45 * mm, 60 * mm])
            weights_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F6EBD5")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D8C7A7")),
                ("PADDING", (0, 0), (-1, -1), 7),
            ]))
            story.extend([Paragraph("Jewellery weights (grams)", heading), weights_table,
                          Paragraph(weights["note"], body)])

        result_capture = Image(self.generate_result_image(state))
        result_capture._restrictSize(177 * mm, 140 * mm)
        result_capture.hAlign = "CENTER"
        if result_capture:
            story.extend([
                Spacer(1, 5 * mm),
                Paragraph("Captured image and details", heading),
                result_capture,
            ])

        result = state.get("result") or {}
        result_items = result.get("items") or []
        if result_items:
            summary_rows = [["Jewel", "Type", "Analysis status"]]
            for item in result_items:
                status = str(item.get("status") or "complete").title()
                if item.get("errors"):
                    status += f" ({len(item['errors'])} warning(s))"
                summary_rows.append([
                    str(item.get("index") or "-"),
                    str(item.get("label") or "-"),
                    status,
                ])
            summary = Table(summary_rows, colWidths=[25 * mm, 78 * mm, 62 * mm])
            summary.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F6EBD5")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D8C7A7")),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]))
            story.extend([Paragraph("Detected jewels", heading), summary])

            for item in result_items:
                story.append(Paragraph(
                    f"Jewel {int(item.get('index') or 0)}: {item.get('label') or '-'}",
                    heading,
                ))
                dimensions_item = item.get("dimensions") or {}
                if dimensions_item:
                    rows = []
                    for label_text, key in (
                        ("Outer diameter", "outer_diameter_mm"),
                        ("Inner diameter", "inner_diameter_mm"),
                        ("Thickness", "thickness_mm"),
                    ):
                        value = dimensions_item.get(key)
                        rows.append([label_text, f"{float(value):.2f} mm" if value is not None else "Unavailable"])
                    story.append(Table(rows, colWidths=[60 * mm, 105 * mm]))
                beads_item = item.get("beads") or {}
                if beads_item:
                    bead_count = beads_item.get("count", len(beads_item.get("detections") or []))
                    story.append(Paragraph(f"Bead count: {bead_count}", body))
                    story.append(
                        Paragraph(
                            "Beads detected"
                            if beads_item.get("beads_detected")
                            else "Beads not detected",
                            body,
                        )
                    )
                stones_item = item.get("stones") or {}
                if stones_item:
                    story.append(Paragraph(str(stones_item.get("risk_status") or "Stone analysis complete"), body))
                    if stones_item.get("estimated_weight_g") is not None:
                        story.append(Paragraph(
                            f"Estimated stone weight: {stones_item['estimated_weight_g']:.2f} g; "
                            f"range: {stones_item['weight_min_g']:.2f} - {stones_item['weight_max_g']:.2f} g",
                            body,
                        ))
                for warning in item.get("errors") or []:
                    story.append(Paragraph(
                        f"Warning ({warning.get('stage', 'analysis')}): {warning.get('message', '')}",
                        body,
                    ))
                for analysis_name, item_path in (
                    ("Dimension analysis", dimensions_item.get("image")),
                    ("Bead analysis", beads_item.get("image")),
                    ("Stone analysis", stones_item.get("image")),
                ):
                    image = self._image(item_path, 177 * mm, 215 * mm)
                    if image:
                        story.extend([
                            PageBreak(),
                            Paragraph(
                                f"Jewel {int(item.get('index') or 0)} — {analysis_name}",
                                heading,
                            ),
                            Spacer(1, 3 * mm),
                            image,
                        ])

        dimensions = (result.get("dimensions") or {}) if not result_items else {}
        if dimensions:
            rows = []
            for label_text, key in (
                ("Outer diameter", "outer_diameter_mm"),
                ("Inner diameter", "inner_diameter_mm"),
                ("Thickness", "thickness_mm"),
            ):
                value = dimensions.get(key)
                rows.append([label_text, f"{float(value):.2f} mm" if value is not None else "Unavailable"])
            story.extend([Paragraph("Size details", heading), Table(rows, colWidths=[60 * mm, 105 * mm])])

        beads = (result.get("beads") or {}) if not result_items else {}
        if beads:
            story.append(Paragraph(f"Bead count: {beads.get('count', len(beads.get('detections') or []))}", body))
            story.extend([
                Paragraph("Bead analysis", heading),
                Paragraph(
                    "Beads detected"
                    if beads.get("beads_detected")
                    else "Beads not detected",
                    body,
                ),
            ])

        stones = (result.get("stones") or {}) if not result_items else {}
        if stones:
            stone_lines = [str(stones.get("risk_status") or "NO RISK - NO STONES DETECTED")]
            if stones.get("found"):
                stone_lines.append("Detected stone regions are highlighted in green.")
            story.extend([Paragraph("Stone details", heading), Paragraph("<br/>".join(stone_lines), body)])

        result_images = [
            ("Dimension analysis", dimensions.get("image")),
            ("Bead analysis", beads.get("image")),
            ("Stone analysis", stones.get("image")),
        ]
        for analysis_name, path in result_images:
            image = self._image(path, 177 * mm, 215 * mm)
            if image:
                story.extend([
                    PageBreak(),
                    Paragraph(analysis_name, heading),
                    Spacer(1, 3 * mm),
                    image,
                ])

        document.build(story)
        buffer.seek(0)
        return buffer
