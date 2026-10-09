from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from ..analysis.beads import analyze_bead_detections, onnx_image_size, prepare_onnx_input
from ..analysis.vision import separate_jewellery_items
from ..core.config import PROJECT_ROOT
from ..domain.workflow import route_for_label
from ..domain.weights import (
    stone_weight_fields,
    weight_summary,
    without_stone_weight_estimates,
)


for module_path in (
    PROJECT_ROOT / "StoneDetection",
    PROJECT_ROOT / "Dimension",
):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from Classification.jewelry_classifier import JewelryZeroShotClassifier  # noqa: E402
from Dimension.bangle_detector import detect_bangle  # noqa: E402
import StoneDetection.jewel_gem_hsv_report as stone_detection  # noqa: E402


BEAD_NMS_SMALL_IOU = 0.45
BEAD_NMS_MEDIUM_IOU = 0.60
BEAD_NMS_LARGE_IOU = 0.72
BEAD_DETECTION_SCORE_THRESHOLD = 0.35
FINGER_RING_MAX_OUTER_SPAN_MM = 35.0
BANGLE_MIN_OUTER_SPAN_MM = 40.0


def _apriltag_scale_mm_per_pixel(calibration: dict[str, Any] | None) -> float:
    """Return one area-preserving scale from the AprilTag X/Y calibration."""
    calibration = calibration or {}
    if not calibration.get("available") or not calibration.get("found"):
        message = str(calibration.get("message") or "AprilTag was not detected")
        raise RuntimeError(
            f"Dimension measurement requires AprilTag calibration: {message}. "
            "Keep the complete tag visible and capture again."
        )
    try:
        scale_x = float(calibration["mm_per_pixel_x"])
        scale_y = float(calibration["mm_per_pixel_y"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("AprilTag calibration does not contain valid X/Y scales") from exc
    if not np.isfinite([scale_x, scale_y]).all() or scale_x <= 0.0 or scale_y <= 0.0:
        raise RuntimeError("AprilTag calibration scales must be finite and positive")

    # A circle's measured area spans both image axes.  The geometric mean is
    # the correct single linear scale for preserving that area when the camera
    # has a small X/Y pixel-scale difference.
    return float(math.sqrt(scale_x * scale_y))


def _ring_bangle_dimension_fallback(
    predicted_label: str,
    item_mask: np.ndarray,
    calibration: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    """Resolve only ring/bangle conflicts using calibrated physical size.

    A deliberately unused 35--40 mm guard band prevents borderline or noisy
    measurements from changing the semantic prediction.
    """
    label = str(predicted_label).strip()
    evidence: dict[str, Any] = {
        "available": False,
        "applied": False,
        "input_label": label,
        "output_label": label,
    }
    if label not in {"Finger Ring", "Bangle"}:
        evidence["reason"] = "not a finger-ring/bangle prediction"
        return label, evidence

    calibration = calibration or {}
    if not calibration.get("available") or not calibration.get("found"):
        evidence["reason"] = "AprilTag calibration unavailable"
        return label, evidence

    try:
        scale_x = float(calibration["mm_per_pixel_x"])
        scale_y = float(calibration["mm_per_pixel_y"])
    except (KeyError, TypeError, ValueError):
        evidence["reason"] = "AprilTag calibration scales unavailable"
        return label, evidence
    if not np.isfinite([scale_x, scale_y]).all() or scale_x <= 0.0 or scale_y <= 0.0:
        evidence["reason"] = "AprilTag calibration scales invalid"
        return label, evidence

    foreground = np.asarray(item_mask) > 0
    points = cv2.findNonZero(foreground.astype(np.uint8))
    if points is None:
        evidence["reason"] = "item mask has no foreground"
        return label, evidence

    _, _, width_px, height_px = cv2.boundingRect(points)
    width_mm = float(width_px * scale_x)
    height_mm = float(height_px * scale_y)
    outer_span_mm = max(width_mm, height_mm)
    evidence.update(
        {
            "available": True,
            "width_mm": width_mm,
            "height_mm": height_mm,
            "outer_span_mm": outer_span_mm,
            "finger_ring_max_mm": FINGER_RING_MAX_OUTER_SPAN_MM,
            "bangle_min_mm": BANGLE_MIN_OUTER_SPAN_MM,
        }
    )

    corrected_label = label
    if label == "Bangle" and outer_span_mm <= FINGER_RING_MAX_OUTER_SPAN_MM:
        corrected_label = "Finger Ring"
        evidence["reason"] = "measured span is physically finger-ring sized"
    elif label == "Finger Ring" and outer_span_mm >= BANGLE_MIN_OUTER_SPAN_MM:
        corrected_label = "Bangle"
        evidence["reason"] = "measured span is physically bangle sized"
    else:
        evidence["reason"] = "measurement does not justify a ring/bangle override"

    evidence["applied"] = corrected_label != label
    evidence["output_label"] = corrected_label
    return corrected_label, evidence


def _bead_nms_iou_threshold(
    bbox: list[int],
    image_shape: tuple[int, ...],
) -> float:
    """Allow more overlap between detections when beads are large in-frame."""
    image_height, image_width = image_shape[:2]
    x1, y1, x2, y2 = bbox
    relative_size = max(
        max(0, x2 - x1) / max(1, image_width),
        max(0, y2 - y1) / max(1, image_height),
    )
    if relative_size >= 0.06:
        return BEAD_NMS_LARGE_IOU
    if relative_size >= 0.035:
        return BEAD_NMS_MEDIUM_IOU
    return BEAD_NMS_SMALL_IOU


def _box_iou(first: list[int], second: list[int]) -> float:
    intersection_width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return float(intersection / union) if union > 0 else 0.0


def _adaptive_bead_nms(
    boxes: list[list[int]],
    scores: list[float],
    image_shape: tuple[int, ...],
) -> list[int]:
    """Greedy NMS with an IoU threshold derived from each bead's image-relative size."""
    selected: list[int] = []
    for candidate_index in sorted(range(len(scores)), key=scores.__getitem__, reverse=True):
        candidate_box = boxes[candidate_index]
        candidate_iou = _bead_nms_iou_threshold(candidate_box, image_shape)
        duplicate = False
        for selected_index in selected:
            selected_iou = _bead_nms_iou_threshold(boxes[selected_index], image_shape)
            if _box_iou(candidate_box, boxes[selected_index]) > max(candidate_iou, selected_iou):
                duplicate = True
                break
        if not duplicate:
            selected.append(candidate_index)
    return selected


def _box_center_in_region(bbox: list[int], region: dict[str, int] | None) -> bool:
    """Keep a full-ROI detection only when its centre belongs to the requested jewel."""
    if not region:
        return True
    center_x = (bbox[0] + bbox[2]) / 2.0
    center_y = (bbox[1] + bbox[3]) / 2.0
    return (
        int(region["x"]) <= center_x < int(region["x"]) + int(region["w"])
        and int(region["y"]) <= center_y < int(region["y"]) + int(region["h"])
    )


class OnnxBeadDetector:
    def __init__(
        self,
        model_path: Path,
        providers: list[str] | None = None,
    ) -> None:
        providers = providers or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = onnx_image_size(self.session)
        self.model_name = model_path.name
        self.model_metadata = self.session.get_modelmeta().custom_metadata_map or {}

    def run(
        self,
        image: np.ndarray,
        threshold: float = BEAD_DETECTION_SCORE_THRESHOLD,
        detection_region: dict[str, int] | None = None,
        measurement_scale: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
        height, width = image.shape[:2]
        tensor, scale, left, top = prepare_onnx_input(image, self.input_size)
        output = np.asarray(self.session.run(None, {self.input_name: tensor})[0])
        predictions = np.squeeze(output)
        if predictions.ndim == 2 and predictions.shape[0] == 5:
            predictions = predictions.T
        boxes: list[list[int]] = []
        scores: list[float] = []
        if predictions.ndim == 2 and predictions.shape[1] >= 5:
            for cx, cy, box_width, box_height, score, *_ in predictions:
                if not np.isfinite([cx, cy, box_width, box_height, score]).all() or float(score) < threshold:
                    continue
                x1 = int(round((float(cx) - float(box_width) / 2 - left) / scale))
                y1 = int(round((float(cy) - float(box_height) / 2 - top) / scale))
                x2 = int(round((float(cx) + float(box_width) / 2 - left) / scale))
                y2 = int(round((float(cy) + float(box_height) / 2 - top) / scale))
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width - 1, x2), min(height - 1, y2)
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    scores.append(float(score))
        selected = _adaptive_bead_nms(boxes, scores, image.shape)
        candidates = [
            {"bbox": boxes[index], "score": round(scores[index], 4)}
            for index in selected
            if _box_center_in_region(boxes[index], detection_region)
        ]
        detections = [dict(candidate) for candidate in candidates]
        bead_analysis = analyze_bead_detections(image, detections, measurement_scale)
        beads_detected = bool(detections)

        onnx_annotated = image.copy()
        for number, candidate in enumerate(candidates, start=1):
            x1, y1, x2, y2 = candidate["bbox"]
            cv2.rectangle(onnx_annotated, (x1, y1), (x2, y2), (0, 155, 255), 3, cv2.LINE_AA)
            cv2.putText(
                onnx_annotated,
                str(number),
                (x1, max(24, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 155, 255),
                2,
                cv2.LINE_AA,
            )
        onnx_header_width = min(width - 10, 330)
        cv2.rectangle(onnx_annotated, (10, 10), (onnx_header_width, 56), (18, 24, 32), -1)
        cv2.putText(
            onnx_annotated,
            f"ONNX candidates: {len(candidates)}",
            (20, 43),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        annotated = image.copy()
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            category = detection.get("size_category")
            box_color = {
                "tiny": (255, 180, 20),
                "small": (24, 170, 90),
                "large": (190, 60, 210),
            }.get(category, (24, 170, 90))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 3, cv2.LINE_AA)
        bead_status = "Beads detected" if beads_detected else "Beads not detected"
        size_counts = bead_analysis["size"].get("counts") or {}
        detail = (
            f"Large {size_counts.get('large', 0)}  Small {size_counts.get('small', 0)}  "
            f"Tiny {size_counts.get('tiny', 0)}"
            if bead_analysis["size"].get("available") and detections
            else bead_analysis["arrangement"]["description"]
        )
        header_width = min(width - 10, 570)
        cv2.rectangle(annotated, (10, 10), (header_width, 88), (18, 24, 32), -1)
        cv2.putText(
            annotated,
            f"{bead_status}: {len(detections)}",
            (20, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            detail,
            (20, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (210, 220, 230),
            2,
            cv2.LINE_AA,
        )
        return {
            "found": beads_detected,
            "beads_detected": beads_detected,
            "status": bead_status,
            "risk": "High" if beads_detected else "Low",
            "candidate_count": len(candidates),
            "onnx_count": len(candidates),
            "onnx_detections": candidates,
            "detections": detections,
            "count": len(detections),
            "decision_source": "onnx_detector",
            "decision_reason": (
                f"The detector found {len(detections)} candidate"
                f"{'s' if len(detections) != 1 else ''} at confidence {threshold:.2f}."
            ),
            **bead_analysis,
            "model": self.model_name,
            "score_threshold": float(threshold),
            "model_input_size": {
                "height": self.input_size[0],
                "width": self.input_size[1],
                "source": "ONNX imgsz metadata/input tensor",
            },
            "model_metadata": {
                key: self.model_metadata[key]
                for key in ("task", "imgsz", "stride", "names", "version")
                if key in self.model_metadata
            },
            "nms_iou": {
                "small": BEAD_NMS_SMALL_IOU,
                "medium": BEAD_NMS_MEDIUM_IOU,
                "large": BEAD_NMS_LARGE_IOU,
                "size_basis": "maximum box dimension relative to image dimensions",
            },
            "detection_region": detection_region,
        }, annotated, onnx_annotated


class AnalysisService:
    def __init__(self, repository, artifact_finalizer=None) -> None:
        self.repository = repository
        self.artifact_finalizer = artifact_finalizer
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jewellery-analysis")
        self._model_lock = threading.RLock()
        self._classifier: JewelryZeroShotClassifier | None = None
        self._beads: OnnxBeadDetector | None = None
        self._active_capture: str | None = None

    @staticmethod
    def _now() -> str:
        return datetime.now().astimezone().isoformat(timespec="milliseconds")

    def shutdown(self) -> None:
        # A running inference task cannot be cancelled safely midway through
        # native ONNX/OpenCV work. Let it finish, while cancelling work that has
        # not started, so Python does not exit underneath a native worker.
        self._executor.shutdown(wait=True, cancel_futures=True)

    def preload_models(self) -> None:
        """Load every inference model once during application startup."""
        self._get_classifier()
        self._get_beads()

    def _get_classifier(self) -> JewelryZeroShotClassifier:
        with self._model_lock:
            if self._classifier is None:
                self._classifier = JewelryZeroShotClassifier(
                    onnx_model_path=PROJECT_ROOT / "Classification" / "siglip2-base-patch32-256_vision_encoder.sim.onnx",
                    prompt_path=PROJECT_ROOT / "Classification" / "jewelry_prompts.json",
                    embedding_cache_path=PROJECT_ROOT / "Classification" / "jewelry_text_embeddings_cache.npz",
                    providers=self._onnx_providers(),
                )
            return self._classifier

    @staticmethod
    def _onnx_providers() -> list[str]:
        requested = [
            item.strip()
            for item in os.environ.get("JEWELLERY_ONNX_PROVIDERS", "CPUExecutionProvider").split(",")
            if item.strip()
        ]
        available = set(ort.get_available_providers())
        selected = [item for item in requested if item in available]
        return selected or ["CPUExecutionProvider"]

    def _get_beads(self) -> OnnxBeadDetector:
        with self._model_lock:
            if self._beads is None:
                self._beads = OnnxBeadDetector(
                    PROJECT_ROOT / "models" / "detection" / "bead_finder.onnx",
                    providers=self._onnx_providers(),
                )
            return self._beads

    def classify(self, capture_id: str) -> dict[str, Any]:
        state = self.repository.get(capture_id)
        if not state:
            raise KeyError(capture_id)
        image = cv2.imread(state["paths"]["working"])
        mask = cv2.imread(state["paths"]["mask"], cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise RuntimeError("The captured jewellery image or Otsu mask is unavailable")
        settings = (
            (state.get("settings_snapshot") or {}).get("analysis") or {}
        ).get("item_separation") or {}
        separated = separate_jewellery_items(
            image,
            mask,
            min_area_px=int(settings.get("min_area_px", 350)),
            min_area_ratio=float(settings.get("min_area_ratio", 0.00045)),
            padding_px=int(settings.get("padding_px", 18)),
        )
        if not separated:
            raise RuntimeError(
                "No separate jewellery item was found. Keep every jewel fully inside the jewellery area, "
                "leave a visible gap between items, and capture again."
            )

        item_dir = self.repository.session_dir(capture_id) / "items"
        item_dir.mkdir(parents=True, exist_ok=True)
        classifier = self._get_classifier()
        classified_items: list[dict[str, Any]] = []
        for index, separated_item in enumerate(separated, start=1):
            crop_path = item_dir / f"item_{index:02d}.png"
            raw_crop_path = item_dir / f"item_{index:02d}_raw.png"
            mask_path = item_dir / f"item_{index:02d}_mask.png"
            if not cv2.imwrite(str(crop_path), separated_item["crop_bgr"]):
                raise RuntimeError(f"Could not save separated jewel {index}")
            if not cv2.imwrite(str(raw_crop_path), separated_item["raw_crop_bgr"]):
                raise RuntimeError(f"Could not save raw separated jewel {index}")
            cv2.imwrite(str(mask_path), separated_item["mask"] * 255)
            raw_crop_rgb = cv2.cvtColor(
                separated_item["raw_crop_bgr"],
                cv2.COLOR_BGR2RGB,
            )
            foreign_object, semantic_reason = classifier.check_non_jewelry(
                Image.fromarray(raw_crop_rgb)
            )
            if foreign_object:
                raise RuntimeError(
                    "A non-jewellery object was detected in the jewellery area. "
                    "Remove phones, boxes, cases, tools, and other foreign objects, "
                    "then recapture."
                )
            crop_rgb = cv2.cvtColor(separated_item["crop_bgr"], cv2.COLOR_BGR2RGB)
            prediction = classifier.classify_image(Image.fromarray(crop_rgb), image_path=str(crop_path))
            if prediction.decision_source in {
                "obvious_non_gold_color",
                "strong_non_jewelry",
            }:
                raise RuntimeError(
                    "A non-jewellery object was detected in the jewellery area. "
                    "Remove phones, boxes, tools, and other foreign objects, then recapture."
                )
            predicted_label, dimension_fallback = _ring_bangle_dimension_fallback(
                prediction.label,
                separated_item["mask"],
                state.get("calibration"),
            )
            decision_source = prediction.decision_source
            if dimension_fallback["applied"]:
                decision_source = "dimension_ring_bangle_fallback"
            classified_items.append({
                "index": index,
                "bbox": separated_item["bbox"],
                "area_px": int(separated_item["area_px"]),
                "paths": {
                    "raw": str(raw_crop_path),
                    "working": str(crop_path),
                    "prepared": str(crop_path),
                    "mask": str(mask_path),
                },
                "predicted_label": predicted_label,
                "model_label": prediction.model_label,
                "model_confidence": float(prediction.model_confidence),
                "decision_source": decision_source,
                "classifier_label": prediction.label,
                "classifier_decision_source": prediction.decision_source,
                "dimension_fallback": dimension_fallback,
                "semantic_validation_reason": semantic_reason,
                "confirmed_label": None,
                "confirmed": False,
                "confidence": float(prediction.confidence),
                "gallery_match": bool(prediction.gallery_match),
                "gallery_similarity": float(prediction.gallery_similarity),
                "gallery_support": int(prediction.gallery_support),
                "gallery_margin": float(prediction.gallery_margin),
                "gold_verification_reason": prediction.gold_verification_reason,
                "scores": [
                    {
                        "label": score.label,
                        "confidence": float(score.confidence),
                        "similarity": float(score.similarity),
                    }
                    for score in prediction.scores
                ],
            })

        predicted_labels = [str(item["predicted_label"]) for item in classified_items]
        state["classification"] = {
            # These aggregate fields keep old clients and reports readable.
            "predicted_label": ", ".join(predicted_labels),
            "confirmed_label": None,
            "confirmed": False,
            "confidence": float(np.mean([item["confidence"] for item in classified_items])),
            "count": len(classified_items),
            "labels": predicted_labels,
            "items": classified_items,
        }
        annotated_path = self._save_classified_evidence(state, classified_items, confirmed=False)
        state["paths"]["evidence"] = annotated_path
        state["paths"]["classified"] = annotated_path
        state["status"] = "awaiting_confirmation"
        state["updated_at"] = self._now()
        self.repository.save(state)
        return state

    def confirm_and_start(
        self,
        capture_id: str,
        label: str | None,
        learn: bool,
        item_confirmations: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        state = self.repository.get(capture_id)
        if not state:
            raise KeyError(capture_id)
        classification = state.get("classification") or {}
        classified_items = classification.get("items") or []
        if not classified_items:
            raise RuntimeError("Check the item type before confirming it")

        confirmations_by_index = {
            int(item["index"]): item for item in (item_confirmations or [])
        }
        if item_confirmations and len(confirmations_by_index) != len(classified_items):
            raise RuntimeError("Confirm one item type for every detected jewel")
        confirmed_labels: list[str] = []
        routes: list[dict[str, Any]] = []
        for item in classified_items:
            confirmation = confirmations_by_index.get(int(item["index"]))
            confirmed_label = str(
                (confirmation or {}).get("label")
                or (label if len(classified_items) == 1 else "")
                or item.get("predicted_label")
                or ""
            ).strip()
            if not confirmed_label:
                raise RuntimeError(f"Choose an item type for jewel {item['index']}")
            should_learn = bool((confirmation or {}).get("learn", learn))
            if should_learn and confirmed_label != item.get("predicted_label"):
                self._get_classifier().learn_correction(item["paths"]["working"], confirmed_label)
            item["confirmed_label"] = confirmed_label
            item["confirmed"] = True
            item["learned"] = should_learn
            item["route"] = route_for_label(confirmed_label)
            confirmed_labels.append(confirmed_label)
            routes.append(item["route"])

        classification["items"] = classified_items
        classification["confirmed_label"] = ", ".join(confirmed_labels)
        classification["confirmed_labels"] = confirmed_labels
        classification["labels"] = confirmed_labels
        classification["confirmed"] = True
        classification["count"] = len(classified_items)
        state["classification"] = classification
        state["route"] = {
            "key": "multi_item" if len(routes) > 1 else routes[0]["key"],
            "dimension": any(route.get("dimension") for route in routes),
            "beads": any(route.get("beads") for route in routes),
            "stones": any(route.get("stones") for route in routes),
            "items": routes,
        }
        annotated_path = self._save_classified_evidence(state, classified_items, confirmed=True)
        state["paths"]["evidence"] = annotated_path
        state["status"] = "processing"
        state["job"] = {
            "status": "queued",
            "stage": "Preparing",
            "message": "Preparing your results",
            "percent": 1,
            "seconds_remaining": self._estimated_total(state["route"]),
            "stage_started_epoch": time.time(),
            "stage_estimate": self._estimated_total(state["route"]),
            "error": None,
        }
        state["updated_at"] = self._now()
        self.repository.save(state)
        self._active_capture = capture_id
        self._executor.submit(self._run, capture_id)
        return state

    def _save_classified_evidence(
        self,
        state: dict[str, Any],
        items: list[dict[str, Any]],
        *,
        confirmed: bool,
    ) -> str:
        image = cv2.imread(state["paths"]["original"])
        if image is None:
            raise RuntimeError("The original captured image is unavailable")
        processing_roi = (state.get("rois") or {}).get("processing")
        offset_x = int((processing_roi or {}).get("x", 0))
        offset_y = int((processing_roi or {}).get("y", 0))
        scale = max(0.65, min(image.shape[:2]) / 900.0)
        dark = (32, 24, 17)
        gold = (36, 129, 189)
        green = (95, 132, 22)
        white = (255, 255, 255)

        def fitted_font(text: str, maximum_width: int, preferred: float, thickness: int) -> float:
            font_size = preferred
            text_width = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_size, thickness
            )[0][0]
            if text_width > maximum_width:
                font_size *= max(0.45, maximum_width / max(1, text_width))
            return font_size

        labels: list[str] = []
        for item in items:
            bbox = item["bbox"]
            x1 = int(bbox["x"]) + offset_x
            y1 = int(bbox["y"]) + offset_y
            x2 = x1 + int(bbox["w"])
            y2 = y1 + int(bbox["h"])
            label = str(
                item.get("confirmed_label") if confirmed else item.get("predicted_label")
            )
            labels.append(label)
            line_thickness = max(3, int(4 * scale))
            cv2.rectangle(image, (x1, y1), (x2, y2), green, line_thickness, cv2.LINE_AA)
            text = f"JEWEL {item['index']}  |  {label.upper()}"
            text_thickness = max(2, int(2 * scale))
            font_size = fitted_font(text, max(140, x2 - x1 - 28), 0.72 * scale, text_thickness)
            (_, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_size, text_thickness
            )
            tag_height = text_height + baseline + max(18, int(18 * scale))
            tag_width = min(
                image.shape[1] - x1,
                cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_size, text_thickness)[0][0]
                + max(28, int(28 * scale)),
            )
            text_top = max(0, y1 - tag_height)
            cv2.rectangle(image, (x1, text_top), (x1 + tag_width, y1), dark, -1)
            cv2.rectangle(image, (x1, text_top), (x1 + max(7, int(8 * scale)), y1), gold, -1)
            cv2.putText(
                image,
                text,
                (x1 + max(17, int(19 * scale)), y1 - baseline - max(6, int(7 * scale))),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_size,
                white,
                text_thickness,
                cv2.LINE_AA,
            )

        captured_at = datetime.fromisoformat(str(state["captured_at"]))
        stamp = captured_at.strftime("%d-%m-%Y  %I:%M:%S %p")
        weight = state.get("weight_g")
        weight_text = f"{float(weight):.2f} g" if weight is not None else "Unavailable"
        type_text = ", ".join(labels).upper()
        panel_height = max(112, int(138 * scale))
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (image.shape[1], panel_height), dark, -1)
        cv2.addWeighted(overlay, 0.96, image, 0.04, 0, image)
        cv2.rectangle(image, (0, panel_height - max(5, int(6 * scale))), (image.shape[1], panel_height), gold, -1)

        margin = max(18, int(22 * scale))
        gap = max(18, int(24 * scale))
        available = image.shape[1] - (2 * margin) - (2 * gap)
        type_width = int(available * 0.43)
        weight_width = int(available * 0.22)
        date_width = available - type_width - weight_width
        columns = (
            (margin, type_width, "JEWELLERY TYPE", type_text),
            (margin + type_width + gap, weight_width, "CAPTURED WEIGHT", weight_text),
            (margin + type_width + weight_width + (2 * gap), date_width, "DATE & TIME", stamp),
        )
        label_size = max(0.42, 0.48 * scale)
        value_size = max(0.66, 0.82 * scale)
        label_thickness = max(1, int(2 * scale))
        value_thickness = max(2, int(2.4 * scale))
        label_y = max(30, int(38 * scale))
        value_y = panel_height - max(24, int(30 * scale))
        for column_index, (left, width, caption, value) in enumerate(columns):
            if column_index:
                divider_x = left - (gap // 2)
                cv2.line(image, (divider_x, margin), (divider_x, panel_height - margin), (82, 86, 92), 1, cv2.LINE_AA)
            cv2.putText(image, caption, (left, label_y), cv2.FONT_HERSHEY_SIMPLEX, label_size, gold, label_thickness, cv2.LINE_AA)
            fitted = fitted_font(value, width, value_size, value_thickness)
            cv2.putText(image, value, (left, value_y), cv2.FONT_HERSHEY_SIMPLEX, fitted, white, value_thickness, cv2.LINE_AA)
        path = self.repository.session_dir(state["id"]) / "capture" / "classified_evidence.jpg"
        if not cv2.imwrite(str(path), image):
            raise RuntimeError("Could not save the classified capture image")
        return str(path)

    @staticmethod
    def _estimated_total(route: dict[str, Any]) -> int:
        item_routes = route.get("items") or [route]
        return max(3, sum(
            (8 if item_route.get("dimension") else 0)
            + (7 if item_route.get("beads") else 0)
            + (35 if item_route.get("stones") else 0)
            for item_route in item_routes
        ))

    def public_state(self, state: dict[str, Any]) -> dict[str, Any]:
        payload = json.loads(json.dumps(state))
        job = payload.get("job") or {}
        if job.get("status") in {"queued", "running"}:
            deadline = float(job.get("stage_started_epoch", time.time())) + float(job.get("stage_estimate", 0))
            job["seconds_remaining"] = max(0, int(math.ceil(deadline - time.time())))
        return payload

    def _update_job(
        self,
        state: dict[str, Any],
        stage: str,
        message: str,
        percent: int,
        estimate: int,
    ) -> None:
        state["job"] = {
            "status": "running",
            "stage": stage,
            "message": message,
            "percent": percent,
            "seconds_remaining": estimate,
            "stage_started_epoch": time.time(),
            "stage_estimate": estimate,
            "error": None,
        }
        state["updated_at"] = self._now()
        self.repository.save(state)

    @staticmethod
    def _save_json(path: Path, payload: dict[str, Any]) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return str(path)

    def _run_dimension(self, state: dict[str, Any], item: dict[str, Any] | None = None) -> dict[str, Any]:
        calibration = state.get("calibration") or {}
        mm_per_pixel = _apriltag_scale_mm_per_pixel(calibration)
        paths = (item or {}).get("paths") or state["paths"]
        label = str((item or {}).get("confirmed_label") or state["classification"]["confirmed_label"])
        suffix = f"item_{int(item['index']):02d}" if item else "dimensions"
        result = detect_bangle(
            paths["working"],
            scale=mm_per_pixel,
            debug=False,
            jewel_type=label,
        )
        output = self.repository.session_dir(state["id"]) / "results" / "dimensions" / f"{suffix}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        annotated = cv2.imread(str(result["annotated_path"]))
        if annotated is not None:
            cv2.imwrite(str(output), annotated)
        return {
            "outer_diameter_mm": result.get("od_mm"),
            "inner_diameter_mm": result.get("id_mm"),
            "thickness_mm": result.get("wall_thickness_mm"),
            "calibration_source": "AprilTag",
            "apriltag_id": calibration.get("id"),
            "mm_per_pixel": mm_per_pixel,
            "image": str(output) if output.exists() else result.get("annotated_path"),
        }

    def _run_beads(self, state: dict[str, Any], item: dict[str, Any] | None = None) -> dict[str, Any]:
        # Run on the complete configured jewellery ROI, letterboxed to the
        # input size declared by the ONNX model. A tight per-item crop changes
        # bead scale and context, so detections are spatially assigned afterward.
        input_path = state["paths"]["working"]
        image = cv2.imread(input_path)
        if image is None:
            raise RuntimeError("Could not load the captured jewellery image")
        calibration = state.get("calibration") or {}
        measurement_scale = None
        if calibration.get("available") and calibration.get("found"):
            measurement_scale = {
                "mm_per_pixel_x": calibration.get("mm_per_pixel_x"),
                "mm_per_pixel_y": calibration.get("mm_per_pixel_y"),
            }
        result, annotated, onnx_annotated = self._get_beads().run(
            image,
            threshold=BEAD_DETECTION_SCORE_THRESHOLD,
            detection_region=(item or {}).get("bbox"),
            measurement_scale=measurement_scale,
        )
        suffix = f"item_{int(item['index']):02d}" if item else "beads"
        path = self.repository.session_dir(state["id"]) / "results" / "beads" / f"{suffix}.png"
        onnx_path = self.repository.session_dir(state["id"]) / "results" / "beads" / f"{suffix}_onnx.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), annotated)
        cv2.imwrite(str(onnx_path), onnx_annotated)
        result["image"] = str(path)
        result["onnx_image"] = str(onnx_path)
        result["input_source"] = "configured_roi"
        return result

    def _run_stones(self, state: dict[str, Any], item: dict[str, Any] | None = None) -> dict[str, Any]:
        # Keep the real weighing-platform pixels available to the stone
        # pipeline.  The prepared image replaces everything outside the Otsu
        # mask with pure white, which prevents the background-match cleanup
        # from learning the platform colour and rejecting background pixels
        # that leaked inside the mask.  The external mask below remains the
        # hard boundary for all jewellery and stone regions.
        paths = (item or {}).get("paths") or state["paths"]
        image = cv2.imread(paths["working"])
        mask = cv2.imread(paths["mask"], cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise RuntimeError("Jewellery image or mask is unavailable")
        mask = (mask > 0).astype(np.uint8)
        preset_candidates = stone_detection.build_candidates_from_component_mask(
            image,
            mask,
            min_area=max(80, int(mask.size * 0.0005)),
            max_candidates=12,
            reject_border_touching=False,
            min_area_ratio_to_largest=0.01,
        )
        preset_candidates = [
            candidate
            for candidate in preset_candidates
            if stone_detection.has_enough_gold_for_stone_detection(
                candidate["crop_bgr"], candidate["crop_mask"]
            )
        ]
        analysis_settings = (
            state.get("analysis_settings")
            or (state.get("settings_snapshot") or {}).get("analysis")
            or {}
        )
        calibration = state.get("calibration") or {}
        measurement_scale = None
        if calibration.get("available") and calibration.get("found"):
            try:
                scale_x = float(calibration["mm_per_pixel_x"])
                scale_y = float(calibration["mm_per_pixel_y"])
                if np.isfinite([scale_x, scale_y]).all() and scale_x > 0 and scale_y > 0:
                    measurement_scale = {
                        "mm_per_pixel_x": scale_x,
                        "mm_per_pixel_y": scale_y,
                    }
            except (KeyError, TypeError, ValueError):
                measurement_scale = None
        analysis = stone_detection.analyze_image_bgr(
            image,
            source_name=paths["prepared"],
            zoom_scale=1,
            preset_candidates=preset_candidates,
            external_mask=mask,
            color_correction=analysis_settings.get("color_correction"),
            background_calibration=analysis_settings.get("background_calibration"),
            analysis_normalization=analysis_settings.get("normalization"),
            # Convert the final stone union from pixels to mm² using the same
            # AprilTag calibration used by dimension analysis.  The stone
            # risk level uses this calibrated face-up area when available.
            measurement_scale=measurement_scale,
            learned_stone_profiles=analysis_settings.get("learned_stone_profiles"),
            # OpenCV refinement keeps every accepted color-region candidate
            # without running a separate neural segmentation for each stone.
            fastsam_model=None,
            fastsam_lock=None,
        )
        report = analysis["report"]
        classification = state.get("classification") or {}
        classified_items = classification.get("items") or []
        jewel_count = int(classification.get("count") or len(classified_items) or 1)
        estimate_weight = jewel_count == 1
        suffix = f"item_{int(item['index']):02d}" if item else "capture"
        output_dir = self.repository.session_dir(state["id"]) / "results" / "stones" / suffix
        output_dir.mkdir(parents=True, exist_ok=True)
        gallery_path = output_dir / "stones.png"
        cv2.imwrite(str(gallery_path), analysis["result_gallery_bgr"])
        report_path = self._save_json(
            output_dir / "result.json",
            report if estimate_weight else without_stone_weight_estimates(report),
        )
        risk = report.get("stone_surface_risk") or {}
        measurements = report.get("stone_measurements") or {}
        stones_found = str(risk.get("level") or "NONE") != "NONE"
        result = {
            # Risk NONE includes calibrated sub-5 mm² regions suppressed as
            # likely reflections/texture rather than reported as real stones.
            "found": stones_found,
            "risk_level": str(risk.get("level") or "NONE"),
            "risk_status": str(risk.get("status") or "NO RISK - NO STONES DETECTED"),
            "stone_area_mm2": measurements.get("total_area_mm2") if stones_found else None,
            "measurement_basis": risk.get("basis"),
            "image": str(gallery_path),
            "report": report_path,
        }
        if estimate_weight:
            result.update(stone_weight_fields(measurements, stones_found))
        return result

    def _run(self, capture_id: str) -> None:
        state = self.repository.get(capture_id)
        if not state:
            return
        try:
            classified_items = (state.get("classification") or {}).get("items") or []
            result_items: list[dict[str, Any]] = []
            completed_steps = 0
            total_steps = max(1, sum(
                int(bool((item.get("route") or {}).get(key)))
                for item in classified_items
                for key in ("dimension", "beads", "stones")
            ))
            for item in classified_items:
                item_result: dict[str, Any] = {
                    "index": int(item["index"]),
                    "label": item["confirmed_label"],
                    "bbox": item["bbox"],
                    "route": item["route"],
                    "errors": [],
                }
                route = item["route"]
                operations = (
                    ("dimensions", "Measuring", "Measuring jewel size", 8, self._run_dimension),
                    ("beads", "Checking beads", "Detecting round beads", 7, self._run_beads),
                    ("stones", "Checking stones", "Detecting stone regions", 35, self._run_stones),
                )
                for key, stage, message, estimate, operation in operations:
                    route_key = "dimension" if key == "dimensions" else key
                    if not route.get(route_key):
                        continue
                    percent = min(95, 5 + int((completed_steps / total_steps) * 90))
                    self._update_job(
                        state,
                        stage,
                        f"Jewel {item['index']}/{len(classified_items)}: {message}",
                        percent,
                        estimate,
                    )
                    try:
                        item_result[key] = operation(state, item)
                    except Exception as exc:  # Keep mixed-item analysis progressing.
                        item_result["errors"].append({"stage": key, "message": str(exc)})
                    completed_steps += 1
                item_result["status"] = "complete" if not item_result["errors"] else "partial"
                result_items.append(item_result)

            result: dict[str, Any] = {
                "count": len(result_items),
                "labels": [item["label"] for item in result_items],
                "items": result_items,
            }
            # Preserve the original single-item response shape for API clients.
            if len(result_items) == 1:
                for key in ("dimensions", "beads", "stones"):
                    if key in result_items[0]:
                        result[key] = result_items[0][key]
            state = self.repository.get(capture_id) or state
            state["result"] = result
            if len(result_items) == 1:
                result["weights"] = weight_summary(state)
            if self.artifact_finalizer is not None:
                self._update_job(
                    state,
                    "Optimizing storage",
                    "Compressing completed images and report",
                    97,
                    3,
                )
                try:
                    state["storage"] = self.artifact_finalizer.finalize(state)
                except Exception as exc:  # Analysis results remain usable if optimization fails.
                    state["storage"] = {
                        "status": "partial",
                        "lossless": True,
                        "processed_after_results": True,
                        "processed_after_analysis": True,
                        "errors": [{"path": capture_id, "message": str(exc)}],
                        "presentation_artifacts": "compressed_on_demand",
                    }
            state["status"] = "complete"
            warnings = sum(len(item.get("errors") or []) for item in result_items)
            state["job"] = {
                "status": "complete",
                "stage": "Complete",
                "message": "Your results are ready" if not warnings else f"Results ready with {warnings} analysis warning(s)",
                "percent": 100,
                "seconds_remaining": 0,
                "error": None,
            }
            state["updated_at"] = self._now()
            self.repository.save(state)
        except Exception as exc:  # noqa: BLE001
            state = self.repository.get(capture_id) or state
            state["status"] = "failed"
            state["job"] = {
                "status": "failed",
                "stage": "Stopped",
                "message": "The jewellery could not be fully checked",
                "percent": int((state.get("job") or {}).get("percent", 0)),
                "seconds_remaining": 0,
                "error": str(exc),
            }
            state["updated_at"] = self._now()
            self.repository.save(state)
        finally:
            self._active_capture = None
