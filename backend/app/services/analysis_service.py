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

from ..analysis.annotations import (identify_result_image, jewel_reference,
                                    render_bead_evidence, render_capture_review)
from ..analysis.bangle_orientation import detect_bangle_orientation
from ..analysis.bangle_placement import check_bangle_placement, PLACEMENT_MESSAGE
from ..analysis.beads import analyze_bead_detections, onnx_image_size, prepare_onnx_input
from ..analysis.vision import separate_jewellery_items
from ..analysis.foreign_objects import ForeignObjectDetector
from ..core.config import PROJECT_ROOT, FOREIGN_OBJECT_MODEL_PATH
from ..domain.workflow import label_key, route_for_label
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

from Classification.jewelry_classifier import (  # noqa: E402
    JewelryZeroShotClassifier, NOT_GOLD_LABEL, PredictionResult,
)
from Classification.siglip384_classifier import Siglip384Classifier  # noqa: E402
from Dimension.bangle_detector import detect_bangle, gold_bangle_mask  # noqa: E402
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
    model_sources: list[str] | None = None,
) -> list[int]:
    """Greedy NMS with an IoU threshold derived from each bead's image-relative size."""
    selected: list[int] = []
    for candidate_index in sorted(range(len(scores)), key=scores.__getitem__, reverse=True):
        candidate_box = boxes[candidate_index]
        candidate_iou = _bead_nms_iou_threshold(candidate_box, image_shape)
        duplicate = False
        for selected_index in selected:
            selected_iou = _bead_nms_iou_threshold(boxes[selected_index], image_shape)
            iou_threshold = max(candidate_iou, selected_iou)
            if model_sources and model_sources[candidate_index] != model_sources[selected_index]:
                # Cross-model duplicates need stricter suppression even for large beads.
                iou_threshold = BEAD_NMS_SMALL_IOU
            if _box_iou(candidate_box, boxes[selected_index]) > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            selected.append(candidate_index)
    return selected


def _box_center_in_region(bbox: list[int], region: dict[str, int] | None) -> bool:
    """Keep a full-frame detection only when its centre belongs to the requested jewel."""
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
        additional_model_paths: list[Path] | None = None,
    ) -> None:
        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = onnx_image_size(self.session)
        self.model_name = model_path.name
        self.model_metadata = self.session.get_modelmeta().custom_metadata_map or {}
        self.additional_detectors = [
            OnnxBeadDetector(path, providers=providers)
            for path in (additional_model_paths or [])
        ]

    def _predict(
        self, image: np.ndarray, threshold: float,
    ) -> tuple[list[list[int]], list[float]]:
        """Decode this model's boxes into the original raw image coordinates."""
        height, width = image.shape[:2]
        tensor, scale, left, top = prepare_onnx_input(image, self.input_size)
        output = np.asarray(self.session.run(None, {self.input_name: tensor})[0])
        predictions = output[0] if output.ndim == 3 and output.shape[0] == 1 else output
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
        return boxes, scores

    def run(
        self,
        image: np.ndarray,
        threshold: float = BEAD_DETECTION_SCORE_THRESHOLD,
        detection_region: dict[str, int] | None = None,
        measurement_scale: dict[str, Any] | None = None,
        detection_mask: np.ndarray | None = None,
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
        height, width = image.shape[:2]
        if detection_mask is not None and detection_mask.shape != (height, width):
            raise ValueError("Bead detection mask must match the full-frame image dimensions")
        boxes: list[list[int]] = []
        scores: list[float] = []
        model_sources: list[str] = []
        models = [self, *self.additional_detectors]
        model_results: list[dict[str, Any]] = []
        for detector in models:
            # Every model receives the same untouched full-frame image, with
            # its own metadata-derived letterbox size (960 or 640).
            model_boxes, model_scores = detector._predict(image, threshold)
            accepted = 0
            for box, score in zip(model_boxes, model_scores):
                if not _box_center_in_region(box, detection_region):
                    continue
                if detection_mask is not None and detection_mask[
                    (box[1] + box[3]) // 2, (box[0] + box[2]) // 2,
                ] <= 0:
                    continue
                boxes.append(box)
                scores.append(score)
                model_sources.append(detector.model_name)
                accepted += 1
            model_results.append({
                "model": detector.model_name,
                "model_input_size": {
                    "height": detector.input_size[0], "width": detector.input_size[1],
                },
                "region_candidate_count": accepted,
            })
        selected = _adaptive_bead_nms(boxes, scores, image.shape, model_sources)
        candidates = [
            {"bbox": boxes[index], "score": round(scores[index], 4),
             "model": model_sources[index]}
            for index in selected
        ]
        detections = [dict(candidate) for candidate in candidates]
        bead_analysis = analyze_bead_detections(image, detections, measurement_scale)
        beads_detected = bool(detections)

        # Rendering happens only after inference, NMS and size analysis finish.
        annotated = render_bead_evidence(image, detections, bead_analysis, detection_region)
        onnx_annotated = render_bead_evidence(image, detections, bead_analysis, detection_region,
                                             title="BEAD DETECTION")
        bead_status = "Beads detected" if beads_detected else "Beads not detected"
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
            "decision_source": "onnx_ensemble" if len(models) > 1 else "onnx_detector",
            "decision_reason": (
                f"The {len(models)} detector(s) found {len(detections)} merged candidate"
                f"{'s' if len(detections) != 1 else ''} at confidence {threshold:.2f}."
            ),
            **bead_analysis,
            "models": model_results,
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
                "cross_model": BEAD_NMS_SMALL_IOU,
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
        self._foreign_objects: ForeignObjectDetector | None = None
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
        """Load all analysis gates before accepting captures."""
        self._get_classifier()
        self._get_beads()
        self._get_foreign_objects()

    def _get_classifier(self) -> JewelryZeroShotClassifier:
        with self._model_lock:
            if self._classifier is None:
                model = os.environ.get("JEWELLERY_CLASSIFIER_MODEL", "base384").strip().lower()
                if model == "base384":
                    self._classifier = Siglip384Classifier(providers=self._onnx_providers())
                elif model == "patch32":
                    self._classifier = JewelryZeroShotClassifier(
                        onnx_model_path=PROJECT_ROOT / "Classification" / "siglip2-base-patch32-256_vision_encoder.sim.onnx",
                        prompt_path=PROJECT_ROOT / "Classification" / "jewelry_prompts.json",
                        embedding_cache_path=PROJECT_ROOT / "Classification" / "jewelry_text_embeddings_cache.npz",
                        providers=self._onnx_providers(),
                    )
                else:
                    raise ValueError("JEWELLERY_CLASSIFIER_MODEL must be 'base384' or 'patch32'.")
            return self._classifier

    @staticmethod
    def _onnx_providers() -> list[str]:
        return ["CPUExecutionProvider"]

    def _get_beads(self) -> OnnxBeadDetector:
        with self._model_lock:
            if self._beads is None:
                self._beads = OnnxBeadDetector(
                    PROJECT_ROOT / "models" / "detection" / "bead_finder.onnx",
                    providers=self._onnx_providers(),
                    additional_model_paths=[
                        PROJECT_ROOT / "models" / "detection" / "beadfinder_old.onnx",
                    ],
                )
            return self._beads

    def _get_foreign_objects(self) -> ForeignObjectDetector:
        with self._model_lock:
            if self._foreign_objects is None:
                self._foreign_objects = ForeignObjectDetector(FOREIGN_OBJECT_MODEL_PATH)
            return self._foreign_objects

    def _check_foreign_objects(self, state: dict[str, Any], separated: list[dict] | None = None) -> dict[str, Any]:
        paths = state["paths"]
        raw = cv2.imread(paths.get("original", ""))
        testbed_mask = cv2.imread(paths.get("testbed_source_mask", ""), cv2.IMREAD_GRAYSCALE)
        if raw is None or testbed_mask is None:
            raise RuntimeError("The raw capture or full-frame testbed mask is unavailable; recapture.")
        if separated is None:
            return self._get_foreign_objects().detect(raw, testbed_mask)
        origin = state["rois"]["processing"]
        components = [
            {"bbox": {**item["bbox"], "x": item["bbox"]["x"] + origin["x"],
                      "y": item["bbox"]["y"] + origin["y"]}, "mask": item["mask"]}
            for item in separated
        ]
        return self._get_foreign_objects().detect(raw, testbed_mask, components=components)

    def _check_bangle_placement(self, item: dict[str, Any], label: str) -> dict[str, Any]:
        if label_key(label) != "bangle":
            item.pop("placement_check", None)
            return {"passed": True, "method": "not_a_bangle"}
        path = (item.get("paths") or {}).get("mask")
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE) if path else None
        evidence = check_bangle_placement(mask)
        item["placement_check"] = evidence
        return evidence

    def _reject_bangle_placement(self, state: dict[str, Any], items: list[dict[str, Any]]) -> None:
        affected = [{"item_index": int(item["index"]), "bbox": item["bbox"],
                     **item["placement_check"]} for item in items
                    if not (item.get("placement_check") or {}).get("passed", True)]
        state["placement_check"] = {"passed": False, "method": "bangle_opening_geometry",
                                    "affected_items": affected}
        state["rejection_kind"] = "placement"
        state["rejection_reason"] = PLACEMENT_MESSAGE
        state["route"] = {}
        state["result"] = {}
        # A merged component is not a reliable jewel count. Keep its evidence
        # separately, rather than presenting it as a confirmed single jewel.
        state["classification"] = {"count": 0, "items": [], "labels": [],
                                   "confirmed": False, "predicted_label": "",
                                   "confirmed_label": None}
        state["status"] = "rejected"
        state["updated_at"] = self._now()
        self.repository.save(state)
        self._save_rejected_review(state)

    def _route_for_item(self, item: dict[str, Any], label: str) -> dict[str, Any]:
        if label_key(label) != "bangle":
            item.pop("bangle_orientation", None)
            return route_for_label(label)
        paths = item.get("paths") or {}
        mask = cv2.imread(paths["mask"], cv2.IMREAD_GRAYSCALE) if paths.get("mask") else None
        image = cv2.imread(paths["working"]) if paths.get("working") else None
        evidence = detect_bangle_orientation(
            mask, gold_mask=gold_bangle_mask(image) if image is not None else None
        )
        item["bangle_orientation"] = evidence
        return route_for_label(label, bangle_orientation=evidence["orientation"])

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
        check = self._check_foreign_objects(state, separated)
        state["foreign_object_check"] = check
        foreign_indices = set(check.get("foreign_component_indices", []))
        state["other_objects"] = [
            {"source_component_index": index, "bbox": item["bbox"], "kind": "foreign_object"}
            for index, item in enumerate(separated, 1) if index in foreign_indices
        ]
        separated = [item for index, item in enumerate(separated, 1) if index not in foreign_indices]
        placement = check.get("placement_check") or {"passed": True}
        if not placement["passed"] or (not check["passed"] and not separated):
            state["rejection_kind"] = "foreign_objects" if not check["passed"] else "placement"
            labels = ", ".join(sorted({d["label"].replace("_", " ") for d in check["detections"]}))
            state["rejection_reason"] = (
                f"Other objects detected ({labels}). Remove them and recapture."
                if not check["passed"] else
                "Jewels are kept together. Keep them well separated and recapture the image."
            )
            # No reliable separated jewellery remains to classify. Do not
            # send contaminants or merged pieces to SigLIP as individual jewels.
            state["classification"] = {"count": 0, "items": [], "labels": [],
                                       "confirmed": False, "predicted_label": "",
                                       "confirmed_label": None}
            state["status"] = "rejected"
            state["updated_at"] = self._now()
            self.repository.save(state)
            self._save_rejected_review(state)
            return state
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
                # Keep the raw-image eliminator's decision, but finish checking
                # the other separated objects so mixed captures can be reviewed.
                raw_image = Image.fromarray(raw_crop_rgb)
                prediction = PredictionResult(
                    label=NOT_GOLD_LABEL, confidence=1.0, scores=[],
                    image_path=str(crop_path), original_image=raw_image,
                    cropped_image=raw_image, is_gold_jewelry=False,
                    gold_verification_reason=semantic_reason,
                    decision_source="strong_non_jewelry",
                )
            else:
                crop_rgb = cv2.cvtColor(separated_item["crop_bgr"], cv2.COLOR_BGR2RGB)
                prediction = classifier.classify_image(Image.fromarray(crop_rgb), image_path=str(crop_path))
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

        for item in classified_items:
            if label_key(item["predicted_label"]) == "bangle":
                self._route_for_item(item, item["predicted_label"])

        non_jewellery = [item for item in classified_items
                         if route_for_label(item["predicted_label"])["key"] == "no_analysis"]
        state["other_objects"].extend(
            {"bbox": item["bbox"], "kind": "non_jewellery",
             "reason": item["gold_verification_reason"]} for item in non_jewellery
        )
        classified_items = [item for item in classified_items if item not in non_jewellery]
        for index, item in enumerate(classified_items, 1):
            item["index"] = index
        predicted_labels = [str(item["predicted_label"]) for item in classified_items]
        state["classification"] = {
            # These aggregate fields keep old clients and reports readable.
            "model_id": getattr(classifier, "text_model_id", ""),
            "predicted_label": ", ".join(predicted_labels),
            "confirmed_label": None,
            "confirmed": False,
            "confidence": float(np.mean([item["confidence"] for item in classified_items])) if classified_items else 0.0,
            "count": len(classified_items),
            "labels": predicted_labels,
            "items": classified_items,
        }
        if not check["passed"] or non_jewellery:
            state["rejection_kind"] = "foreign_objects" if not check["passed"] else "non_jewellery"
            state["status"] = "rejected"
            state["rejection_reason"] = "Remove the other objects from the testbed and recapture."
            state["updated_at"] = self._now()
            self.repository.save(state)
            self._save_rejected_review(state)
            return state
        for item in classified_items:
            self._check_bangle_placement(item, item["predicted_label"])
        if any(not (item.get("placement_check") or {}).get("passed", True) for item in classified_items):
            self._reject_bangle_placement(state, classified_items)
            return state
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
        override_not_gold: bool = False,
        save_only: bool = False,
    ) -> dict[str, Any]:
        state = self.repository.get(capture_id)
        if not state:
            raise KeyError(capture_id)
        if state.get("status") == "rejected" or (state.get("foreign_object_check") or {}).get("passed") is False:
            raise RuntimeError(state.get("rejection_reason") or "Foreign objects detected; recapture.")
        classification = state.get("classification") or {}
        classified_items = classification.get("items") or []
        if not classified_items:
            raise RuntimeError("Check the item type before confirming it")

        if save_only and state.get("status") != "awaiting_confirmation":
            raise RuntimeError("Corrections can only be saved while awaiting confirmation")

        confirmations_by_index = {
            int(item["index"]): item for item in (item_confirmations or [])
        }
        if item_confirmations and (
            len(item_confirmations) != len(classified_items)
            or set(confirmations_by_index) != {int(item["index"]) for item in classified_items}
        ):
            raise RuntimeError("Confirm one item type for every detected jewel")
        pending_confirmations: list[tuple[dict[str, Any], str, bool, dict[str, Any]]] = []
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
            if (
                route_for_label(str(item.get("predicted_label") or ""))["key"] == "no_analysis"
                and route_for_label(confirmed_label)["key"] != "no_analysis"
            ):
                raise RuntimeError(
                    f"Jewel {item['index']} was classified as Not Gold Jewelry and cannot be overwritten. "
                    "Remove the object and recapture."
                )
            should_learn = bool((confirmation or {}).get("learn", learn))
            pending_confirmations.append(
                (item, confirmed_label, should_learn, self._route_for_item(item, confirmed_label))
            )

        rejected_predictions = [
            int(item["index"])
            for item in classified_items
            if route_for_label(str(item.get("predicted_label") or ""))["key"] == "no_analysis"
        ]
        rejected_confirmations = [
            int(item["index"])
            for item, confirmed_label, _should_learn, _route in pending_confirmations
            if route_for_label(confirmed_label)["key"] == "no_analysis"
        ]
        if rejected_confirmations and not save_only:
            jewels = ", ".join(str(index) for index in rejected_confirmations)
            raise RuntimeError(
                f"Jewel(s) {jewels} are confirmed as Not Gold Jewelry. "
                "Remove the object and recapture."
            )
        if rejected_predictions and not save_only:
            jewels = ", ".join(str(index) for index in rejected_predictions)
            raise RuntimeError(
                f"Jewel(s) {jewels} were classified as Not Gold Jewelry. "
                "Detailed analysis is blocked; remove the object and recapture."
            )

        for item, confirmed_label, _learn, _route in pending_confirmations:
            self._check_bangle_placement(item, confirmed_label)
        if any(not (item.get("placement_check") or {}).get("passed", True)
               for item, _label, _learn, _route in pending_confirmations):
            self._reject_bangle_placement(state, classified_items)
            raise RuntimeError(PLACEMENT_MESSAGE)

        confirmed_labels: list[str] = []
        routes: list[dict[str, Any]] = []
        for item, confirmed_label, should_learn, item_route in pending_confirmations:
            learn_rejection = save_only and item_route["key"] == "no_analysis"
            if (
                (learn_rejection or (should_learn and confirmed_label != item.get("predicted_label")))
                and confirmed_label != item.get("gallery_saved_label")
            ):
                self._get_classifier().learn_correction(item["paths"]["working"], confirmed_label)
                item["gallery_saved_label"] = confirmed_label
            if save_only:
                item["correction_label"] = confirmed_label
                continue
            item["confirmed_label"] = confirmed_label
            item["confirmed"] = True
            item["learned"] = should_learn
            item["route"] = item_route
            confirmed_labels.append(confirmed_label)
            routes.append(item["route"])

        if save_only:
            state["classification"] = classification
            state["updated_at"] = self._now()
            self.repository.save(state)
            return state

        classification["items"] = classified_items
        classification["confirmed_label"] = ", ".join(confirmed_labels)
        classification["confirmed_labels"] = confirmed_labels
        classification["labels"] = confirmed_labels
        classification["confirmed"] = True
        classification["not_gold_override"] = bool(rejected_predictions)
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

    def _save_rejected_review(self, state: dict[str, Any]) -> None:
        """Best-effort display of an existing rejection; never change its gate."""
        original = (state.get("paths") or {}).get("original")
        if not original:
            return
        try:
            photo = cv2.imread(str(original))
            if photo is None:
                return
            rendered = render_capture_review(photo, state, (state.get("classification") or {}).get("items") or [], rejected=True)
            path = self.repository.session_dir(state["id"]) / "capture" / "classified_evidence.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            if cv2.imwrite(str(path), rendered):
                state["paths"]["evidence"] = str(path)
                self.repository.save(state)
        except Exception:  # Display failures must never replace the existing rejection.
            # An export failure must not replace or bypass the original rejection.
            return

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
        image = render_capture_review(image, state, items, confirmed=confirmed)
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
        paths = (item or {}).get("paths") or state["paths"]
        label = str((item or {}).get("confirmed_label") or state["classification"]["confirmed_label"])
        if label_key(label) in {"finger ring", "ring"}:
            return {"applicable": False, "status": "skipped", "reason": "Analysis is disabled for finger rings."}
        if label_key(label) == "bangle":
            target = item if item is not None else {"paths": paths}
            self._route_for_item(target, label)
            orientation = target["bangle_orientation"]
            if not orientation["measurement_allowed"]:
                return {"applicable": False, "status": "skipped",
                        "reason": orientation["reason"], "bangle_orientation": orientation}
        suffix = f"item_{int(item['index']):02d}" if item else "dimensions"

        mm_per_pixel = _apriltag_scale_mm_per_pixel(calibration)
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
            annotated = identify_result_image(annotated, jewel_reference(item))
            cv2.imwrite(str(output), annotated)
        return {
            "applicable": True,
            "outer_diameter_mm": result.get("od_mm"),
            "inner_diameter_mm": result.get("id_mm"),
            "thickness_mm": result.get("wall_thickness_mm"),
            "calibration_source": "AprilTag",
            "apriltag_id": calibration.get("id"),
            "mm_per_pixel": mm_per_pixel,
            "image": str(output) if output.exists() else result.get("annotated_path"),
        }

    def _run_beads(self, state: dict[str, Any], item: dict[str, Any] | None = None) -> dict[str, Any]:
        # Preserve the full-frame scale/context used for training. Apply the
        # testbed polygon and per-jewel bounds only to the decoded predictions.
        input_path = state["paths"]["original"]
        image = cv2.imread(input_path)
        if image is None:
            raise RuntimeError("Could not load the original captured full-frame image")
        mask_path = state["paths"].get("testbed_source_mask")
        detection_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if mask_path else None
        if detection_mask is None:
            raise RuntimeError("The full-frame testbed mask is unavailable; capture again")
        processing_roi = (state.get("rois") or {}).get("processing")
        if not processing_roi:
            raise RuntimeError("The testbed coordinate mapping is unavailable; capture again")
        detection_region = dict(processing_roi)
        if item is not None:
            bbox = item["bbox"]
            # The processing origin can be negative because the testbed view
            # includes virtual square padding. No resize is applied to it.
            detection_region = {
                "x": int(processing_roi["x"]) + int(bbox["x"]),
                "y": int(processing_roi["y"]) + int(bbox["y"]),
                "w": int(bbox["w"]),
                "h": int(bbox["h"]),
            }
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
            detection_region=detection_region,
            measurement_scale=measurement_scale,
            detection_mask=detection_mask,
        )
        suffix = f"item_{int(item['index']):02d}" if item else "beads"
        path = self.repository.session_dir(state["id"]) / "results" / "beads" / f"{suffix}.png"
        onnx_path = self.repository.session_dir(state["id"]) / "results" / "beads" / f"{suffix}_onnx.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        annotated = identify_result_image(annotated, jewel_reference(item))
        onnx_annotated = identify_result_image(onnx_annotated, jewel_reference(item))
        cv2.imwrite(str(path), annotated)
        cv2.imwrite(str(onnx_path), onnx_annotated)
        result["image"] = str(path)
        result["onnx_image"] = str(onnx_path)
        result["input_source"] = "full_frame"
        result["coordinate_space"] = "full_frame"
        result["region_filter"] = "box_center_in_testbed_mask_and_jewel_region"
        return result

    def _run_stones(self, state: dict[str, Any], item: dict[str, Any] | None = None) -> dict[str, Any]:
        label = (item or {}).get("confirmed_label") or (state.get("classification") or {}).get("confirmed_label")
        if label_key(label) in {"finger ring", "ring"}:
            return {"applicable": False, "status": "skipped", "reason": "Analysis is disabled for finger rings."}
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
        risk = report.get("stone_surface_risk") or {}
        measurements = report.get("stone_measurements") or {}
        area = measurements.get("total_area_mm2")
        stones_found = str(risk.get("level") or "NONE") != "NONE"
        area_text = f"{float(area):.2f} mm\u00b2" if stones_found and area is not None else "Not reported"
        report["reflection_notice"] = None
        gallery_weights = None
        if estimate_weight:
            # Reuse the overall result/UI calculation, including caps and ranges.
            gallery_weights = weight_summary({
                "weight_g": state.get("weight_g"),
                "result": {"count": 1, "items": [{
                    "stones": {"found": stones_found, **stone_weight_fields(measurements, stones_found)},
                    "beads": ((state.get("result") or {}).get("beads") or {}),
                }]},
            })
        stone_metric = ("STONE SURFACE AREA", area_text)
        if estimate_weight:
            stone_g = gallery_weights.get("stone_g")
            stone_metric = ("STONE WEIGHT (ESTIMATED)", f"{stone_g:.2f} g" if stone_g is not None else "Unavailable")
        gallery = stone_detection.build_result_gallery(analysis["jewel_views"], summary_fields=[
            ("RESULT", str(risk.get("status") or "No stones")),
            stone_metric,
        ], item_reference=jewel_reference(item), weights=gallery_weights)
        cv2.imwrite(str(gallery_path), gallery)
        report_path = self._save_json(
            output_dir / "result.json",
            report if estimate_weight else without_stone_weight_estimates(report),
        )
        result = {
            # Risk NONE includes calibrated sub-5 mm² regions suppressed as
            # likely reflections/texture rather than reported as real stones.
            "reflection_flagged": False,
            "reflection_notice": None,
            "found": stones_found,
            "risk_level": str(risk.get("level") or "NONE"),
            "risk_status": str(risk.get("status") or "No stones"),
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
            # Also enforce the current policy for queued captures with old routes.
            for item in classified_items:
                if label_key(item.get("confirmed_label")) in {"finger ring", "ring", "bangle"}:
                    item["route"] = self._route_for_item(item, item["confirmed_label"])
            for item in classified_items:
                self._check_bangle_placement(item, item.get("confirmed_label") or "")
            if any(not (item.get("placement_check") or {}).get("passed", True) for item in classified_items):
                self._reject_bangle_placement(state, classified_items)
                state["job"] = {"status": "failed", "stage": "Placement check",
                                "message": PLACEMENT_MESSAGE, "percent": 0,
                                "seconds_remaining": 0, "error": PLACEMENT_MESSAGE}
                self.repository.save(state)
                return
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
                    "route": dict(item["route"]),
                    "bangle_orientation": item.get("bangle_orientation"),
                    "errors": [],
                }
                route = item_result["route"]
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
            # Persist the current routing policy for reports and reopened results.
            for item, item_result in zip((state.get("classification") or {}).get("items") or [], result_items):
                item["route"] = item_result["route"]
                if item_result.get("bangle_orientation") is not None:
                    item["bangle_orientation"] = item_result["bangle_orientation"]
            state.setdefault("route", {}).update({
                "key": "multi_item" if len(result_items) > 1 else (
                    result_items[0]["route"]["key"] if result_items else "no_analysis"
                ),
                "dimension": any(item["route"].get("dimension") for item in result_items),
                "beads": any(item["route"].get("beads") for item in result_items),
                "stones": any(item["route"].get("stones") for item in result_items),
                "items": [item["route"] for item in result_items],
            })
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
