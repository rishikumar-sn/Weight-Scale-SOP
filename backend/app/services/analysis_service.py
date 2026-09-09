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

from ..analysis.vision import separate_jewellery_items
from ..core.config import PROJECT_ROOT
from ..domain.workflow import route_for_label
from ..domain.weights import stone_weight_fields, weight_summary


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
# Keep the detector cutoff aligned with the production Hailo pipeline.  The
# exported ONNX model emits many valid beads between 0.50 and 0.75, so the old
# 0.75 cutoff incorrectly turned clear bead jewellery into "not detected".
BEAD_DETECTION_SCORE_THRESHOLD = 0.50
BEAD_FALSE_POSITIVE_THRESHOLD = 0.75
# Temporary detector-only test mode requested during live bead validation.
# Keep the verifier model and telemetry code available so it can be restored
# after the detector results have been reviewed.
BEAD_FALSE_POSITIVE_VERIFICATION_ENABLED = False
BEAD_MIN_VERIFIED_COUNT = 2
# Historical replay with the current detector found that genuine pearl work can
# pass the crop verifier at 22.5%, while plain rope-chain false positives pass
# at 0%. Keep a little margin below the observed genuine case.
BEAD_MIN_VERIFICATION_RATIO = 0.20
BEAD_RED_FALLBACK_MIN_CANDIDATES = 2
BEAD_RED_FALLBACK_MIN_COMPONENTS = 8
BEAD_RED_FALLBACK_MAX_AREA_CV = 0.45


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


def _bead_evidence_image(
    image_bgr: np.ndarray,
    region: dict[str, int] | None,
) -> tuple[np.ndarray, int, int]:
    """Return the item region used by the colour fallback and its full-ROI offset."""
    if not region:
        return image_bgr, 0, 0
    image_h, image_w = image_bgr.shape[:2]
    x1 = max(0, min(image_w, int(region["x"])))
    y1 = max(0, min(image_h, int(region["y"])))
    x2 = max(x1, min(image_w, x1 + int(region["w"])))
    y2 = max(y1, min(image_h, y1 + int(region["h"])))
    return image_bgr[y1:y2, x1:x2], x1, y1


def _bead_classifier_crop(
    image_bgr: np.ndarray,
    bbox: list[int],
    padding_ratio: float = 0.05,
) -> np.ndarray:
    """Match the production false-positive classifier crop preparation."""
    x1, y1, x2, y2 = (int(value) for value in bbox)
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    crop_width = max(1, int(math.ceil(width * (1.0 + 2.0 * padding_ratio))))
    crop_height = max(1, int(math.ceil(height * (1.0 + 2.0 * padding_ratio))))
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    crop_x1 = int(math.floor(center_x - crop_width / 2.0))
    crop_y1 = int(math.floor(center_y - crop_height / 2.0))
    crop_x2 = crop_x1 + crop_width
    crop_y2 = crop_y1 + crop_height
    image_h, image_w = image_bgr.shape[:2]
    source_x1, source_y1 = max(0, crop_x1), max(0, crop_y1)
    source_x2, source_y2 = min(image_w, crop_x2), min(image_h, crop_y2)
    rectangular = np.full((crop_height, crop_width, 3), 114, dtype=np.uint8)
    if source_x2 > source_x1 and source_y2 > source_y1:
        target_x1 = source_x1 - crop_x1
        target_y1 = source_y1 - crop_y1
        rectangular[
            target_y1 : target_y1 + source_y2 - source_y1,
            target_x1 : target_x1 + source_x2 - source_x1,
        ] = image_bgr[source_y1:source_y2, source_x1:source_x2]
    side = max(crop_width, crop_height)
    crop = np.full((side, side, 3), 114, dtype=np.uint8)
    left = (side - crop_width) // 2
    top = (side - crop_height) // 2
    crop[top : top + crop_height, left : left + crop_width] = rectangular
    return crop


def _repeated_red_bead_evidence(image_bgr: np.ndarray) -> dict[str, Any]:
    """Measure repeated, similarly sized dark-red bead bodies in the image.

    This narrowly recovers rudraksha/coral-style beads that are outside the
    verifier's training domain.  Repetition and area consistency prevent an
    ornate setting with a mixture of red stone sizes from becoming a fallback.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    red = (
        (saturation >= 70)
        & (value >= 35)
        & (value <= 210)
        & ((hue <= 10) | (hue >= 165))
    ).astype(np.uint8) * 255
    red = cv2.morphologyEx(
        red,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(red, 8)
    minimum_area = max(18, int(red.size * 0.00004))
    maximum_area = max(minimum_area + 1, int(red.size * 0.035))
    components: list[dict[str, Any]] = []
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if not minimum_area <= area <= maximum_area or min(width, height) <= 2:
            continue
        component = np.where(labels == label, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        perimeter = float(cv2.arcLength(contour, True))
        circularity = (
            4.0 * math.pi * float(cv2.contourArea(contour)) / (perimeter * perimeter)
            if perimeter > 0
            else 0.0
        )
        aspect_ratio = max(width, height) / float(max(1, min(width, height)))
        if circularity >= 0.35 and aspect_ratio <= 2.2:
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            components.append(
                {
                    "bbox": [x, y, x + width, y + height],
                    "area_px": area,
                    "circularity": round(circularity, 3),
                }
            )
    areas = [int(component["area_px"]) for component in components]
    area_array = np.asarray(areas, dtype=np.float32)
    area_cv = (
        float(np.std(area_array) / max(1.0, float(np.mean(area_array))))
        if area_array.size
        else 0.0
    )
    supported = (
        len(components) >= BEAD_RED_FALLBACK_MIN_COMPONENTS
        and area_cv <= BEAD_RED_FALLBACK_MAX_AREA_CV
    )
    return {
        "supported": supported,
        "component_count": len(components),
        "median_area_px": round(float(np.median(area_array)), 1) if area_array.size else 0.0,
        "area_coefficient_of_variation": round(area_cv, 3),
        "minimum_component_count": BEAD_RED_FALLBACK_MIN_COMPONENTS,
        "maximum_area_coefficient_of_variation": BEAD_RED_FALLBACK_MAX_AREA_CV,
        "components": components,
    }


def _bead_presence_decision(
    candidate_count: int,
    verified_count: int,
    repeated_red_evidence: dict[str, Any],
) -> tuple[bool, str, float]:
    acceptance_ratio = verified_count / float(max(1, candidate_count))
    verified_consensus = (
        verified_count >= BEAD_MIN_VERIFIED_COUNT
        and acceptance_ratio >= BEAD_MIN_VERIFICATION_RATIO
    )
    red_fallback = (
        candidate_count >= BEAD_RED_FALLBACK_MIN_CANDIDATES
        and bool(repeated_red_evidence.get("supported"))
    )
    if verified_consensus:
        return True, "verified_consensus", acceptance_ratio
    if red_fallback:
        return True, "repeated_red_bead_fallback", acceptance_ratio
    return False, "insufficient_verified_bead_evidence", acceptance_ratio


def _bead_detector_only_decision(candidate_count: int) -> tuple[bool, str]:
    detected = candidate_count >= BEAD_MIN_VERIFIED_COUNT
    return (
        detected,
        "detector_consensus" if detected else "insufficient_detector_evidence",
    )


class BeadFalsePositiveFilter:
    def __init__(
        self,
        model_path: Path,
        threshold: float = BEAD_FALSE_POSITIVE_THRESHOLD,
    ) -> None:
        self.model_path = model_path
        self.threshold = float(threshold)
        self.model: Any = None
        self.torch: Any = None
        self.transform: Any = None
        self.image_type: Any = None
        self.lock = threading.Lock()

    def _load(self) -> None:
        if self.model is not None:
            return
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Bead verification model not found: {self.model_path}")
        import torch
        from PIL import Image
        from torch import nn
        from torchvision import models, transforms

        model = models.mobilenet_v3_small(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
        model.load_state_dict(torch.load(self.model_path, map_location="cpu", weights_only=True))
        model.eval()
        self.model = model
        self.torch = torch
        self.image_type = Image
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def accept(self, crops_bgr: list[np.ndarray]) -> list[dict[str, Any]]:
        if not crops_bgr:
            return []
        with self.lock:
            self._load()
            tensors = []
            for crop_bgr in crops_bgr:
                rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                tensors.append(self.transform(self.image_type.fromarray(rgb)))
            inputs = self.torch.stack(tensors)
            with self.torch.inference_mode():
                probabilities = self.torch.softmax(self.model(inputs), dim=1).cpu().numpy()
        return [
            {
                "accepted": float(probability[1]) >= self.threshold,
                "true_detection_probability": float(probability[1]),
            }
            for probability in probabilities
        ]


class OnnxBeadDetector:
    def __init__(
        self,
        model_path: Path,
        verifier_path: Path,
        providers: list[str] | None = None,
    ) -> None:
        providers = providers or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.model_name = model_path.name
        self.verifier = BeadFalsePositiveFilter(verifier_path)

    def run(
        self,
        image: np.ndarray,
        threshold: float = BEAD_DETECTION_SCORE_THRESHOLD,
        detection_region: dict[str, int] | None = None,
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
        height, width = image.shape[:2]
        scale = min(640.0 / width, 640.0 / height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        left = (640 - resized_width) // 2
        top = (640 - resized_height) // 2
        canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
        canvas[top : top + resized_height, left : left + resized_width] = resized
        tensor = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.0
        output = np.asarray(self.session.run(None, {self.input_name: tensor[None]})[0])
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
        checks = (
            self.verifier.accept([
                _bead_classifier_crop(image, candidate["bbox"])
                for candidate in candidates
            ])
            if BEAD_FALSE_POSITIVE_VERIFICATION_ENABLED
            else [
                {"accepted": True, "true_detection_probability": None}
                for _ in candidates
            ]
        )
        checked_detections = [
            {
                **candidate,
                "verification_accepted": (
                    check["accepted"]
                    if BEAD_FALSE_POSITIVE_VERIFICATION_ENABLED
                    else None
                ),
                "true_detection_probability": check["true_detection_probability"],
                "accepted": check["accepted"],
                "acceptance_source": (
                    "verifier"
                    if BEAD_FALSE_POSITIVE_VERIFICATION_ENABLED
                    else "detector_only"
                ),
            }
            for candidate, check in zip(candidates, checks)
        ]
        verified_detections = [
            detection
            for detection in checked_detections
            if detection["verification_accepted"]
        ]
        evidence_image, evidence_offset_x, evidence_offset_y = _bead_evidence_image(
            image,
            detection_region,
        )
        repeated_red_evidence = _repeated_red_bead_evidence(evidence_image)
        if evidence_offset_x or evidence_offset_y:
            for component in repeated_red_evidence["components"]:
                x1, y1, x2, y2 = component["bbox"]
                component["bbox"] = [
                    x1 + evidence_offset_x,
                    y1 + evidence_offset_y,
                    x2 + evidence_offset_x,
                    y2 + evidence_offset_y,
                ]
        if BEAD_FALSE_POSITIVE_VERIFICATION_ENABLED:
            beads_detected, decision_source, verification_acceptance_ratio = (
                _bead_presence_decision(
                    len(candidates),
                    len(verified_detections),
                    repeated_red_evidence,
                )
            )
        else:
            beads_detected, decision_source = _bead_detector_only_decision(len(candidates))
            verification_acceptance_ratio = 0.0
        red_fallback_used = decision_source == "repeated_red_bead_fallback"
        detections = (
            [
                {
                    **component,
                    "score": None,
                    "verification_accepted": False,
                    "true_detection_probability": None,
                    "accepted": True,
                    "acceptance_source": "repeated_red_bead_fallback",
                }
                for component in repeated_red_evidence["components"]
            ]
            if red_fallback_used
            else (
                (
                    verified_detections
                    if BEAD_FALSE_POSITIVE_VERIFICATION_ENABLED
                    else checked_detections
                )
                if beads_detected
                else []
            )
        )
        annotated = image.copy()
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (24, 170, 90), 3, cv2.LINE_AA)
        bead_status = "Beads detected" if beads_detected else "Beads not detected"
        header_width = min(width - 10, 330)
        cv2.rectangle(annotated, (10, 10), (header_width, 56), (18, 24, 32), -1)
        cv2.putText(
            annotated,
            bead_status,
            (20, 43),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
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
            "verification_enabled": BEAD_FALSE_POSITIVE_VERIFICATION_ENABLED,
            "verified_count": (
                len(verified_detections)
                if BEAD_FALSE_POSITIVE_VERIFICATION_ENABLED
                else None
            ),
            "verification_acceptance_ratio": (
                round(verification_acceptance_ratio, 4)
                if BEAD_FALSE_POSITIVE_VERIFICATION_ENABLED
                else None
            ),
            "verification_rejected_count": (
                len(candidates) - len(verified_detections)
                if BEAD_FALSE_POSITIVE_VERIFICATION_ENABLED
                else None
            ),
            # Backward-compatible alias. This is classifier telemetry rather
            # than labelled detector ground truth.
            "false_positive_count": (
                len(candidates) - len(verified_detections)
                if BEAD_FALSE_POSITIVE_VERIFICATION_ENABLED
                else None
            ),
            "detections": detections,
            "count": len(detections),
            "verification_fallback_used": red_fallback_used,
            "decision_source": decision_source,
            "decision_reason": (
                f"Detector-only test mode accepted {len(candidates)} candidates."
                if decision_source == "detector_consensus"
                else f"Detector-only test mode found only {len(candidates)} candidate; at least {BEAD_MIN_VERIFIED_COUNT} are required."
                if decision_source == "insufficient_detector_evidence"
                else f"{len(verified_detections)} of {len(candidates)} candidates passed verification."
                if decision_source == "verified_consensus"
                else (
                    "Repeated similarly sized dark-red bead bodies supported the detector candidates."
                    if red_fallback_used
                    else f"Only {len(verified_detections)} of {len(candidates)} candidates passed verification; bead evidence was insufficient."
                )
            ),
            "repeated_red_bead_evidence": repeated_red_evidence,
            "model": self.model_name,
            "score_threshold": float(threshold),
            "verification_model": self.verifier.model_path.name,
            "verification_threshold": self.verifier.threshold,
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
        bead_detector = self._get_beads()
        if BEAD_FALSE_POSITIVE_VERIFICATION_ENABLED:
            bead_detector.verifier._load()

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
                    PROJECT_ROOT / "models" / "detection" / "beadcheck_mobilenet_v3.pt",
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
            crop_rgb = cv2.cvtColor(separated_item["crop_bgr"], cv2.COLOR_BGR2RGB)
            prediction = classifier.classify_image(Image.fromarray(crop_rgb), image_path=str(crop_path))
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
                "predicted_label": prediction.label,
                "confirmed_label": None,
                "confirmed": False,
                "confidence": float(prediction.confidence),
                "gallery_match": bool(prediction.gallery_match),
                "gallery_similarity": float(prediction.gallery_similarity),
                "gallery_support": int(prediction.gallery_support),
                "gallery_margin": float(prediction.gallery_margin),
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
        # The detector was trained on the complete configured jewellery ROI at
        # 640x640. A tight per-item crop changes bead scale and context and can
        # also make the crop verifier reject otherwise valid detections. Run on
        # that trained domain, then spatially assign detections to this item so
        # multi-jewellery captures remain independent.
        input_path = state["paths"]["working"]
        image = cv2.imread(input_path)
        if image is None:
            raise RuntimeError("Could not load the captured jewellery image")
        result, annotated, onnx_annotated = self._get_beads().run(
            image,
            threshold=BEAD_DETECTION_SCORE_THRESHOLD,
            detection_region=(item or {}).get("bbox"),
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
        suffix = f"item_{int(item['index']):02d}" if item else "capture"
        output_dir = self.repository.session_dir(state["id"]) / "results" / "stones" / suffix
        output_dir.mkdir(parents=True, exist_ok=True)
        gallery_path = output_dir / "stones.png"
        cv2.imwrite(str(gallery_path), analysis["result_gallery_bgr"])
        report_path = self._save_json(output_dir / "result.json", report)
        risk = report.get("stone_surface_risk") or {}
        measurements = report.get("stone_measurements") or {}
        stones_found = str(risk.get("level") or "NONE") != "NONE"
        return {
            # Risk NONE includes calibrated sub-5 mm² regions suppressed as
            # likely reflections/texture rather than reported as real stones.
            "found": stones_found,
            "risk_level": str(risk.get("level") or "NONE"),
            "risk_status": str(risk.get("status") or "NO RISK - NO STONES DETECTED"),
            "stone_area_mm2": measurements.get("total_area_mm2") if stones_found else None,
            **stone_weight_fields(measurements, stones_found),
            "measurement_basis": risk.get("basis"),
            "image": str(gallery_path),
            "report": report_path,
        }

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
