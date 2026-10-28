# Necklace and chain thread exclusion

Necklace/chain analysis uses `models/segmentation/thread.onnx` (YOLO11s-seg,
Thread class, 960 input). The model receives the complete original captured
image, letterboxed to 960×960. It never receives a necklace bounding-box crop.
CPU ONNX Runtime is used with confidence 0.25, mask threshold 0.50, and NMS IoU
0.45. All retained thread instance masks are combined.

Decoded masks are mapped from original-camera coordinates into each jewel using
the testbed processing origin and item bounding box, including square padding.
Only thread pixels intersecting that item's foreground are excluded. Exclusion
is applied to stone candidate generation and refinement; no rectangular region
is removed. The display mask and image keep the thread visible, without green
stone overlays on its segmented pixels. Stone areas and estimates exclude thread.

Only Chain / Necklace and its historical neck-jewellery aliases use this model.
Other jewellery categories retain their existing behavior. Results show “Thread
is present” or “Thread is not present” in the UI, stone image, and PDF. JSON results
store this under `stones.thread`, including thresholds and excluded pixel count.
A model/inference error produces an analysis error, never “Thread is not present”.
Existing completed captures are unchanged; new captures use this behavior after
restarting the application. No additional Python package or online download is
required. Both desktop and USB bundles contain the model and application changes.

Run `scripts/verify_thread_segmentation.py` with the bundled environment to replay
saved captures and verify empty and multiple detections, coordinate mapping,
category gating, pixel preservation, and partial/full thread exclusion. Evidence
is written under `testing/thread-segmentation`, without modifying saved sessions.
The three saved necklace captures used for replay yielded no thread detections;
positive exclusion and neighboring-stone preservation are verified synthetically.
