# Crop-trained jewel/thread classifier v2 - 2 November 2026

Active runtime model: models/classification/jewel-thread-small.onnx.
Source: jewel_thread_cls_4class_yolos_960-v2.pt in Downloads. Export: CPU float32,
opset 17, static 1 x 3 x 960 x 960 input, four probability outputs. Source and ONNX
SHA-256 values are in the adjacent .onnx.infer.json. Export-only dependencies are
isolated under experiments/thread_model_export/.venv; app requirements are unchanged.

## Input and application order

The supplied training dataset data/datasets/jewel-thread-crops-20261031 contains
native raw per-object crops. The application supplies the same raw_crop_bgr
produced by Otsu object separation, for EACH object in single and multiple captures.
It does not pass the full camera frame, a colour-normalized stone image, or the
white-isolated SigLIP crop to this model. No new square padding is introduced.
The checkpoint transforms and application preprocessing match exactly: BGR to RGB,
PIL bilinear shortest-edge resize to 960, center crop, CHW float32 / 255.

Processing order:

1. Separate objects with the existing Otsu workflow.
2. Existing anomaly and placement checks; failures stop both classifiers.
3. Save each native raw item crop and run the v2 thread classifier on EVERY item.
4. Block the capture if any item is Thread-Only or Empty-Test-Bed.
5. Require the existing foreground/jewellery-instance verification.
6. Run SigLIP jewellery-type prediction and existing SOP routing.

This supersedes the temporary chain-only restriction. Rings, earrings, bangles,
Mattal, chains and other separated objects all receive independent thread checks.
The Thread-Only check runs before foreground verification so a standalone tassel
can be identified without first requiring it to be a jewellery instance. A positive
thread-model result does not bypass the foreground checks. API overrides and queued
jobs cannot bypass a persisted failed gate. Thread segmentation remains unchanged.

Classes in checkpoint/output order:

- 0 Empty-Test-Bed -> empty_testbed: reject and recapture.
- 1 Jewel_with_Thread -> jewel_with_thread: allow.
- 2 Jewel_without_Thread -> jewel_without_thread: allow.
- 3 Thread-Only -> thread_only: reject and recapture.

## Verification

59 selected saved dataset crops were compared against PyTorch. Preprocessing was
identical and maximum probability error was 8.344650268554688e-7. All 59 produced
the expected allow/block decision; 57/59 matched the supplied class labels. The two
disagreements were automatically labelled with-thread examples predicted without
thread; both are still allowed. Both reviewed thread-only crops, all seven reviewed
rings and all 12 sampled earrings passed their expected checks. The 12 manually
reviewed seed crops matched all supplied classes.

These examples include training data and automatically suggested labels. This is
conversion/regression evidence, not an independent accuracy benchmark.

Real saved-session replay used actual object separation, anomaly detection, the
new ONNX model and the real SigLIP classifier. The large thread-bearing necklace
20261021_183130_f1d428d0 passed as jewel_with_thread and Chain / Necklace. The six-item
20261023_115548_2f813fbb capture passed every crop, with jewellery predictions for
rings, earrings and a bangle. An older earring capture was blocked because its saved
full-frame testbed mask is missing; no verification guard was bypassed. Two reviewed
thread-only crops were separately replayed with the real ONNX model and fixture
crop/anomaly evidence because their original full camera frames are unavailable;
both were rejected before SigLIP.

93 focused tests passed, covering every class/category, single and multiple crops,
anomaly/placement ordering, unchanged raw crop pixels, foreground verification,
recapture guards, stone settings/output and dataset-builder regressions. No live
camera or physical scale test was performed. Restart the app to load the new session.

Re-export using the development environment and original checkpoint:

```powershell
experiments/thread_model_export/.venv/Scripts/python.exe scripts/export_jewel_thread.py --source "C:\Users\karth\Downloads\jewel_thread_cls_4class_yolos_960-v2.pt" --output models/classification/jewel-thread-small.onnx
```

Detailed artifacts and previous local model/source backups are under
experiments/thread_model_v2_20261102. USB backups and verification are under testing.
