# Updated jewel/thread classification model - 31 October 2026

The checkpoint jewel-thread-classify-new.pt was converted to the existing runtime
path models/classification/jewel-thread-small.onnx (CPU, float32, opset 17,
1 x 3 x 960 x 960 input). Its SHA-256 and export details are recorded in the
adjacent .onnx.infer.json file. The supplied checkpoint remains in Downloads.
No export/training dependencies were added to the application environment.

This checkpoint has FOUR classes in this order:

0. Empty-Test-Bed -> empty_testbed
1. Jewel_with_Thread -> jewel_with_thread
2. Jewel_without_Thread -> jewel_without_thread
3. Thread-Only -> thread_only

The runtime reads and normalizes class names from ONNX metadata rather than
assuming the previous three-class indices. Both jewel classes pass the gate;
empty_testbed and thread_only require recapture. Empty testbed predictions have
their own rejection message. API overrides and queued jobs cannot bypass a failed
gate. Legacy three-class ONNX exports remain supported. This classification model
is separate from the existing thread segmentation model, which remains unchanged.

## Application scope

Jewellery type prediction now runs BEFORE this provisional model. Only Chain /
Necklace, Mattal and explicit chain-type aliases use the thread classifier.
Earrings / Nosepin, Finger Ring, Bangle, generic Bracelet and Other Gold Jewellery
skip it. Historical chain/necklace, haram, mangalsutra, dollar-chain and kasu-mala
labels remain supported. A generic Bracelet prediction is not assumed to be a
chain bracelet, because the category also includes cuffs.

Each separated item is routed independently in both single and mixed captures.
Skipped items store applicable=false and status=skipped; they cannot be rejected
by this thread model. A failed eligible item still requires recapture of the
capture. Anomaly and foreground checks continue to apply to every item. The
seven ring errors in the raw-model audit below are no longer used as application
thread-gate decisions for items predicted Finger Ring.

Preprocessing matches the checkpoint transforms: RGB PIL bilinear shortest-edge
resize to 960, center crop, channels first, float32 / 255. The original per-item
raw-crop inference workflow is preserved.

## Validation and limitations

On the existing 12 labelled USB review crops, preprocessing matched exactly and
ONNX probabilities matched PyTorch within 2.384185791015625e-7 absolute error.
Both thread-only crops were correctly blocked. However, all seven labelled gold
rings and one other jewellery crop were also labelled thread_only. The two
jewellery-with-thread crops were labelled jewel_without_thread (still allowed).
That is 2/12 exact-class matches and 4/12 correct allow/block decisions on this
curated failure-heavy set; these are NOT population accuracy measurements.
Synthetic uniform black/grey/white images received jewel labels, so the extra
empty-testbed output does not guarantee empty-image rejection. Existing anomaly,
foreground and placement checks still run before this gate.

Regression tests cover class order/name mapping for both model versions, all four
allow/block decisions, mixed-item rejection, recapture guards and queued-job
blocking. The frontend production build passes. No live hardware test was run.
Restart the application after replacing files to load the new model session.

## Re-export and check

Use a development environment with PyTorch, matching Ultralytics 8.4.193, ONNX,
ONNX Runtime and the application's dependencies:

```powershell
python scripts/export_jewel_thread.py --source "C:\Users\karth\Downloads\jewel-thread-classify-new.pt" --output models/classification/jewel-thread-small.onnx
python scripts/verify_thread_model_update.py --source "C:\Users\karth\Downloads\jewel-thread-classify-new.pt" --onnx models/classification/jewel-thread-small.onnx --samples-dir "E:\weightScaleinstall\testing\mixed-thread-20261030\review_samples" --output experiments/thread_model_export/verification.json
```

Backups and detailed verification are under experiments/thread_model_export
locally, and testing/thread-classifier-update-20261031 on the USB deployment.
