# REAL/PRINT integration — 2 November 2026

The reviewed enriched MobileNetV3-Large classifier is now part of the main
jewellery capture workflow. The standalone research lab remains separate.

## Operator behavior

Capture as usual. After the existing anomaly/placement, thread/empty and
foreground-support gates pass, every candidate is checked for printed jewellery.
Any PRINT result rejects the **whole capture**, including mixed REAL/PRINT
scenes. The rejected screen offers **Recapture only**. No confirmation,
correction-learning, category override, queued analysis, result image or report
can restore an already rejected print capture. The operator must remove the
printout and create a new capture. On ACCET, use its normal scale-button
recapture workflow.

An uncertain prediction or failed verification also blocks the capture with a
recapture message. Model loading is included in the existing startup preload.
Missing model files, an incorrect checksum, malformed configuration/output or
unavailable original pixels never fall back to accepting jewellery.

## Implementation

- `backend/app/analysis/real_print.py`: CPU ONNX inference, checksum/config
  validation, original-pixel dual crops and exact training preprocessing.
- `AnalysisService`: one cached model, per-item full-resolution source mapping,
  evidence crop archival and whole-capture decision before SigLIP.
- `real_print_check`: aggregate and per-item labels, probabilities, fusion,
  latency, original-coordinate boxes, archived input paths and model provenance
  persisted with newly processed captures. Existing sessions are not migrated
  or replayed automatically.
- Common blocking logic protects confirmation, correction learning, queued
  analysis, public history/media and report/result access. A PRINT/review
  rejection is latched to its capture; calling classification again does not
  bypass it.
- Rejected evidence uses `P1: PRINT` / review region annotations and suppresses
  jewellery-type J labels. React displays the corresponding recapture heading
  and its existing single recapture action.

The production separator's extra proposal padding is removed using each
component mask's bounds. The resulting item box is mapped through the virtual
processing ROI into the original image; tight/context padding is then applied
and clipped at the image boundaries. Both archived inputs retain natural
background, texture and colour. Tare/packet captures are unchanged.

## Deployed model

`models/classification/real-print-mobilenetv3.onnx` and its `.json` sidecar are
self-contained production assets, included by the existing offline model-tree
packaging. The sidecar records model/checkpoint SHA-256, class order
`[REAL, PRINT]`, fixed two-view batch, 224-pixel RGB PIL-bilinear letterbox
with fill 240, ImageNet normalization, crop paddings, thresholds and dataset
version. The offline verification script now checks this asset as well.

PRINT threshold is 0.50 with an uncertainty half-band of 0.05. A score within
the band is REVIEW, not acceptance; only REAL passes. The best checkpoint's
validation-selected fusion weight is 0% tight / 100% context. Both views are
still inferred and recorded, matching the lab's per-view diagnostics.

Training used 139 REAL and 103 quality-eligible PRINT candidates, with 169
training, 37 validation and 36 held-out examples. Best epoch was 6. The initial
held-out result was 21/21 REAL and 15/15 PRINT. Source identities are inferred
from captures and visual similarity; repeated physical designs and the limited
range of printers/papers remain research limitations. Further genuine-on-paper
and independent print-source validation is needed. This gate does not measure
gold purity or certify physical material.

## Verification and activation

Regression tests cover mixed scenes, REAL pass-through, PRINT/review/failure
blocking, original-pixel crop mapping (including a negative virtual ROI),
checksum rejection, stale/queued confirmation bypass prevention, rejection
annotations, and exact ONNX/PyTorch preprocessing and probability parity over
the saved 36-sample held-out set. The production runtime never imports the lab;
the checkpoint parity test is skipped when research files are absent.

The React TypeScript/Vite production build was rebuilt. Restart the main backend
and refresh the app to load the new backend model and frontend. No new training
or package installation is required. Existing settings, captures, databases,
model galleries, datasets and previous checkpoints were not rewritten during
integration. Live end-to-end physical capture should be checked after restart;
development tests use temporary repositories and do not confirm production
sessions or learn gallery corrections.

Validation results: **94 focused tests passed**, including the original gate
regressions and new REAL/PRINT/API guards. The broader backend check reported
356 passed and 31 failed; an isolated replay of unchanged HEAD's production
modules reported the same 31 failing test names (339 passed). Those existing
failures concern unrelated legacy expectations and are not represented as a
clean full-suite pass. Prediction parity was checked on all 36 held-out lab
samples; deployed ONNX and checkpoint predictions agreed within `2e-5`.
