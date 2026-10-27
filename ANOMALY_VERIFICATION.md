# Anomaly detection verification — 24 October 2026

Status: PASSED on saved Linux captures, using real CPU ONNX inference.

## Cause and fix

`anomolies.onnx` was already integrated for ESSAE JX500 and EMBSYS WM and
already received the full raw `capture/original.png`. The plastic-cover capture
`20261023_163520_61a64631` had two Not Gold items alongside jewellery. The old
condition skipped the second gate whenever any separated item was Not Gold.

The existing jewellery rejection runs first. Captures containing jewellery then
run the anomaly gate even when another item is Not Gold; those decisions remain
authoritative. An anomaly detection rejects the whole capture and blocks analysis;
the existing rejected screen offers Recapture. A capture rejected entirely by the
first jewellery gate stops before anomaly inference.

The anomaly model is preloaded during backend startup, and both Linux backend
launchers validate the anomaly and scale-specific testbed assets. The saved check
now records the raw input dimensions and testbed pixel count for diagnosis.

## Raw image and circular hole

The full raw camera frame enters the anomaly model, with its normal resize and
letterboxing. Predicted object masks are intersected with the full-frame testbed
mask afterward. ESSAE's separate `circle` label already removes the hole and rim
from that mask. This working exclusion is preserved. Review-image cropping is
only display formatting, unchanged, and never supplies anomaly-model input.

## Checks

- Replayed all six non-tare captures through fresh JX500 testbed segmentation and
  the anomaly model. All fresh masks matched their saved masks exactly.
- Five earlier captures remained clear; the hole caused no false rejection.
- Plastic-cover capture: `plastic_packet` confidence 0.599579, above threshold
  0.35; correctly rejected using the raw frame.
- Actual classifier and service replay: mixed jewellery/Not Gold/plastic capture
  rejected; attempting to confirm it raised an error before analysis could start.
- Actual classifier and service replay: clean finger-ring capture
  `20261023_064836_43d263c8` reached awaiting confirmation.
- Forced first-gate Not Gold rejection stopped before the anomaly gate was called.
- Both Linux launcher preflights accepted deployed assets and rejected a missing
  anomaly model in isolated fixtures. Shell and Python syntax checks passed.
- No external network connections were attempted. Replays used a temporary
  database and session copies; original captures and history were not modified.

Detailed evidence: `data/anomaly-replay.json`.
Repeat command: see `scripts/verify_anomaly_sessions.py` and `README_OFFLINE.md`.

Limits: no live camera or USB scale test was performed. Saved captures are all
JX500; EMBSYS uses the same anomaly gate, but no EMBSYS capture was available for
image-level replay. Model weights and detection thresholds were not changed.
Restart the backend to load the updated code.
