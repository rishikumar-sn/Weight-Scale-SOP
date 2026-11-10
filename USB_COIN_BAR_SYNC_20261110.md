# Coin/bar model USB update — 10 November 2026

Destination: `E:/weightScaleinstall/WEIGHT-INTEGRATION-APP`.

Copied 26 runtime, model, frontend, documentation and regression-test files.
The active model is `models/segmentation/anomalies.onnx`, SHA-256
`3384e6cac7ac003d9a86e316a7bcc764a5815b538f2aca4c3017d90f2746b0e3`.
The old `anomolies.onnx` was removed from the deployed application and archived
under the update evidence directory, outside the application.

Coin/bar detections bypass SigLIP, retain segmentation outlines, use J-number/type
tags and carry gold-verification warnings in review and final results. Ordinary
jewellery and blocking anomaly/REAL-PRINT checks retain their existing flow.

Every copied file matched the workspace by SHA-256. Frontend HTML references
resolve to the copied production assets. Settings, databases and correction
galleries matched their pre-copy hashes; captures and environments were not
copied or changed.

Tests ran with explicitly verified imports from the USB runtime and loaded the
USB model: **136 passed, 1 skipped**. The skip requires the research checkpoint,
which is intentionally excluded from the portable installation. One pytest
import-rewrite warning was emitted.

Evidence, original replaced files and the retired model:
`E:/weightScaleinstall/testing/coin-bar-update-2026-11-10T124306_0530/`.

Bundle summary and integrity-manifest entries were refreshed. Restart the app
and refresh the browser on the destination computer to activate the update.
No package installation, backend restart or live hardware operation was performed.
