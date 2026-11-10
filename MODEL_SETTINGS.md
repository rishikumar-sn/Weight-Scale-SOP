# Capture model settings

Open Settings > Models, below Stone analysis. All checks default to enabled.
Toggle individual checks, then select Apply models. Settings persist across restarts
and apply immediately to the next capture on every scale, without reloading models
or restarting the backend. Unsaved edits do not change the workflow.

Each capture retains its settings snapshot through classification, confirmation and
analysis. Recapture after applying to use new settings on an already captured item.
Previous rejections and completed results are retained.

Available switches:
- Anomaly / foreign-object detection (including its overlapping-item placement check).
- Foreground verification against anomaly-model jewellery masks.
- Tray-rim filtering with semantic verification.
- Thread-only / empty-testbed classification.
- REAL/PRINT classification.
- Raw-crop non-jewellery semantic validation before jewellery type classification.
- Bangle opening placement geometry, including confirmation and queued analysis.
- Thread segmentation used to exclude thread pixels from stone analysis.

Disabling the anomaly model automatically bypasses foreground verification and rim
filtering, which depend on its masks. Their individual preferences are retained.
Skipped gates record passed=true and skipped=true rather than invented predictions.
Disabled anomaly, thread-only and print models are also omitted from startup preload.

Required testbed segmentation, AprilTag calibration and jewellery type classification
remain active. Bead, stone and dimension analysis continue according to item routing.
The semantic switch bypasses the separate raw-crop eliminator; the type classifier
can still return its own non-jewellery prediction.

API: PUT /api/settings/models with boolean fields foreign_objects,
component_presence, rim_filter, thread_only, real_print, semantic_validation,
bangle_placement and thread_segmentation. GET /api/settings includes the saved map.
Unknown model keys are rejected; omitted fields default to true.

Deployment: after installing this code update, restart an already running backend
once and refresh the browser. Subsequent Apply models changes need no restart.
Live camera/scale demo validation is still required on the target hardware.
