# Coin/bar anomaly model integration — 10 November 2026

The application now selects `models/segmentation/anomalies.onnx`, copied from
`D:/PROJECTS_BY_RISHI/models/foreign_obj_model/annomolies.onnx`.
SHA-256: `3384e6cac7ac003d9a86e316a7bcc764a5815b538f2aca4c3017d90f2746b0e3`.
The old export was moved outside the project to
`D:/PROJECTS_BY_RISHI/models/foreign_obj_model/anomolies-legacy-20261021.onnx`.
Its former project path, `models/segmentation/anomolies.onnx`, is now ignored by Git.
No backend application process was running during deployment; the next normal
application launch loads the selected export. The production frontend was rebuilt.

## Comparison

Replayed 64 saved raw captures, stratified across 28 historical label/rejection
groups, using each capture's saved full-frame testbed mask. Both models use
960×960 input. The decoder reads class IDs from ONNX metadata rather than assuming
the old class order.

| Observation | Existing four-class model | New six-class model |
| --- | ---: | ---: |
| Captures rejected by the anomaly gate | 21 | 29 |
| `foreign_obj` detections | 23 | 26 |
| `plastic_packet` detections | 5 | 16 |
| `paper` detections | 2 | 2 |
| Jewellery instances at ≥0.50 confidence | 113 | 115 |
| Median CPU detection time | 561 ms | 562 ms |

Every existing anomaly rejection remained rejected. Eight additional captures
were rejected. This includes three already rejected by the separate print model.
Visual inspection of the additional rejection `20261031_162152_2cde2ee0` showed
printed jewellery cut-outs; the old anomaly model accepted it. The new model
called those regions packets. Another historical anomaly changed from packet to
paper. Rejection consistency is stronger evidence here than exact anomaly type.
Among captures accepted by both models, the candidate never returned fewer
jewellery instances than the baseline.

These are replay observations, **not accuracy, recall or mAP measurements**.
Historical app decisions are not independent ground truth. There were no coin/bar
detections in this sample and no labelled coin/bar validation set was available;
real coin/bar precision, recall and REAL/PRINT compatibility remain unmeasured.
No live camera, weighing scale or external-device deployment was tested.

Detailed local evidence: `data/audits/anomaly_candidate_20261110/comparison.json`.
Reproduce with:

```powershell
.venv/Scripts/python.exe scripts/compare_anomaly_models.py --baseline D:/PROJECTS_BY_RISHI/models/foreign_obj_model/anomolies-legacy-20261021.onnx --candidate models/segmentation/anomalies.onnx --report data/audits/anomaly_candidate_20261110/comparison.json --per-group 3
```

## Resulting behavior

- Classes are `jewellery`, `coin`, `bar`, `foreign_obj`, `paper`, `plastic_packet`.
  Coin/bar thresholds are 0.50. Existing foreign-object thresholds are unchanged.
- Coin/bar segmentation is recognized object evidence, separate from blocking
  anomalies. It must uniquely cover at least half of one separated foreground
  component, without materially covering another component. Ambiguous, merged,
  or unmatched recognized objects require recapture.
- Coin/bar detections provide foreground presence evidence. Their decoded mask
  contours are retained in raw camera coordinates and drawn on review/result
  exports with `J1 Coin`, `J2 Bar`, etc. Existing foreground crop masks remain the
  archived item masks; the new contours are model segmentation evidence.
- Recognized coins/bars skip SigLIP semantic validation, category classification,
  correction-gallery learning, and the jewellery/thread crop classifier. Mixed
  captures still classify ordinary jewellery independently with SigLIP.
- Paper, packets, other foreign objects, placement and REAL/PRINT gates retain
  their blocking behavior. REAL/PRINT verification still runs for coins/bars.
- Coin/bar confirmation remains available. Their model-assigned type cannot be
  overwritten with a jewellery type. They have a review-only route without
  jewellery dimension, bead or stone analysis.
- Review UI and exported review images warn: “Coin [or Bar] present in the capture.
  Verify whether it is gold before proceeding; the image model identifies the
  object type and does not verify gold content or purity.” The final UI, result
  image and PDF also retain a gold-verification warning.

## Verification

- Focused regression suite: 137 passed, covering detector thresholds/masks,
  foreground ownership, rim filtering, settings, thread and REAL/PRINT gates,
  result rows, and eight new coin/bar cases. Coin/bar tests cover confirmation
  through final completion without SigLIP, ambiguous ownership, mixed captures,
  review/final tags and persistent warnings.
- `npm run build`: passed.
- Visually inspected synthetic review and final export previews in
  `data/audits/anomaly_candidate_20261110/preview/`; these demonstrate layout,
  not model accuracy.
- Broader checks exposed eight existing display-test failures and eight existing
  multi-jewellery-test failures. Each same failure reproduced with the respective
  source module loaded from Git HEAD. A weight-summary check also failed in the
  untouched weights module (`net_g` returns 0.0 where the test expects None).

Only the new anomaly export is bundled in the project. Historical model
comparisons require an explicit external baseline path. Ordinary audit scripts
use the active model from configuration. Include the new ONNX asset and rebuilt
frontend when copying this update to another installation.
