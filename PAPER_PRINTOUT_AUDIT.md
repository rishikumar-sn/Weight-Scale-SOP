# Rectangular jewellery printout fallback audit — 31 October 2026

The requested position is **after the anomaly detector passes the capture**,
before accepting jewellery classification. A detected sheet would reject the
capture and require recapture. This experiment does not meet the required
coverage, so no production rejection behavior has been enabled.

## Evaluated captures

95 saved captures were processed: 30 containing printouts (14 from 30 October
and 16 from 31 October), 64 jewellery controls, and one empty-platform capture.
Two printout captures also contain a real ring. Recent captures and the 50
historical jewellery controls were inspected in contact sheets. These are visual
capture-level labels, not material assays or individual-object annotations.

Saved classifier labels were not used as printout ground truth: for example,
`20261031_162016_7c9ecddd` is visibly a printout but has a confirmed necklace label.
Historical controls were selected from confirmed jewellery captures with saved
testbed masks. Older confirmed captures without those masks were skipped and
listed in the output. Repeated views of the same sheet/jewel are present;
95 captures do not mean 95 independent physical objects.

## Result with a fresh anomaly-model replay

| Check | Result |
| --- | --- |
| Printout-containing captures rejected by anomaly model | 5 / 30 |
| Printout-containing captures reaching fallback | 25 |
| Conservative fallback catches among those 25 | 2 / 25 |
| Jewellery controls reaching fallback | 63 / 64 |
| Additional jewellery-control rejections by fallback | 0 / 63 |
| Combined anomaly + conservative fallback printout rejection | 7 / 30 |
| Looser diagnostic catches among anomaly misses | 4 / 25 |

The conservative fallback caught `20261030_181728_962c3159` and
`20261030_181858_9aff7bad`. The looser diagnostic additionally caught
`20261030_181852_272b2952` and `20261030_182128_89b39024`.
It was evaluated on the retained top 15 candidates per capture, so it is a
diagnostic comparison rather than an exhaustive threshold optimization.

Geometry alone also marked a rectangular plastic packet containing a real chain
(`20261013_172334_c8cad109`) as paper. The anomaly model already rejects that
packet, so it causes no additional rejection at the requested fallback position.
It still demonstrates that rectangular shape does not establish paper material.

Zero additional control flags in this sample is not a guarantee for unseen gold
items. Both mixed real-ring/paper captures were missed by the fallback. Since
capture rejection already fails to cover the printouts, per-object detection and
reliable exclusion of the adjacent real ring have not been established.

## Implemented experimental cues

1. **Perspective-aware reference subtraction:** warp the actual empty-platform
   capture `20261031_140034_91fbf4bc` using the four saved AprilTag corners, then
   compensate median Lab channel offsets and threshold colour residuals. Applied
   only to 30–31 October captures with matching image dimensions and calibration.
   Historical controls have no matched reference and use the other cues.
2. **Canny and Sobel:** multiple edge thresholds and a gradient mask.
3. **Contours and quadrilaterals:** convex four-corner approximations with size
   and testbed-interior checks. No assumption that the sheet is axis-aligned.
4. **Colour and texture:** Lab edge contrasts and mostly neutral paper interiors;
   local colour residuals propose contours. Median gradient texture is recorded
   as a diagnostic, not a validated independent material discriminator.
5. **Line segments:** OpenCV LSD, opposing near-parallel segment pairs, intersections
   to propose quadrilaterals, and accumulated segment support along each side.

The tray rim and calibration marker are excluded. Detection uses original camera
pixels cropped to the processing region, downsampled to at most 720 pixels per
side. It does not depend on the white-isolated jewellery crop, which discards
the paper/platform boundary.

The conservative rule requires edge-support score >= 0.50 (minimum side edge
support multiplied by neutral-interior fraction), line coverage >= 0.30 on at
least three sides, mean line coverage >= 0.45, and median side Lab contrast >= 3.

Weak paper/platform contrast, glare, broken boundaries and small cutouts leave
many sheets without a sufficiently supported complete quadrilateral. Tag-based
reference alignment can amplify corner error away from the small marker; the
reference also comes from a different capture time. A fresh empty-platform
reference for each unchanged camera/lighting setup would be needed to assess
reference subtraction properly. These results evaluate this prototype, not every
possible implementation of the five techniques.

## Outputs and reproduction

Read-only script: `scripts/audit_paper_geometry.py`. It only writes experiment
outputs under `data/audits/paper_geometry_20261031/`. Source captures, stored
classifications, models, settings and application code are not changed.

```powershell
.\.venv\Scripts\python.exe scripts/audit_paper_geometry.py --inventory
.\.venv\Scripts\python.exe -u scripts/audit_paper_geometry.py
.\.venv\Scripts\python.exe -u scripts/audit_paper_geometry.py --summarize --replay-anomalies
```

The capture window is frozen through 31 October 2026. The replay uses local
`models/segmentation/anomolies.onnx` and records its SHA-256. No training or
production-gallery updates occur.

- `review.html`: image gallery with proposed boundaries and evaluation outcomes.
- `summary.json`, `evaluation.csv`: counts and missed printout IDs.
- `results.json`: candidate quadrilaterals, edge/line/colour/texture diagnostics.
- `visual_labels.json`, `all_cases_*.jpg`: capture labels and inspection sheets.
- `anomaly_replay.json`: fresh anomaly results and model checksum.
- `skipped_missing_testbed_mask.json`: excluded historical captures.

Validation: the full geometry audit and anomaly replay completed successfully;
the script also passes Python compilation. This was an exploratory evaluation
with thresholds adjusted on these captures, not an independent held-out test.

Before enabling the requested fallback, improve coverage on the missed sheets
and validate on separately captured printouts and real jewellery, including
mixed placements and packets. The eventual gate belongs after a successful
anomaly check; an anomaly-model error is not a successful pass.
