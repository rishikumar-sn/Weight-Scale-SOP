# Tray rim foreground filter — 2 November 2026

Otsu proposals can include the detected tray's raised rim. A proposal with no
jewellery-instance support previously rejected an otherwise valid capture.
The filter in `backend/app/analysis/rim_filter.py` now runs after the existing
foreign-object/placement gates and before per-item thread and type classification.

It uses the detected testbed polygon and AprilTag physical scale. It does not
change the ROI, Otsu extraction, model weights, prompts, category thresholds,
item masks, dimension calculations, or stone analysis.

An exclusion requires all of the following:

- The anomaly and placement gates pass. An explicit foreign-object detection
  cannot be ignored, including at the border.
- AprilTag calibration is valid and the detected testbed confidence is at least
  0.50. Its mask has one outer contour. Internal marker/fixture holes are filled
  only for computing distance to the outer boundary.
- At least one other proposal has positive jewellery-instance support. A tray
  with only unsupported components cannot become an accepted empty capture.
- The proposal has **zero** instance overlap. Even partial overlap below the
  existing 20% presence threshold protects a possible jewel from exclusion.
- Every actual foreground pixel lies within 5 mm of the outer boundary, with a
  second cap of 4% of the working-image short side. Padded crop boxes are not
  used to decide proximity.
- The strip is at most 3 mm thick, with rotated aspect ratio at least 3. Its
  boundary-distance variation is at most 1.5 times its short dimension, so a
  narrow object pointing into the tray is retained.
- The strip has no closed opening. Fewer than 2% of its raw foreground pixels
  are coloured (HSV saturation >=60 and value >=45); coloured metal, stones and
  thread are retained.
- Raw-crop SigLIP produces neither a learned positive jewellery match nor a
  jewellery category confidence of 0.70 or more. This conditional category
  score is a conservative veto, not a calibrated probability of jewellery.
  A semantic-check exception retains the component and the existing gates.

Interior or ambiguous components remain subject to the existing verification
and rejection rules. Absence of jewellery support does not prove that a region
is a foreign object. The filter never invents a foreign-object label.

Each decision is saved as `rim_edge_check`, including original component index,
box, boundary distance, geometry, colour share and semantic evidence. Detector
component ownership is remapped after exclusion; retained image/mask arrays are
unchanged. Historical sessions are not rewritten.

## SigLIP investigation

The category competition contains jewellery classes only. Broad gold validation
is advisory, because using it globally can reject thin genuine jewellery. In
this capture the raw rim has gold similarity about 0.039, which exceeds the
existing strong-negative gate's maximum of 0.035. Its negative similarity is
about 0.079, but the gold ceiling prevents rejection. It therefore receives a
weak Finger Ring category guess (~35%, raw; ~30%, masked), without a gallery
correction. These percentages do not prove that the rim is jewellery.

The most similar individual ring prompts describe a small circular finger band
and a V/chevron ring. They are valid descriptions of genuine rings and were not
removed. Four tray-rim-specific prompts were tested separately using the local
text encoder. They score about 0.104 on the raw rim and 0.075 on the masked rim;
neither reaches the tested conservative 0.11 specific-object threshold. A global
prompt modification was therefore not deployed. An exploratory 0.07 threshold
rejects both rim versions without rejecting the 36 control crops, but it was
selected using this same failure and has not been validated on independent rim
and foreign-object scenes. Production prompts, text caches,
gallery and model weights remain unchanged.

Reproduction and detailed local evidence are in `testing/rim-rule-20261102/` in
the installation root. Run `test_rim_filter.py`, `audit_saved_captures.py` and
`test_edge_prompts.py` with the application's Python environment. Audits write
only below that testing directory and perform no confirmation or learning.

This is a conservative heuristic with regression evidence, not a guarantee of
zero errors for every future scene. An unsupported neutral thin object lying
parallel to the rim can resemble a rim strip. Supported jewellery, coloured
foreground, closed shapes, confident semantic positives and explicit detected
anomalies are protected. If evidence is insufficient, recapture is retained.
Restart the backend (or start it normally if closed) to load the change.
