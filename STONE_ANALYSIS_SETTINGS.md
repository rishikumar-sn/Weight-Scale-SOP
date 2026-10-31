# Stone analysis settings and color summaries

Settings > Stone analysis is below Weight scale. Enable stone weight estimation
is off by default and shared across all scale models and Windows/Linux profiles.
The setting is captured with each new capture; historical reports retain their
saved setting and existing reports are not rewritten.

The main result image omits the stone description. The dedicated stone analysis
image and PDF show the description. When enabled, the existing single-jewel
stone/net estimates also appear in the result image, stone image and PDF. Multiple-jewel captures still
omit estimates because individual scale weights are unavailable. When disabled,
the stone section prints only its description, without weight estimates or a
stone-area value. The measured scale weight remains in the capture summary.

Color analysis uses the existing region masks only as locations. Native capture
pixels are mapped through the tray and item offsets, including virtual square
padding. Masks are mapped to native resolution with nearest-neighbor interpolation;
photograph pixels are never enlarged, color-normalized or interpolated for color
classification. Interior sampling reduces contamination from metal edges.

Each region uses at most 1,536 source pixels and four CIELAB K-means clusters with
15 bounded NumPy Lloyd iterations. Deterministic initialization avoids changing
OpenCV's global random state used by segmentation. Chromatic cluster distributions
are compared with reference shades; neutral shadows and clipped highlights do not
outvote sufficient colored evidence. Weak, clipped, or conflicting evidence yields
Uncertain. Confidence is an algorithmic evidence score, not a calibrated probability.
No new models, training, or runtime dependencies are required.

Descriptions omit Uncertain and low-confidence colors and mention at most two
colors with at least 20% of confidently colored region area. Black, purple and grey
are grouped as dark colored. Regions below 12 native pixels or 0.5 mm? are omitted
from descriptions. Without scale calibration, the area threshold is 0.05% of the
jewellery mask. These filters do not change detection, segmentation or deductions.
No descriptive sentence is printed when no reportable stones are present.

Sizes use equivalent face-up diameter: small below 2 mm, medium 2 to below 4 mm,
large at least 4 mm. Without calibration, relative jewellery-mask area is used:
small below 1%, medium 1 to below 5%, large at least 5%. Large connected regions
(at least 25 mm?, or 15% without calibration; at least 85% connected) are described
as continuous stone areas; this is a region description, not a verified individual
stone count. Uncertain classifications remain in diagnostic JSON for review but
are omitted from printed color descriptions.

## Verification

Run the saved-capture batch without overwriting production captures:

```powershell
.venv/Scripts/python.exe scripts/batch_stone_colors.py --limit 30
# Use --limit 0 to process all saved stone captures.
```

Outputs are isolated under experiments/stone_color_batch, including per-capture
images, diagnostic JSON and results.json. The final wording/layout was also checked
with a 12-capture batch under experiments/stone_color_final. Captures lack human
color ground truth, so batch success checks execution and output behavior rather
than establishing color accuracy. Color thresholds and size boundaries can be
refined with labelled captures during calibration.

Focused tests cover source coordinate mapping, shadows/glare, red versus pink,
neutral/clipped evidence, uncertain/tiny summary filtering, scale-independent
settings, image/PDF estimates, and unchanged segmentation/measurement outputs.
The frontend production build passes. The wider existing regression selection has
11 failures also reproduced against the unchanged commit; these relate to existing
ring routing, weight cap expectations and display text expectations.
