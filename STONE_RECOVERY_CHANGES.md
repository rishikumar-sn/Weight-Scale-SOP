Stone recovery review — 24 October 2026

Investigated session 20261024_152502_f448b5c3, item 1. The previous replay
produced 245 proposals but processed only 48. Tiny high-confidence regions
occupied much of that budget. The non-gold residual joined touching beads into
one 76,764-pixel component, exceeding its 29,286-pixel per-component limit.
That component was discarded before splitting. Most individual bead proposals
therefore lacked the residual source tag required by overflow recovery.

Changes in StoneDetection/stone_analysis_v2.py:
- Split oversized connected colored regions with watershed before applying
  the existing per-face area guard. Allow up to 128 parts on this path; ordinary
  downstream instance splitting retains its eight-part limit.
- Validate split faces with >=80 pixels, >=80% saturated non-gold/non-warm hue
  evidence, solidity >=0.55, compactness >=0.18, and aspect ratio <=3.5.
- Allow independently segmented HSV/LAB proposals to recover beyond the primary
  budget under the same chromatic and shape checks, followed by existing color,
  gold, overlap and refinement checks.
- Increase bounded overflow capacity from 24 to 128. Keep primary processing at
  48 and neural refinement at eight. Overflow uses deterministic OpenCV proposal
  validation and existing protection against overlap with accepted instances.

Fresh replay results:
- Pink necklace: 33 -> 63 final regions; jewelry stone coverage 24.71% -> 54.64%;
  calibrated stone area 1738.25 -> 3839.07 mm².
- Approximately 94% of strong pink pixels are now covered, versus approximately
  38% before. This is a color-pixel proxy, not annotated stone recall or proof
  of gemstone identity. Small edge/glare areas can remain unmasked.
- Saved 20261024_131129_9886f1a5 stays at zero final stone regions.
- Saved 20261024_141233_6fd594a8 item 1 stays at one region / 0.02% coverage.
- Synthetic checks recover 60 colored beads beyond a full primary budget,
  reject warm metal, white glare and neutral shadows from the new recovery
  path, and split a connected 12-bead strand before the area guard.
- Existing stone_area_calculator validation passes.

Repeat from the application directory:
  .venv/bin/python scripts/verify_stone_overflow.py
  .venv/bin/python scripts/verify_saved_stone_recovery.py
  .venv/bin/python StoneDetection/stone_area_calculator.py

Replays and previews: ../testing/pink-stones/replays/.
Original saved session files and database remain unchanged. Restart the backend
and rerun stone analysis (or capture again) to update application results.
