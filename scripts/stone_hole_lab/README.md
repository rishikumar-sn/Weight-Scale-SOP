# Isolated white-stone/background-hole lab

This directory is an offline experiment. Nothing here is imported by the main
app. Outputs, camera photos, replay reports and masks live under
`data/audits/stone-hole-lab/`, which is ignored by Git. Saved sessions, the app
database and production settings are not changed.

## Findings from 21 October 2026

- Latest capture: `20261021_134420_a434dc32`, item 01. Its 42 V2 candidates do
  **not** reach the primary limit of 48. Final output has 11 regions, including
  seven visually confirmed white chain openings.
- The existing `remove_enclosed_background_from_jewel_mask()` removes zero
  pixels there. It receives a prepared crop whose outside pixels are white,
  so its adaptive background signature does not represent the grey tray.
  Replacing only its input with the original image removes the link holes.
- That broad replacement is **unsafe**: it removes a reviewed real white
  center face in `20260904_112410_79db0245/item_02` (baseline region 1), changes
  another white face, and changes colored regions in other archived captures.
- Pale pendant ornaments sampled in the latest raw image are present in the
  foreground mask. The processed gold mask covers them, including samples
  whose original pixels classify as white. Blindly relaxing the outer-boundary
  rule would not address this cause. Their complete recovery is unresolved.

## Narrow prototype and results

The verifier uses original pixels, true floating-point CIELAB/CIE76 differences,
a spatial empty-tray reference registered by marker translation, a local
photometric offset, texture, metal enclosure, and repeated link context.
It only rejects white regions with all of those signals. Groups require three
individually matching regions on the same jewellery component. Colored,
learned, insufficiently sampled and uncertain regions retain their original
masks. Missing or incompatible reference geometry disables rejection.

Pilot thresholds are explicit in `verifier.py`; they were tuned on the two
latest target images. They are not an independent accuracy benchmark.

454 archived items were replayed with the current code and saved settings:

| Check | Result |
| --- | --- |
| Archive items changed by narrow verifier | 3, all the same problem necklace |
| Reviewed link-hole regions rejected | 16 |
| Latest capture | 11 baseline regions → 4 retained; 7 holes rejected |
| Latest retained mask union | 312 pixels; includes 3 uncertain white regions |
| Archive items with compatible reference geometry | 98 |
| Archive items with rejection disabled by reference geometry | 356 |
| Other baseline region masks/colors | Preserved exactly by the post-filter |
| Live white/colored control after user placed it | 35 → 35, all masks retained |
| Live regions classified white | 3 → 3 |

The live control and empty reference were captured using the existing camera
service with the 90-frame average ready; the camera was released afterward.
No production capture was created, and no scale reading was needed.

**Not ready for integration.** The latest capture still has an out-of-scope
multicolor clasp opening and three uncertain white gaps. Its pale pendant
ornaments remain baseline false negatives. Retained area includes uncertain
regions and must not be described as verified-only stone area. Unchanged
predictions on the 356 incompatible archive items do not prove reference
safety. The three changed archive pictures are one design, not three independent
designs. The new live control is one additional design, not a full safety set.

Open `data/audits/stone-hole-lab/review.html` for before/after images and
`comparison.json` for decisions, counts, areas, fingerprints and limitations.
`review_labels.json` contains the small visually reviewed set; archive
predictions themselves are not ground truth.

## Reproduce from the project root (PowerShell)

```powershell
# Use only when the main app is not holding the camera, with an empty tray.
.venv/Scripts/python.exe scripts/stone_hole_lab/capture_reference.py

# After placing a real white-stone control item, capture and analyze in the lab.
.venv/Scripts/python.exe scripts/stone_hole_lab/capture_reference.py --scene white_stone_control
.venv/Scripts/python.exe scripts/stone_hole_lab/live_control.py

# Current baseline, without touching stored production predictions.
.venv/Scripts/python.exe scripts/stone_hole_lab/replay.py --workers 4 --resume

# Detailed target masks, including candidate and final rejection stages.
.venv/Scripts/python.exe scripts/stone_hole_lab/replay.py 20261021_134420_a434dc32 --detailed --workers 1

# Diagnostic ablation only: do not integrate this variant.
.venv/Scripts/python.exe scripts/stone_hole_lab/replay.py 20260904_112410_79db0245 --mode raw-cleanup --detailed --workers 2

.venv/Scripts/python.exe scripts/stone_hole_lab/summarize.py
.venv/Scripts/python.exe -m pytest scripts/stone_hole_lab/test_verifier.py StoneDetection/test_stone_analysis_v2.py backend/tests/test_ring_stone_fallback.py -q
```

`--resume` trusts completed cached runs. Omit it after changing the detection
code, settings, reference or extraction logic. Camera captures overwrite the
chosen lab scene, so copy the experiment output directory before starting a
different lighting/setup experiment. The marker gate deliberately supports
translation only; it refuses apparent scale/rotation/perspective changes rather
than extrapolating a noisy small-marker homography across the tray.

The lab temporarily wraps detector functions inside its own Python process to
observe candidates and final masks. Wrappers are restored in `finally`, and
are never installed in the production process. The narrow proposal is applied
to completed masks afterward so no candidate ordering, segmentation or colored
stone growth can change.

## Reflection follow-up

Run `.venv/Scripts/python.exe scripts/stone_hole_lab/reflection_probe.py` for the
separate reflection experiment. Results and before/after highlight images are
in `data/audits/reflection-lab/review.html` and `results.json`.

Seven visually reviewed controls were used: two plain gold chains, one plain
gold bracelet, three items with white stones (including the saved live control),
and the problem pendant with missed pale ornaments. These are visual labels,
not measured material identities. Gold-colored gemstones were not independently
identified or validated in this set.

The current reflection warning was false in all 454 cached archive replays.
Its glare extractor requires gray brightness above 240 and removes small
components. It also returns no raw glare on these seven controls.

The experimental alternative uses original HSV value, a positive local
brightness difference against Gaussian blur, minimum three-pixel components,
and the existing reflection coverage/density warning calculation. It was tested
with and without low-saturation gating and a local strict-gold-color veto.

At the representative brightness floor 180 / local difference 20:

| Controls | Relative highlights | With nearby gold-color veto |
| --- | --- | --- |
| Plain metal warnings | 3 of 3 | 2 of 3 |
| White-stone item warnings | 1 of 3 | 0 of 3 |
| Highlight pixels in sampled missed pale ornaments | 0 | 0 |

Across 18 brightness/contrast/saturation combinations and 9 gold-context
combinations, none warned on all three stone controls while rejecting all
three plain-metal controls. Lowering brightness thresholds warns on both groups;
raising them misses the stone items. This is an exploratory control sweep,
not a classifier accuracy estimate or an exhaustive algorithm search.

The reviewed pale ornament interiors have median HSV value 106 and 95th
percentile 124 in the problem capture. They are not strong highlights under
this lighting/view, so this test does not recover them. The reviewed chain-hole
interiors also contain no selected highlights; eliminating holes does not
remove reflections arising on real gold surfaces. On the live control, the
representative detector covers only 16.6% of the three reviewed baseline white
face masks: highlight area must not be used as full stone surface area.

**Conclusion:** reflection can propose a region for further inspection or
describe reflectivity on an independently verified stone. These tested rules
cannot establish that a region is a stone, distinguish gold-colored gems from
gold metal, or justify the assertion "highly reflective stones are present."
No production detector, UI message, area, weight or settings were changed.
