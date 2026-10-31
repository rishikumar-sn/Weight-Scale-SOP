# Printout capture audit — 30 October 2026

The current models recognize the jewellery pictured on paper. They do not reliably establish that a physical jewel is present. Depth Anything V2 Small also predicts apparent relief on the printouts. A separate physical-object/printout gate is needed before jewellery type classification.

## What was inspected and run

- Inspected all 14 non-tare captures from today, `20261030_181639_883281be` through `20261030_182128_89b39024`. All visibly contain printed jewellery images, consistent with the user's ground truth. Six reached `awaiting_confirmation`; eight were rejected by other checks. None of these six is described here as confirmed or analyzed successfully.
- Visually inspected 20 recent confirmed historical captures; selected eight for inference: four captures from 27 October on the same ACZET setup, two necklaces from 21 October, one bangle from 22 October, and one six-item capture from 23 October. The exact IDs are in `scripts/audit_printout_captures.py:HISTORY`.
- Ran the existing local **Depth Anything V2 Small HF model**, on CPU, at requested size 518: **22 captures × two contexts = 44 inferences**. Inputs use the saved processing square around the detected tray: (1) original camera pixels in that square and (2) the same square with pixels outside the saved tray mask white. The foreground itself is never cut out for depth inference. Measurements use only the tray interior; the calibrated fixture and AprilTag are excluded from background fitting where available.
- Reran the production foreground component separator and SigLIP base384 classifier on **44 components × raw and masked views = 88 category predictions**, with the existing correction gallery. Also ran the raw/masked semantic rejection check. These are diagnostic classifier calls, bypassing the service's earlier rejection gates so their failure modes can be observed.
- Reran the anomaly segmentation model on all 22 original full frames, associating its masks with the same foreground components. Also replayed two historical plastic-packet captures.
- Saved depth arrays, input views, comparison images, component montages and JSON measurements under `data/audits/printout_20261030/`. Source captures, confirmations, gallery learning, settings, and production behavior were not intentionally modified. The pre-existing modification to `data/settings/app.json` was left alone.

Repeat from the repository root:

```powershell
.venv/Scripts/python.exe scripts/audit_printout_captures.py depth
.venv/Scripts/python.exe scripts/audit_printout_captures.py siglip
.venv/Scripts/python.exe scripts/audit_printout_captures.py ownership
```

The SigLIP mode resumes completed sessions from its JSON output. Use a fresh output directory or move the prior JSON before comparing different model/gallery configurations. It does not call correction learning. HF downloads are disabled in the script. The packet check and inventory/contact sheets were additional one-off audit steps; the three commands reproduce the principal model comparisons.

## Observed failures

### Printout detection

The saved anomaly results contain **zero paper detections in all 14 captures**. Only the first capture has a foreign-object detection (0.533). Thirteen captures have a high-confidence `jewellery` instance even though the object is a printout; those instance confidences range approximately 0.882–0.958. The current model is strongly responding to the printed subject.

Fresh inference gives maximum unfiltered paper-class anchor scores of approximately **0.000072–0.04649** across the printouts, below the current 0.15 paper threshold. Some real captures have higher paper scores than some printouts. Merely lowering the paper threshold cannot separate this sample reliably; these unfiltered scores are diagnostic anchor maxima, not calibrated image-level probabilities.

Every printout retains a component labelled `Chain / Necklace` in the fresh masked SigLIP replay. Changing the global confidence threshold would also be insufficient: some printed-necklace predictions exceed 0.93. The correction gallery even confirms the visual resemblance of some printed items to known necklaces; it supplies no physical-presence evidence.

### Paper outlines become extra jewels

In `20261030_181915_f80d25c3`, the centre print is J2. The two long border/background strips become J1 and J3. Fresh masked classification calls both strips `Finger Ring` at roughly 0.30 confidence. The saved service result changes these to `Bangle` through the calibrated size fallback. One strip measures about **2.01 × 133.00 mm**. Its long span meets the size rule even though the mask has no plausible ring/bangle body.

In `20261030_181728_962c3159`, the printed necklace becomes J1 and the long bottom paper edge becomes J2. That edge changes from `Mattal` with raw context to `Finger Ring` after masking, and then to `Bangle` in the service size fallback.

The mask builder in `backend/app/analysis/vision.py` unions thresholded darkness, dilated Canny edges and colour evidence, then closes gaps. Paper boundaries, dim tray areas, creases and shadow edges can therefore become connected foreground. The separator's aspect-ratio filter only rejects narrow components below an area limit; sufficiently long paper strips survive. Removing surrounding pixels can also erase the evidence that a component belongs to paper.

Across today's 31 fresh components, masked SigLIP gives 22 jewellery labels; raw SigLIP gives 31. Fifteen components change category between views. Raw context is useful evidence, but using raw crops alone does not solve the problem.

### Depth and geometric features

| Diagnostic feature, raw tray crop | Printouts, 14 | Confirmed real, 8 |
| --- | --- | --- |
| Foreground minus background median relative depth | −0.063 to 0.772 | 0.322 to 0.867 |
| Median absolute foreground residual from fitted tray plane | 0.071 to 0.647 | 0.272 to 0.808 |
| Foreground relative depth P90−P10 | 0.464 to 1.137 | 0.622 to 1.320 |

Depth is normalized by each tray's P05–P95 range. These are **relative model units, not millimetres**. Foreground statistics include the existing mask's artifacts; they are not measurements over manually labelled jewel pixels. A robust affine background-plane fit is only a diagnostic of the predicted map, not recovered physical geometry.

The ranges overlap. Changing the exterior of the tray to white also changes depth scores: for example `20261030_182019_73a63f93` changes from 0.189 to 0.509 foreground/background contrast. The model visibly gives relief to the printed necklace and sometimes the sheet itself. A depth-derived surface rendering would visualize that same prediction and would not provide independent 3D evidence.

Long straight-line detection was tested using Canny/Hough within an inset tray mask. It finds paper borders, but also straight sections and repeated edges in real necklaces. One confirmed real necklace produces more long-line detections than any printout. Use enclosing paper geometry and surface evidence together; a line-count or line-length threshold alone is unsafe.

The official [Depth Anything V2 repository](https://github.com/DepthAnything/Depth-Anything-V2) distinguishes its relative-depth model from metric variants. This experiment tests the existing relative model, with no measured height reference.

### Mask agreement and packet labels

Seventeen of today's 31 components have **zero** overlap with a detected YOLO jewellery-instance mask. The 13 components across the eight selected real captures all have overlap, with maximum per-instance component coverage approximately 0.831–0.980. This is promising evidence for routing unsupported fragments to **uncertain**, not a validated universal threshold. One glare-heavy printout has no jewellery instance at all; conversely, some paper pixels can lie inside an overly broad jewellery mask. Printed main necklaces usually have strong YOLO support, so this check does not establish authenticity.

Two saved packet examples, `20261021_175409_a2850223` and `20261021_182641_e20ab9ca`, are visibly actual ornaments inside transparent bags. Fresh anomaly inference detects `plastic_packet` at 0.624 and 0.478, above the 0.35 threshold; both saved sessions are rejected. Their old manifests retain a necklace classification as well. This establishes that A and J annotations can coexist in historical rejected evidence; it does not establish that a packet was accepted.

The current `AnalysisService.classify` checks anomalies before either classifier and returns immediately when the check fails. Component ownership filtering uses a foreign-mask overlap requirement of 50%; thin reflective borders or separate shadows can fall outside that mask. If a current deployment still classifies J items after A1 causes rejection, compare its loaded code/version and capture timestamps with this checkout. Do not assume the J numbering is YOLO's physical object count: J labels originate from foreground components.

## Recommended implementation order

1. **Add a dedicated physical-presence gate before category classification.** Predict `physical jewellery`, `printed/displayed jewellery`, `foreign object` and `uncertain` from raw tray context plus local detail patches. Train with today's printouts as hard negatives, including the pictured jewellery itself, rather than relying only on finding white paper. Include glossy/matte prints, cropped borderless prints, curved sheets, screens, empty cards, packets, reflections, and actual thin/flat/stone-set pieces. Paper detection and jewellery-content detection are separate tasks.
2. **Train the paper/packet detector on these failures.** Annotate the entire sheet, including the printed ornament, blank interior, corners and curled edges. Reject a capture containing a forbidden sheet/packet even when it also contains a genuine jewel. Retain the current anomaly-first service behavior.
3. **Treat foreground components as proposals.** Check learned jewellery objectness, YOLO mask support and raw/masked consistency before assigning a J identity. Unsupported strips should trigger uncertainty/recapture. Keep an explicit pixel-ownership map for jewellery, paper/packet and uncertain pixels; do not classify detached packet glints as independent jewels. Expand foreign masks only with learned boundaries or supported context, since arbitrary dilation can erase nearby genuine jewels.
4. **Gate size correction behind valid object geometry.** A long span must not turn a paper strip into a bangle. Require physical-jewellery evidence and plausible ring/bangle shape before applying the size fallback. Include open rings and standing bangles in validation so a simple hole requirement does not reject them.
5. **Improve segmentation with calibrated background and semantic masks.** Capture an empty tray per camera/lighting setup, align it with the AprilTag, normalize illumination and compare colour/texture as well as brightness. Model low-chroma shadows separately. Preserve specular white pixels only when their ownership belongs to a supported jewel; do not declare every white reflection background. Use Otsu for local boundary refinement inside supported objects rather than as the source of object identity. Colour-only filters would miss white metal/stone highlights and still accept gold-coloured ink.
6. **Keep SigLIP as the type classifier after presence passes.** Give the presence model raw context and the category model the isolated object. Test explicit printout/paper-edge/packet/reflection prompts as supplementary rejection features, but do not treat prompt similarity or gallery similarity as proof of physical material. Calibrate an abstention decision using held-out examples; no universal confidence cutoff is established here. Never learn rejected/uncertain printouts into the jewellery gallery.
7. **Use depth as one supporting feature.** Evaluate local support-relative relief, plane residuals, gradients and consistency across views jointly with material/context evidence. The current study does not justify a production depth threshold. Higher input resolution may reveal more detail, but is not an independent test of physical relief.
8. **For stronger evidence, acquire a controlled capture pair.** With fixed exposure and the camera stationary, alternate left/right illumination and inspect movement of real cast shadows and local highlights. Use a short unaveraged burst: the current 90-frame rolling average mixes temporal cues. Alternatively, move the camera by a known small baseline and align the tray plane to measure residual parallax, or evaluate calibrated stereo/structured-light depth at the working distance. Glossy paper, sheet curl, reflective metal, small heights and occlusions still need testing. These options require new captures/hardware and were not tested here.

Material, texture, depth and reflection cues have also been combined in [presentation-attack research](https://arxiv.org/abs/2007.02157). Applying that idea to jewellery is an engineering proposal, not evidence that a face anti-spoofing model transfers directly.

## Validation before release

Split by physical jewel, source print image, printer/paper and capture day; adjacent duplicate frames must stay in one split. Include hard genuine cases: thin chains, lying-flat bangles, open rings, pale gold, stones, strong highlights, threads and shadows. Measure printout false acceptance, genuine-jewellery false rejection, uncertainty/recapture rate and item-count accuracy separately. Include mixed real-plus-print and jewel-inside-packet scenes.

The 14 printout frames contain repeated designs/poses, and the eight real samples contain repeated objects and different fixtures. This is a diagnostic comparison, not a held-out accuracy benchmark. No new presence model was trained, no active lighting/parallax experiment was performed, and no production rejection thresholds were changed. Physical presence detection also does not certify gold purity.

Evidence files: `inventory.json`, `today.jpg`, `history.jpg`, `depth_results.json`, `siglip_results.json`, `ownership_results.json`, `packet_results.json`, and each session's `comparison.jpg` / `components.jpg`, all in `data/audits/printout_20261030/`.
