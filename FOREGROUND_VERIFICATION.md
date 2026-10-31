# Foreground verification — 31 October 2026

## Implemented behavior

Otsu connected components are proposals, not confirmed jewellery identities.
Before thread classification, SigLIP, dimension fallback or gallery learning,
each component must have at least **20% of its foreground pixels supported by
a detected jewellery-instance mask**. The existing anomaly model provides these
masks at its unchanged 0.50 jewellery confidence threshold. This uses pixel
overlap, not enclosing rectangle overlap.

An unsupported component rejects the entire capture as `unverified_foreground`:
no J labels, no jewellery types, no analysis and no correction learning. The app
requests recapture. Components are not silently deleted: a missed genuine jewel
must not disappear from the count or weight interpretation. Missing instance
evidence is also uncertainty, not permission to fall back to SigLIP.

Any detected `foreign_obj`, `paper` or `plastic_packet` remains authoritative:
the whole capture rejects before either classifier. A detector exception now
persists a label-free `verification_unavailable` rejection, clearing any previous
classification, rather than leaving a stale result available after a retry.

The review renderer suppresses all J labels when anomaly/presence evidence
blocks a capture. Public API state also removes stale jewellery classifications
and results from legacy blocked captures. Requests for their saved review image
render a clean image from original pixels in memory, retaining A annotations.
The recent-history API also hides their stale category labels. Historical
captures are not rewritten. Confirmation/correction requests cannot
bypass these blocks, even if a legacy status says `awaiting_confirmation`.

Main code:

- `backend/app/analysis/component_presence.py`
- `backend/app/services/analysis_service.py`
- `backend/app/analysis/annotations.py`
- `backend/app/main.py`
- `frontend/src/App.tsx` (neutral recapture heading for uncertainty/model failure)

No model weights, category prompts, correction gallery or anomaly thresholds
were changed. Printout authenticity detection is deferred as requested: a
printed ornament that receives convincing instance support can still pass this
new check when no sheet/edge anomaly is found.

## What was inspected

Fresh CPU inference on **94 saved captures / 169 foreground components**:
all 88 available captures dated 15–30 October with the required saved image and
tray-mask assets, plus six earlier confirmed captures with those assets.
The latest saved captures available at audit time were from 30 October; no
31 October capture was present. This does not claim coverage of the entire older
archive, much of which lacks the required saved tray masks.

All **89 components in 50 confirmed captures** passed the new presence check.
Their minimum support was approximately 0.390, above the 0.20 requirement.
There were **35 unsupported unconfirmed components**, including tray rims,
empty tray background, marker fragments, fixture/foreign objects, plastic
glints and paper outlines. The old type-ranking result for those components was:

| Highest masked category score | Components |
| --- | ---: |
| Finger Ring | 23 |
| Earrings / Nosepin | 8 |
| Chain / Necklace | 2 |
| Mattal | 2 |

Twelve captures had anomaly detections. Eighteen more had unsupported foreground
despite no detected anomaly. Nine of those eighteen predate the printout tests,
including detached tray edges, the tray fixture and an empty-tray region. These
are the important anomaly-miss cases addressed by the new uncertainty decision.

This is retrospective diagnostic coverage, not a held-out accuracy estimate.
The 20% requirement is deliberately permissive for partial masks, and all
available confirmed controls passed; future camera/fixture changes still require
validation. A false positive jewellery-instance mask can support a non-jewel.
This change cannot guarantee detection of every unseen shadow or reflection.

## Which embeddings matched the artifacts

Inspected **103 components in raw and masked form (206 embedding comparisons)**:
every unsupported component and every component in recent multi-component
captures. Outputs include every class prototype cosine/category score, group
prototype cosine, and five nearest correction-gallery indices and labels.

The category classifier compares six jewellery classes; it has no background
category in that ranking. Its class vectors are normalized averages of product
prompts, not one individual prompt per prediction. Group priors can change
which class wins even when another class has a slightly larger cosine. The
category percentage is conditional on that jewellery-class competition, not
the probability that the fragment is a real jewel.

| Example / component | Highest masked category score | Class cosine | Negative-group cosine | Gold-group cosine |
| --- | --- | ---: | ---: | ---: |
| `20261021_134357_f5754f55` C3, tray edge | Finger Ring, 31.5% | 0.0585 | 0.0835 | 0.0409 |
| `20261021_175152_30daafe4` C3, packet glint | Finger Ring, 65.9% | 0.0497 | 0.0608 | 0.0199 |
| `20261030_181915_f80d25c3` C1, long edge strip | Finger Ring, 30.3% | 0.0539 | 0.0704 | 0.0390 |

The tray edge's nearest gallery vector is index 64, `Not Gold Jewelry`, cosine
0.8878. The packet glint already has an essentially exact `Not Gold Jewelry`
gallery match at index 1398. The long strip's nearest vector is index 1407,
`Not Gold Jewelry`, cosine 0.9037. Thus the gallery already knows some of these
negative examples; category-only scores must not be confused with the current
final classifier decision. Exact/strong negatives can already reject after
classification, but unfamiliar edges remain vulnerable. The new guard does not
depend on remembering a particular crop in the gallery.

Broad gold verification is advisory in the current classifier, and strong
non-jewellery rejection has conservative absolute-similarity/margin conditions.
For example, the tray edge's gold similarity exceeds the existing 0.035 ceiling,
so stronger negative evidence does not itself force that raw semantic rejection.
The current gallery format records vectors and labels without source-image
filenames; the audit therefore reports indices, not invented source matches.

Masked crops erase useful packet/tray context and leave thin silhouettes that
resemble rings, earrings or chains. The subsequent size fallback can turn a
long ring-labelled strip into a bangle. Preventing unsupported fragments from
reaching those steps addresses that chain of errors without changing jewellery
type or sizing thresholds.

## Validation

- **55 focused tests passed**, covering mask support, raw-image anomaly gating,
  uncertainty, model failures, stale predictions, renderer suppression,
  confirmation/learning blocks, supported multi-item classification, existing
  classifier decisions, SigLIP integration and Otsu masking.
- Frontend production build passed; Python compilation and `git diff --check`
  passed.
- Real service replay uses original pixels, local ONNX models, a copied gallery
  and temporary output directories. Packet/anomaly and unsupported-region
  captures produce zero classified items and cannot be confirmed or learned.
- Two older genuine controls pass the new presence gate but are rejected by the
  existing thread-only classifier: `20261021_175701_21511494` (ring) and
  `20261023_112624_26c35dc9` (small mixed jewellery). This is a separate existing
  model limitation; its thresholds were not changed here.
- The older three-file workflow/display suite had 20 failing tests before this
  work and now has 16 failures / 39 passes. Every remaining failure was present
  before these changes. Focused foreign-object fixtures were updated for current
  thresholds, component masks and persisted rejection behavior. The remaining
  older layout/workflow failures are not claimed to pass.
- No live camera, lighting or scale experiment was performed. Restart the
  backend and load the rebuilt frontend to use these changes.

Reproduce the model audit:

```powershell
.venv/Scripts/python.exe scripts/audit_component_presence.py support
.venv/Scripts/python.exe scripts/audit_component_presence.py embeddings
.venv/Scripts/python.exe scripts/audit_component_presence.py service
```

Evidence is in `data/audits/component_presence_20261031/`: `support.json`,
`embeddings.json`, `prompt_provenance.json`, `service_replay.json`, component
contact sheets and clean packet-review images. No production confirmation or
gallery learning is performed by these commands. The gallery snapshot is copied
into the ignored audit directory; it is not an upload.
