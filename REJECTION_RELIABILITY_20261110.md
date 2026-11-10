# Rejection localization and MobileNet coin correction

## Findings in the last three captures

| Capture | Recorded decision and cause | Replay with corrected model |
| --- | --- | --- |
| `20261110_125324_16877aa0` | Rejected: foreground component 3 had zero supporting jewellery/coin/bar mask overlap. The anomaly model itself returned no blocking detection. | Still rejected; U3 marks the rectangular object at bottom right. |
| `20261110_125422_a771ae9c` | Rejected: lower small coin was REVIEW in REAL/PRINT verification. Tight print score 0.933071; context/fused score 0.541092, inside the 0.45–0.55 review band. | All four objects pass REAL/PRINT; classification proceeds to confirmation. Small coin context print score is 0.002949. |
| `20261110_125453_2e1af416` | Completed: two coins, both passed REAL/PRINT. | Both still pass; classification proceeds to confirmation. |

REVIEW was an uncertain real-versus-print decision, not a preview, proof of a
print, or a measurement of gold authenticity. Coin/bar type recognition likewise
does not establish gold content or purity.

## Review presentation repair

New captures archive foreground-component locations in original camera
coordinates. Failed foreground, anomaly, thread, placement, semantic and
REAL/PRINT checks now produce structured rejection regions. The same region IDs,
locations and explanations drive both UI messages and exported review labels.
Valid objects are not labelled as the cause of a failed foreground check. Missing
locations remain explicitly unavailable rather than being guessed. Reclassification
clears stale stage evidence, while existing print/review rejection latches remain.

U-tags indicate unverified objects; A-tags indicate predicted anomalies; P-tags
indicate REAL/PRINT failures, with uncertain coins explicitly labelled
`P4 Coin: Verification uncertain`. Thread, placement and semantic failures also
retain their individual marked regions. No gold verdict is inferred from these
checks.

The first two saved rejected reviews were repaired in place. Original manifests
and review images are backed up under
`data/audits/rejection_reliability_20261110/before/`. Their recorded rejection
statuses, raw captures, inference scores and database fields were preserved.
New-model acceptance is demonstrated in separate replay images, not substituted
for historical decisions.

## MobileNet correction

The deployed `models/classification/real-print-mobilenetv3.onnx` now uses the
classifier-head correction checkpoint
`experiment/real_print_lab/checkpoints/run_20261110_coin_correction/best.pt`.
Production model SHA-256:
`b8911d1d2887b5ea6102ff3aaeeb8f72badb0689ee00a0db1aa4a4aef767b7b6`.

The visual backbone was frozen. The final classification layer was trained using
the original 169 training crop pairs plus three original-pixel tight/context
pairs of the user-confirmed physical small coin. All three views belong to one
training group. The original 37 validation samples and 36 test samples were not
moved or relabelled. Selection used validation loss while retaining its 37 correct
decisions and correcting the new training examples. The untouched test set was
evaluated after selecting the checkpoint.

Threshold 0.50, uncertainty band ±0.05, context padding 0.40, tight padding 0.10
and fusion weight 0.0 remain unchanged. Coins still run REAL/PRINT verification;
they do not bypass the print gate. Older weights and metadata are archived
outside the project at
`D:/PROJECTS_BY_RISHI/models/real_print_model/before_coin_correction_20261110/`.

The correction script and detailed evaluation are in the lab folder. Three views
of one physical coin are training examples, not independent evidence of coin
generalization. More independent coins, bars, lighting conditions and printed
coin/bar attacks are still needed for a broad reliability estimate.

## Verification and activation

- Frozen validation: 37/37 correct.
- Untouched test set: 36/36 correct, including 21 REAL and 15 PRINT.
- Eight additional saved PRINT crops remain PRINT.
- Torch/ONNX probabilities agree within 2e-5 for the 36 test pairs and three
  correction pairs.
- All three capture replays exercise the real anomaly and corrected MobileNet
  models in temporary sessions, without SigLIP on recognized coins/bars.
- Focused regression suite: 144 passed. Frontend production build passed.
- Corrected rejected-review and new-model replay images were visually inspected.

Replay evidence: `data/audits/rejection_reliability_20261110/`.
Reproduce with `.venv/Scripts/python.exe scripts/replay_rejection_reliability.py`.
Use `--repair-reviews` only when also updating the two historical presentations.

Restart the running app and refresh its browser, then make a new capture. Cached
ONNX sessions and previously latched rejection decisions are not silently replaced
in a running capture. No backend restart or live camera/scale test was performed.
These changes are local; USB drive E: was disconnected during this repair.
