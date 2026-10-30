# USB update import — 30 October 2026

Imported 13 new or changed deployment files from
`E:\weightScaleinstall\WEIGHT-INTEGRATION-APP`. Each copied file was verified
against the USB source with SHA-256. Other compared application code and model
assets already matched. Local settings, captures, database and environments
were preserved.

The update adds `models/classification/jewel-thread-small.onnx` and its metadata,
the CPU classifier, per-item thread checks, rejection UI, launcher/model
preflight checks, and export/verification/audit scripts. The existing thread
segmentation model remains in place.

The model predicts `jewel_with_thread`, `jewel_without_thread`, or `thread_only`.
Anomaly and placement checks run first. Every separated item must pass the
thread gate before jewellery classification; a thread-only result rejects the
capture and requires recapture. Confirmation overrides and queued analysis
cannot bypass a persisted failed thread check.

## Verification

- Frontend production build passed and regenerated the matching USB bundle.
- All four scale-capture frontend tests passed.
- Actual ONNX inference completed on 12 curated USB review crops.
- Four thread-gate workflow cases, two anomaly/placement ordering cases,
  rejected-input preservation, API override and queued-job guards passed.
  The USB verifier was run with temporary input/output paths because its
  original session manifests are absent from the deployment bundle. Its workflow
  fixture used the supplied tassel crop and a temporary full crop mask, rather
  than replaying the original full camera frame.
- Backend/general tests: 265 passed, 37 failed, 25 subtests passed. Repeating the
  suite with the pre-import `HEAD` analysis service produced the same 37 failing
  test names and pass counts. These existing failures remain unresolved.
- `git diff --check` passed. No live camera or scale capture was performed.

## Supplied model limitations

The curated samples reproduce seven genuine rings classified as `thread_only`
and one mixed-capture thread-only item classified as `jewel_with_thread`.
The latter can incorrectly pass the gate. This import preserves the supplied
model and preprocessing; it does not resolve these accuracy issues. The sample
set is deliberately failure-heavy and is not an accuracy benchmark.

Source evidence: `E:\weightScaleinstall\testing\mixed-thread-20261030\report.md`.

Backup of replaced files and import hash manifest:
`C:\Users\karth\AppData\Local\Temp\weight-usb-import-20261030-181016`.

Local thread verification artifacts:
`C:\Users\karth\AppData\Local\Temp\weight-thread-verification-88i4sgoc`.

Restart the backend to load the imported classifier and service changes.
