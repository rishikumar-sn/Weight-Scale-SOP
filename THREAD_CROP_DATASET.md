# Build the cropped jewel/thread training dataset

Run this from the project directory in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_thread_crop_dataset.ps1 -Notify
```

The command uses the app's existing .venv. It processes every saved manifest,
prints progress, appends run.log, and shows a completion popup (15 seconds).
The dataset is written to data/datasets/jewel-thread-crops-20261031. DONE.txt and
summary.json record completion. This setup does not start training or launch the
full run automatically. Run the same command again to resume after interruption.

## Outputs

- images/Jewel_without_Thread: suggested jewellery-without-thread crops, including
  rings, bangles, earrings, Mattal/plain chains without detected thread, and other
  positively predicted jewellery.
- images/Jewel_with_Thread: reviewed seeds plus chain/Mattal item crops with strong
  thread-segmentation overlap inside that same item's foreground mask.
- images/Thread-Only: the known manually reviewed thread-only seeds.
- review/: weak thread evidence, mostly-thread masks, rejected/unverified crops,
  low-confidence predictions, missing coordinate mappings, and unlabelled crops
  retained from old failed thread gates.
- review.html: clickable image gallery for inspection.
- labels.csv and manifest.jsonl: exact source/session, label basis, review flag,
  original pixel hash, reconstruction and duplicate details.
- _cache/thread_masks: cached full-frame segmentation masks for resumable runs.

A small smoke run and tests verified per-item mask mapping, independent labels in
mixed captures, original-pixel preservation, exact duplicate suppression, reviewed
seed priority and resume behavior. No full dataset run has been started.

Use only the reviewed class folders under images/ when assembling the training
dataset. Do not pass the entire output directory (review/, _cache/, reports) to a
classification trainer as if its subfolders were class names.

## Input/label policy

Use native-resolution per-object RAW crops selected by the existing Otsu separation
and stored jewellery predictions, matching the live thread classifier's raw-item
input. A missing old raw crop is reconstructed from its original image and stored
tray/item coordinates. White-isolated working crops are kept only in review when
raw reconstruction is unavailable. No full multi-jewel frame is put in a jewellery
class, and no resizing or square padding is applied to the exported item images.
Do not switch to a different training crop/pad transform without matching it in
application inference when deploying the next model.

The old thread classifier is NOT used to assign labels: it has known false ring
rejections. The 12 reviewed seeds contain 8 without-thread jewels, 2 with-thread
jewels, and 2 thread-only crops, stored separately under
 data/datasets/jewel-thread-reviewed-seeds-20261031.

Other labels are automatic SUGGESTIONS, identified by needs_label_review=true.
Non-chain jewellery is suggested as Jewel_without_Thread. Chain/Mattal inputs use
the full ORIGINAL camera frame for thread segmentation (once per session), then
project masks into each individual crop; thread on a neighbouring object does not
label this item. Positive overlap requires >=24 pixels and >=0.5% of foreground,
using detections at confidence >=0.5. Weaker overlap is review; >=80% coverage is
review rather than automatic Thread-Only. No detected thread is not proof of
absence, so automatically labelled images must be reviewed before training.

Confirmed jewellery labels take precedence over model predictions. Unconfirmed
predictions below 0.5 confidence and rejected/unsupported captures go to review.
Not Gold Jewelry entries are excluded. Legacy labelled captures without saved per-item records are re-separated using
the existing Otsu object splitter. A single component can inherit the historical
label. Multiple components without individual labels are cropped into review
instead of copying a full multi-jewel image into a training class.

Exact duplicate pixels are exported once; manually reviewed seeds are processed
first and win over conflicting automatic suggestions. Source captures/settings
are never modified. No train/validation split is generated: keep all crops from
the same session and repeated views of the same physical jewel together when
splitting. This is a training supplement; merge reviewed samples with the main
training dataset, including any real Empty-Test-Bed examples needed for four-class
training. No synthetic empty backgrounds are added as ground truth.

## Direct Python command and options

```powershell
.\.venv\Scripts\python.exe -u .\scripts\build_thread_crop_dataset.py --resume
```

Use --output to choose another dataset folder, --seed-dir (repeatable) for additional
reviewed class folders, or --max-sessions / --session-id for a smoke run. Options
and the segmentation-model checksum are saved; use a new output directory if they
change. Completed source manifests must remain unchanged during a resumed run.
Sessions with errors are retried when resumed. Check summary.json for errors and
excluded counts. The script never runs the full model training process.
