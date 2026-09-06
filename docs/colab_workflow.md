# Colab workflow

## Prepare code and storage

Notebook 01 needs only a standard Colab **CPU** session and its bundled Pillow.
It does not install packages or write audit outputs before audit/preview review.
For future notebooks 02/03, select a GPU runtime. The pinned stack targets Python 3.10–3.12, PyTorch
2.5.1 / torchvision 0.20.1 with CUDA 12.1 wheels. The installer fails on unsupported
Python versions; these pins have not yet been verified in a fresh Colab runtime.

Upload a code-only ZIP of this repository and extract to
`/content/aquafina-yolo-detector`. Include `src`, `configs`, `requirements`, `tests`,
`notebooks`, `docs`, `scripts` and `pyproject.toml`. Exclude `.test-tmp`, datasets,
weights, build artifacts, caches, notebook outputs and `.git`. Open the relevant
notebook in Colab. Later, a user-approved pushed repository can be cloned instead.
Colab never needs to commit or push your code.

```text
/content/drive/MyDrive/aquafina-yolo/
  raw/temple/extracted/detection_dataset/   Immutable TempleRAIL source
    JPEGImages/                     4,870 images
    Annotations/                    4,870 XML annotations
    Labels/                         Paired Darknet TXT labels
    ImageSets/Main/                 Original train.txt and val.txt
    Readme.txt
  processed/temple/                 Confirmed conversion output
    train2017/                     4,000 copied training images
    val2017/                       870 copied validation images
    annotations/train.json         Single-category COCO
    annotations/val.json
    audit.json
    anomaly_report.json
    class_counts.json
    train_manifest.json
    val_manifest.json
    split_manifest.json
    conversion.json                Completion marker, raw hash verification
  weights/pretrained/               Official weights + provenance sidecar
  environment/                      Runtime dependency audit
  runs/smoke-v1/                     Short train/resume check
  runs/baseline-v1/                  Checkpoints, run.json, logs, TensorBoard
  reports/baseline-v1/               Predictions, operating point, final report
  predictions/                      Inference outputs
```

Only future GPU setup in notebooks 02/03 downloads source code and dependencies. The source is checked out at an
immutable commit under `/content/YOLOX`. It is imported through `PYTHONPATH`, not
installed through upstream's optional ONNX dependencies. Restart the notebook
session if Colab says existing imports changed, then rerun path setup and audit.
Avoid importing NumPy or PyTorch before setup in a fresh session.

## Notebook 01: audit, preview, then confirmed conversion

Use the updated local `notebooks/01_prepare_data.ipynb`. Cell numbers count both
markdown and code cells. Run this exact order:

| Cell | Action | Expected output |
|---|---|---|
| 3 | Mount Drive, import lightweight audit code, set exact TEMPLE_ROOT/PROCESSED_ROOT | Mounted Drive and the two paths; no output folder created |
| 5 | Read-only audit and split reconstruction | Counts, pairing/dimensions/box/class checks, source train-list anomaly and dog report printed in memory |
| 7 | Read-only sample preview | Up to eight images: green Aquafina, orange competitors/background, red excluded dog |
| 9 | Leave CONFIRM_CONVERSION=False for read-only mode; set True only after review | False: no writes. True: converted train/val summaries, output path, raw_unchanged=True |
| 11 | Read completion marker | complete, train_images=4000, val_images=870, raw_sha256_before_after_equal=True and output listing |

Expected audit from the supplied inventory: 4,870 images, 4,870 XML, 4,873 TXT
files **across the whole raw tree**, zero XML parsing errors and zero invalid
Darknet lines. Source class counts: Aquafina 5,227, Deer 4,326, Kirkland 3,552,
Nestle 3,735. One XML dog is flagged and excluded; its raw XML is unchanged.
Additional pairing, class/geometry, decoding and cross-split content checks must
also pass. These extra checks have not yet been run against the actual Drive data.

The original train list should report 13,740 entries, 4,870 unique IDs, 8,870
duplicate entries, 4,870 IDs occurring more than once, and overlap with all 870
validation IDs. It is inspected only for reporting. Reconstruction preserves the
unique `val.txt` IDs and discovers IDs from `JPEGImages`; it asserts exactly
4,000 training IDs, 870 validation IDs, and zero overlap. No random repartitioning
or fake test set occurs. Notebook 03 refuses final test evaluation without an
independent test file.

Read [the full conversion policy](temple_dataset.md) before confirmation. Audit and
preview do not create directories, reports, caches or install dependencies from
the notebook. Mounting Drive establishes access; the first explicit output write
is inside confirmed conversion. Raw files are hashed during audit and rechecked
before and after conversion. This is IO-intensive and can take minutes on Drive.

Known dog/train-list anomalies are warnings. Unexplained class/box mismatches,
invalid pairs/dimensions, wrong counts and cross-split duplicate image contents
block conversion. Inspect the printed anomaly report; no partial conversion is
started for a failed audit. A failed conversion after writing begins has no final
completion marker. Use a fresh version path on retry rather than overwriting files.

**Stop after notebook 01 for the current task.** The following sections document
future training/evaluation and do not authorize starting them now.

## Future smoke and baseline

After confirmed preparation, notebooks 02/03 point PREPARED to the same
`/content/drive/MyDrive/aquafina-yolo/processed/temple`. Dataset downloading
is never automatic. Notebook 02 first tests synthetic negative and mixed batches
through the real loader, Mosaic and CUDA backward path. It then offers the explicit
official-weight download to Drive; the SHA256 and URL are recorded in a sidecar.

The weight adaptation check expects only class prediction layer shape mismatches.
Run the two-epoch smoke workflow: stop after epoch 1, inspect checkpoint fields,
resume to epoch 2. Run baseline only after that passes. Baseline uses 100 epochs,
640×640, FP16 and batch 8, official optimizer/LR schedule, seed 42, no MixUp/flip,
and no Mosaic in the last 15 epochs. Short smoke runs use no warmup and one final
no-Mosaic epoch to keep the short schedule valid.

On OOM, start a new run with batch 4, then 2. The official per-image learning rate
scales automatically. Do not change batch size mid-resume. A missing checkpoint or
Drive disconnection fails visibly; there is no silent fallback to local weights.

Prepared data can be copied to `/content/aquafina-data` for faster IO; set DATA to
that temporary path. Raw/prepared originals remain in Drive. Run/checkpoint paths
must remain under `/content/drive/MyDrive`. No local desktop training is needed.

## Resume

`latest_ckpt.pth` is saved at every completed epoch. `best_ckpt.pth` contains the
best validation AP checkpoint (the first checkpoint is retained even when AP=0).
The inference `model` is EMA; `training_model` stores the corresponding raw model.
Full-state resume restores optimizer, scaler, EMA update count and RNG states.
Worker-prefetch/sampler positions are not preserved, so resume is not a claim of
bitwise reproducibility. Partial epochs are replayed.

After reconnect, remount Drive, restore code/source/dependencies and set the same
DATA, RUN and BATCH before running the resume cell. Annotation hashes, project code and schedule
must match. Check that checkpoint files and `run.json` exist in Drive first.
Writing uses temporary files then replacement, but Drive disconnections can still
interrupt uploads; verify checkpoint readability before relying on persistence.

## Evaluate and infer

Notebook 03 generates COCO predictions with score floor 0.001 and NMS IoU 0.65.
COCO AP is measured on these low-score predictions, not thresholded display boxes.
Operating metrics use greedy one-to-one matching at IoU 0.5. Competitor image FPR
is the fraction of competitor-only images with at least one prediction; it is not
the number of bad boxes divided by all boxes. Rates are fractions, not percentages.

Validation threshold selection maximizes recall under precision ≥0.95 and
competitor image FPR ≤0.05, breaking ties by precision, lower FPR, then higher
threshold. No detections or no competitor samples cannot qualify. If selection
fails, AP remains available but no usable operating point is written. Review data
and validation errors; collect fresh training negatives, preserving held-out splits.

Enable final test only once model and threshold are frozen. Prediction metadata
binds the operating point to checkpoint hash, input size, NMS, score floor and
upstream revision. Test cannot select a threshold. Do not repeatedly use test
results to tune the model. Reports refuse to overwrite prior report files.

TempleRAIL conversion preserves train/val only. Original validation remains the
validation/threshold-selection set; it cannot also be called an independent test.

For media inference, pass the operating point with the matching checkpoint. CLI
examples, after setup:

```text
python -m aquafina_detector.predict --checkpoint DRIVE_RUN/best_ckpt.pth --input DRIVE_INPUT/image.jpg --output DRIVE_OUTPUT/new-run --operating-point DRIVE_REPORT/operating_point.json
python -m aquafina_detector.predict --checkpoint DRIVE_RUN/best_ckpt.pth --input DRIVE_INPUT/video.mp4 --output DRIVE_OUTPUT/new-video --operating-point DRIVE_REPORT/operating_point.json
```

Replace the uppercase example paths with actual absolute Drive paths. Output
boxes use `[x1,y1,x2,y2]`, class ID 0 and a confidence score. COCO results instead
use `[x,y,width,height]` and category ID 1. Video results include empty-detection
frames, zero-based indices and nominal frame/FPS timestamps; annotated MP4 omits
audio. Variable-frame-rate timing is not preserved exactly.

Clear notebook outputs before returning files to the local repository. Run
`python scripts/check_repository.py` before requesting any commit approval.
