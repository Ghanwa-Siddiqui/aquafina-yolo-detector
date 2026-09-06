# Colab workflow

## Prepare code and storage

Notebook 01 needs only a standard Colab **CPU** session and its bundled Pillow.
It does not install packages. Archive staging and audit caching are automatic;
converted-dataset writes still require confirmation after audit/preview review.
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
  raw/temple/detection_dataset.tar.gz       Read-only source archive
  raw/temple/extracted/detection_dataset/   Existing extraction, left untouched
    JPEGImages/                     4,870 images
    Annotations/                    4,870 XML annotations
    Labels/                         Paired Darknet TXT labels
    ImageSets/Main/                 Original train.txt and val.txt
    Readme.txt
  cache/temple_audit/<archive-md5>.json     Completed audit state, automatically saved
  processed/temple/                 Confirmed conversion output
    train2017/                     3,991 copied training images
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

## Notebook 01: stage locally, audit/cache, preview, then confirmed conversion

Use the updated local `notebooks/01_prepare_data.ipynb`. Cell numbers count both
markdown and code cells. Run this exact order:

| Cell | Action | Expected output |
|---|---|---|
| 3 | Mount Drive and set archive, stage, cache and output paths | Mounted Drive and proposed paths |
| 5 | Sequentially copy/hash one archive, verify MD5, safely extract locally | MD5 verified, extraction progress, local input at /content/temple_stage/detection_dataset |
| 7 | Reuse eligible audit cache or perform full local audit | Cache-hit message OR progress every 250 images, counts, anomalies and cache-save path |
| 9 | Duplicate-conflict diagnostics and report export | Expected 37 conflicting groups, 9 cross-split; all 9 side-by-side panels plus full comparisons and report path |
| 11 | Read-only local preview | Up to eight images: green Aquafina, orange competitors/background, red excluded dog |
| 13 | Leave CONFIRM_CONVERSION=False to skip conversion; set True only after review | False: staging/cache retained, no converted dataset. True: converted summaries and raw_unchanged=True |
| 15 | Read completion marker (future conversion only) | complete, train_images=3991, val_images=870, excluded_training_images=9, cross_split_duplicate_content=0, raw_sha256_before_after_equal=True and output listing |

The archive must be exactly:
`/content/drive/MyDrive/aquafina-yolo/raw/temple/detection_dataset.tar.gz`.
Verify MD5 `fca7260d4785af1dec18aa320fa9fc4a` before extracting. A mismatch stops
without extraction and prints the actual hash. Do not disable this check.

Temporary files live under `/content/temple_stage`; audit input is
`/content/temple_stage/detection_dataset`. The single source archive is read once
per staging invocation, computing MD5 and SHA256 while copying. No thousands-file
copy from the Drive extraction occurs. Local files are hashed while extracting.
An existing completed local extraction is checked before reuse. A fresh Colab
runtime restages from the same archive; it can still reuse the Drive audit cache.

Cache file:
`/content/drive/MyDrive/aquafina-yolo/cache/temple_audit/fca7260d4785af1dec18aa320fa9fc4a.json`.
Set `REUSE_AUDIT_CACHE=True` (default) in cell 7. Reuse requires the same verified
MD5 **and** SHA256, matching audit source-code/expected-counts signature, intact
cache payload, matching local file hashes and zero blocking audit errors. It skips
image decoding and XML/label cross-checking, not archive verification or local
integrity checks. Set False to force a fresh audit. Completed blocked reports are
cached for inspection but are never reused as verified audits. Interrupted audits
do not create a completed cache; corrupt or stale cache entries trigger a fresh audit.

Fresh-audit progress includes `Audited 250/4,870 images`, `500/4,870`, and so on
through `4,750/4,870`, followed by `4,870/4,870`. Archive copy/extraction also prints
progress. The actual Drive archive has not been staged or timed from local Codex;
no runtime speed guarantee is implied by the fixture tests.

Expected audit from the supplied inventory: 4,870 images, 4,870 XML, 4,873 TXT
files **across the whole raw tree**, zero XML parsing errors and zero invalid
Darknet lines. Source class counts: Aquafina 5,227, Deer 4,326, Kirkland 3,552,
Nestle 3,735. One XML dog is flagged and excluded; its raw XML is unchanged.
Additional pairing, class/geometry, decoding and cross-split content checks must
also pass. The user reports a completed real audit with 9 crossing duplicate
groups; this updated resolution policy has only been tested on local fixtures.

The original train list should report 13,740 entries, 4,870 unique IDs, 8,870
duplicate entries, 4,870 IDs occurring more than once, and overlap with all 870
validation IDs. It is inspected only for reporting. Reconstruction preserves the
unique `val.txt` IDs and discovers IDs from `JPEGImages`; it asserts exactly
4,000 training IDs, 870 validation IDs, and zero ID overlap. The completed real
audit found 9 crossing SHA-256 duplicate groups. Verify paired XML and Darknet
annotations, retain each unique validation representative, and exclude its training
copy only if both complete annotation sets are equivalent. The real audit found
37 conflicting groups (9 cross-split), so this condition is not met. Conversion
remains blocked: processed_train_count=4000, excluded_training_count=0,
cross_split_duplicate_content=9, blocking_errors=40. The conditional target is
3,991 processed train images and all 870 validation images, with zero
cross-split duplicate content. Each exclusion and its retained counterpart, hash
and reason appear in anomaly_report.json. No random repartitioning
or fake test set occurs. Notebook 03 refuses final test evaluation without an
independent test file.

Read [the full conversion policy](temple_dataset.md) before confirmation. The audit
itself reads staged files only; its wrapper saves completed audit state to Drive.
Preview images stay in memory. Staging/cache writes precede confirmation, while
permanent converted output remains gated. The original archive and original Drive
extraction are never modified or deleted. Conversion rechecks its **local staged
input** hashes before/after; it does not walk the original Drive extraction.
Writing and verifying the permanent converted images still uses Drive and may be
slow; this optimization removes per-file Drive reads from the audit path.

Interrupted extraction is never reused as complete. Restart the runtime or choose
a fresh temporary stage path (and update TEMPLE_ROOT accordingly) if a conflict is
reported. No automatic cleanup deletes original inputs or conflicting stages.

Known dog/train-list anomalies are warnings. Unexplained class/box mismatches,
invalid pairs/dimensions, wrong counts, multiple validation representatives,
ambiguous groups and conflicting duplicate annotations block conversion. Only verified equivalent groups can become resolvable warnings. The current real
conflicts are not equivalent and must remain blocking.
The changed code/policy compatibility key invalidates the old audit cache; upload
the updated repository, restart the Colab session to clear imported old code, then
run cells 3, 5, 7 and 9 again (diagnostics now occupies cell 9). Cell 13 still defaults to CONFIRM_CONVERSION=False. Inspect the printed anomaly report; no partial conversion is
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

## Duplicate-conflict investigation (no conversion)

Upload the updated repository and restart Colab to avoid old imported modules.
Run **3 -> 5 -> 7 -> 9**, then optionally **11** for the general preview. Stop there;
leave cell 13 unconfirmed. Cell 15 only checks a future completed conversion.
Audit cell 7 intentionally completes even with blocking errors so diagnostics run.

Cell 9 writes only:
`/content/drive/MyDrive/aquafina-yolo/diagnostics/temple/fca7260d4785af1dec18aa320fa9fc4a/duplicate_conflict_report.json`.
Re-running it replaces this diagnostic JSON; it never creates processed output.
All image panels remain in memory. The original archive and extracted data stay
unchanged. Report construction and display never mutate the audit or split lists.

The report separates cross-split, train-only and validation-only annotation
conflicts, examining every exact-image group, including groups blocked for multiple
validation representatives. The real findings predict 37 total and 9 cross-split;
the remaining 28 are classified from actual membership, not assumed train-only.
All cross-split conflicting groups are displayed without a sample limit. Cyan is
that file's original Darknet geometry; magenta is its XML geometry. Numbered boxes
link to the printed JSON with image ID, reconstructed split, classes, original
Darknet lines, normalized center/size and corners, XML pixel boxes, signed edge
differences and IoU. Darknet diagnostics retain original unclamped coordinates.

Comparisons cover XML versus Darknet within each file and both formats separately
across every pair of duplicate files. Geometry-first matching uses highest IoU,
then lowest coordinate distance, then index order, with ties and unmatched boxes
reported. Matches, especially disjoint or tied ones, are inspection aids rather
than proof of object identity. Different class labels, missing objects and changed
boxes may coexist. Differences of at most 1e-6 normalized are labeled rounding
candidates; human review is required to determine harmlessness. No safety tolerance
is relaxed. Filename-only differences are explicitly identified and already ignored
by audit equivalence, so they cannot by themselves explain a blocking conflict.
