# TempleRAIL source and conversion policy

## Source and expectations

Original archive: `/content/drive/MyDrive/aquafina-yolo/raw/temple/detection_dataset.tar.gz`.
Expected MD5: `fca7260d4785af1dec18aa320fa9fc4a`.
The existing Drive extraction under `raw/temple/extracted/detection_dataset` is
left untouched. Notebook audit input is `/content/temple_stage/detection_dataset`.
Processed root: `/content/drive/MyDrive/aquafina-yolo/processed/temple`.
The audit uses Pillow and standard-library XML parsing, never a YOLO framework.
The core audit does not download or write files. Its staging/cache wrapper copies
one archive to temporary disk and saves completed audit state to Drive. Preview
images stay in RAM. These writes do not modify the original raw archive or dataset.

## Staging and reusable audits

`temple_stage.stage_archive` reads the source archive once sequentially, copying
to `/content/temple_stage` while calculating MD5 and SHA256. A mismatched MD5 stops
before extraction. Extract only regular files/directories rooted at detection_dataset;
reject traversal, absolute paths, links, devices and duplicate file entries. Hash
each extracted file as it is written. No per-file copying from the Drive extraction
occurs. A completed local stage can be reused after archive/local hash checks.
Temporary disk needs room for both the compressed archive and extracted files.

`temple_stage.audit_staged` caches the full completed report, records, anomalies,
class counts, split IDs and file hashes in one atomic JSON envelope at
`/content/drive/MyDrive/aquafina-yolo/cache/temple_audit/<archive-md5>.json`.
Reuse requires matching archive MD5/SHA256, source-code/policy signature, payload
checksum and staged-file hashes, plus a successful audit with no blocking errors.
These are integrity checks, not authentication of externally supplied cache files.
Do not import caches from untrusted sources. Blocked reports can be inspected but
are not considered reusable verified audits. Incomplete/stale/corrupt cache files
cause a fresh local audit. Interrupted audits never publish a completed cache.

Set REUSE_AUDIT_CACHE=False in notebook 01 to force a fresh audit. A new Colab
runtime must restage/verify the archive, but a compatible Drive cache still avoids
image decoding and XML/Darknet checks. Fresh audits report every 250 images and
the final image count; all per-file integrity checks run against local temporary
storage. Code/policy changes intentionally invalidate prior cached decisions.

The default audit enforces the user's verified image/XML/TXT and Darknet instance
counts. TXT counts include the whole source tree, not just label files. Image and
paired label/XML identifiers are filename stems, unique across their respective
directories. Split tokens may be bare stems or image paths, with an optional VOC
membership column. Leading zeroes in IDs are preserved.

## Split reconstruction

`train.txt` is never a source of membership. Its entries, unique/duplicate counts,
unknown IDs and overlap are recorded as an anomaly. Preserve the 870 unique IDs
in `val.txt`; reject validation IDs missing from JPEGImages. Training is all 4,870
discovered image IDs minus those validation IDs. Enforce 4,000/870 and zero overlap.
All original validation IDs remain validation; no extra test split is manufactured.

Apply policy `validation-priority-paired-annotations-v1` after reconstruction.
For each exact SHA-256 group crossing splits, require one validation representative
and verified image/XML/Darknet pairing for every member. Compare complete XML and
complete Darknet annotations separately, including competitors and dog exceptions.
Ignore object order, line numbers, filename and class-name casing; require equal
parsed coordinates and dimensions across copies. The 2-pixel XML/Darknet pairing
tolerance does not excuse cross-copy annotation differences. Existing maximum
1e-6 normalized boundary clamps remain allowed before this comparison.

Retain the validation image and exclude all equivalent training members. The real
audit expects 9 crossing groups and 9 exclusions, leaving **3,991 train / 870 val**
and zero exact cross-split content overlap. Record each excluded `image_id`,
`retained_validation_id`, `sha256` and `reason` in anomaly_report.json's
`excluded_training_images`. No raw files are changed or deleted.
Multiple validation representatives (even without training members), unverified
pairs, conflicting annotations in any duplicate group, and unexpected counts
remain blocking. Equivalent training-only groups remain in training and share a
hash-based group_id. Hash groups do not prove capture-session independence.
The source-code and policy compatibility signature invalidates old cached audits.
Near-duplicate or scene leakage remains a manual review responsibility.

## Classes, coordinates and dog exception

| Darknet ID | Source class | Canonical handling |
|---|---|---|
| 0 | Aquafina | Model class 0; single COCO category ID 1, aquafina_bottle |
| 1 | Deer | No positive annotations; background |
| 2 | Kirkland | No positive annotations; background |
| 3 | Nestle | No positive annotations; background |

Decode images and compare XML width/height with actual dimensions. Check finite,
nonempty in-bounds XML boxes and normalized Darknet center/size and corner ranges.
Match known XML classes (case-insensitive) one-to-one against same-class Darknet
boxes within 2 pixels on every edge. This tolerance covers typical VOC integer
origin and normalized rounding differences; it is not a license for arbitrary
geometry changes. Unknown classes and unexplained differences block conversion.

Darknet normalized boxes are the conversion coordinate source. Multiply by decoded
width/height, then emit original-pixel COCO [x,y,width,height]. Only tiny boundary
rounding excursions up to 1e-6 normalized are clamped in processed coordinates;
each such occurrence is explicitly reported. Larger excursions block conversion.
Raw XML and TXT coordinates are never rewritten or normalized in place.

Every XML object named `dog` is a reported exception and never a positive. If a
Darknet box has no same-class XML match but matches that dog's geometry, exclude it
too and report its source class/line. This prevents an anomalous class-0 box from
becoming Aquafina. A dog with no Darknet counterpart is simply reported/excluded.
Known class matches take priority, so a nearby valid Aquafina object is retained.
Reports distinguish original counts, counts after dog exclusion, and counts after
duplicate exclusion. Converted Aquafina totals are computed from retained images.

## Hard negatives and visible-brand policy

Keep every image in its reconstructed split except verified training duplicates. Images containing Deer/Kirkland/Nestle
but no retained Aquafina get `subset=competitor`, with **empty annotations**. Mixed
images get `mixed_brand` and only Aquafina boxes. Aquafina-only images get
`visible_brand`. Remaining images are negatives. No competitor class is added to
the model, and no positive image is dropped merely because it also contains other
brands. These hard negatives teach suppression of other bottles.

The source labels supply brand identity; the converter does not perform OCR or
prove visibility. Each image records `visibility_review=source_labels_not_independently_verified`.
Review previews and more source images for the project's visible-evidence policy
before training. The automatic `visible_brand` subset means Aquafina-only source
labels here, not a completed independent visual inspection.

## Writes and manifests

Conversion requires explicit `confirm=True` at the Python API and
`CONFIRM_CONVERSION=True` after audit/preview in notebook 01. Recheck staged-input hashes
before writing. Refuse existing destinations, destinations inside raw, or an
ancestor of raw; resolve paths to prevent symlink redirects. Copy images (never
symlink them) to train2017/val2017. Do not move or edit raw files.

Write `audit.json`, `anomaly_report.json`, `class_counts.json`, `train_manifest.json`
and `val_manifest.json`, plus `annotations/train.json`, `annotations/val.json` and
the pipeline-compatible `split_manifest.json`. Manifest entries retain source IDs,
canonical numeric IDs, filenames and image hashes. Original validation membership
is directly reviewable in `val_manifest.json`.

Validate the completed canonical dataset and compare all staged-input hashes again. Write
`conversion.json` **last**, including complete status, counts and the raw-file hash
snapshot. An absent marker means incomplete output; do not train from it. An audit
report's `writes_performed=False` describes the core read-only audit, not the
wrapper's automatic cache save or the later confirmed conversion. Archive provenance
is retained in audit.json. The original Drive extraction is not read again for
these checks and is never modified or deleted.

No input annotations.json is needed. No test set is produced. The generic COCO
preparation API remains available for independent, manually annotated datasets.
