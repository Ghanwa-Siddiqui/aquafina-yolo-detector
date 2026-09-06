# TempleRAIL source and conversion policy

## Source and expectations

Raw root: `/content/drive/MyDrive/aquafina-yolo/raw/temple/extracted/detection_dataset`.
Processed root: `/content/drive/MyDrive/aquafina-yolo/processed/temple`.
The audit uses Pillow and standard-library XML parsing, never a YOLO framework.
It does not download anything or modify/save any file. Preview images stay in RAM.

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

Exact duplicate images across these fixed splits block conversion rather than
moving validation IDs silently. Identical image content within a split shares a
hash-based group_id. Capture sessions and scene metadata were not supplied:
`scene=temple_unspecified`; hash groups do not prove capture-session independence.
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
Reports show original counts and counts after dog exclusion separately; converted
Aquafina count is 5,227 only if the dog exception does not remove a class-0 box.

## Hard negatives and visible-brand policy

Keep every image in its reconstructed split. Images containing Deer/Kirkland/Nestle
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
`CONFIRM_CONVERSION=True` after audit/preview in notebook 01. Recheck raw hashes
before writing. Refuse existing destinations, destinations inside raw, or an
ancestor of raw; resolve paths to prevent symlink redirects. Copy images (never
symlink them) to train2017/val2017. Do not move or edit raw files.

Write `audit.json`, `anomaly_report.json`, `class_counts.json`, `train_manifest.json`
and `val_manifest.json`, plus `annotations/train.json`, `annotations/val.json` and
the pipeline-compatible `split_manifest.json`. Manifest entries retain source IDs,
canonical numeric IDs, filenames and image hashes. Original validation membership
is directly reviewable in `val_manifest.json`.

Validate the completed canonical dataset and compare all raw hashes again. Write
`conversion.json` **last**, including complete status, counts and the raw-file hash
snapshot. An absent marker means incomplete output; do not train from it. An audit
report's `writes_performed=False` describes the preceding read-only audit, not the
later confirmed conversion that persists that report.

No input annotations.json is needed. No test set is produced. The generic COCO
preparation API remains available for independent, manually annotated datasets.
