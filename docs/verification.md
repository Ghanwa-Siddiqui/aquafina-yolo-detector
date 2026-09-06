# Verification record

Local environment: Windows, Python 3.12.10. NumPy, Pillow, OpenCV and pytest are
available. PyTorch/CUDA are not installed in the local project environment.

Completed locally:

- 72 tests passed with the isolated pycocotools dependency; two CUDA tests skipped.
- Archive fixtures verify a single sequential source read, MD5 mismatch rejection,
  safe local extraction, unchanged original inputs, complete-stage reuse and
  rejection of modified stages. Cache tests verify reuse without a full audit,
  conversion from cached state, stale/corrupt/blocked cache rejection, explicit
  forced re-audit, interrupted-audit handling and the 250-image progress interval.
- TempleRAIL fixtures cover corrupt train-list reconstruction (including the exact
  4,000/870 count invariant), original validation preservation, dog exclusions,
  class/coordinate mapping, hard negatives, corrupt input rejection, confirmation,
  stale audits and raw-file immutability. Nine synthetic cross-split groups resolve
  with unchanged validation membership; the full 4,870-ID fixture yields 3,991/870.
  Conflicting XML/Darknet annotations, dog differences, ambiguous pairing, multiple
  validation representatives and unexpected counts block conversion. Old cache
  schemas/policy keys are rejected; tampered completion counts are rejected. Incomplete
  conversions are rejected, and a train/val-only conversion passes the verifier.
- Notebook 01's setup/stage/audit-cache/preview order and conversion gate are tested.
  All 26 source/notebook files pass syntax and clean-output validation.
  This policy was tested locally only; no real Drive conversion or training ran.
- The actual COCO evaluator returns AP=1 for perfect boxes and AP=0 for disjoint
  boxes. Annotation ID 0 is rejected because COCO uses it as an unmatched sentinel.

- COCO validation rejects wrong class IDs, malformed/out-of-bounds boxes, missing
  group metadata, path traversal, annotation inconsistencies and crowd targets.
- Preparation preserves negatives and decoded image dimensions; exact duplicate
  groups never leak between splits and output versions cannot be overwritten.
- Metrics count duplicate detections as false positives and distinguish box counts
  from negative-image FPR. Validation-only selection rejects empty predictions,
  missing competitor evidence and unsatisfied targets.
- Box restoration handles YOLOX's top-left letterboxing and clipping.
- Saved-video IO produces a result for every frame, including no-detection frames.
- Notebook structure, cleared outputs, Python syntax and prohibited imports are
  checked by `scripts/check_repository.py`.

Pending in Colab (not claimed as passed):

- Run notebook 01 against the actual TempleRAIL Drive archive. MD5 verification and
  elapsed performance on that archive have not been measured locally. The user-provided
  inventory is encoded as the expected baseline, but the new pairing/geometry and
  content-hash audit has only run on local fixtures. No real source files were
  accessed or converted locally.

1. Fresh pinned installation and actual CUDA imports/model construction.
2. Actual upstream negative/mixed loading, Mosaic, class mapping, finite FP16
   loss and backward passes (`pytest -m gpu`).
3. Official pretrained head adaptation and the two-epoch train/resume smoke run.
4. Drive remount/reconnect and checkpoint readability.
5. Real baseline training, validation operating point, held-out test and real media
   predictions. No Aquafina recall, precision, AP or rejection claim exists yet.

Synthetic fixtures are software tests only; they do not establish model quality.
