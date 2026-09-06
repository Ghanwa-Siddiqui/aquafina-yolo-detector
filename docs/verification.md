# Verification record

Local environment: Windows, Python 3.12.10. NumPy, Pillow, OpenCV and pytest are
available. PyTorch/CUDA are not installed in the local project environment.

Completed locally:

- 45 tests passed with the isolated pycocotools dependency; two CUDA tests skipped.
- TempleRAIL fixtures cover corrupt train-list reconstruction (including the exact
  4,000/870 count invariant), original validation preservation, dog exclusions,
  class/coordinate mapping, hard negatives, corrupt input rejection, confirmation,
  stale audits, cross-split duplicate content and raw-file immutability. Incomplete
  conversions are rejected, and a train/val-only conversion passes the verifier.
- Notebook 01's read-only setup/audit/preview order and confirmation gate are tested.
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

- Run notebook 01 against the actual TempleRAIL Drive dataset. The user-provided
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
