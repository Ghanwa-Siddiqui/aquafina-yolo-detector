"""Regenerate clean, reviewable notebooks using only the Python standard library."""
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(text).strip() + "\n"}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": textwrap.dedent(text).strip() + "\n"}


SETUP = [md("""
    ## Repository and Drive
    In Colab choose a GPU runtime with Python 3.10–3.12. Upload a ZIP containing
    this repository's code, notebooks, configs, requirements and tests, and extract
    it to `/content/aquafina-yolo-detector` using the Files pane. Do not include
    datasets, checkpoints, caches, or `.git`. This works without a commit or push.
    After an approved future push, cloning your own repository is also an option.
    The repository on your computer remains the authoritative code copy.
    """), code("""
    from pathlib import Path
    import os, sys, subprocess
    from google.colab import drive
    drive.mount('/content/drive')
    REPO = Path('/content/aquafina-yolo-detector')
    DRIVE = Path('/content/drive/MyDrive/aquafina-yolo')
    assert (REPO / 'pyproject.toml').is_file(), 'Extract repository code first'
    os.chdir(REPO)
    sys.path.insert(0, str(REPO / 'src'))
    os.environ['PYTHONPATH'] = str(REPO / 'src') + ':/content/YOLOX'
    PREPARED = DRIVE / 'processed/temple'
    """)]

GPU_SETUP = [md("""
    ## Install the pinned GPU environment
    This downloads official YOLOX **source** and Python dependencies only.
    If Colab reports that already-imported packages changed, restart the session,
    rerun the Repository and Drive cell, then continue with the audit cell.
    Do not import torch/numpy before this install cell in a fresh session.
    """), code("""
    subprocess.run([sys.executable, '-m', 'aquafina_detector.bootstrap', '--repo', str(REPO)], check=True)
    """), code("""
    sys.path.insert(0, '/content/YOLOX')
    import torch
    from aquafina_detector.bootstrap import audit_runtime
    from aquafina_detector.common import write_json
    audit = audit_runtime()
    assert torch.__version__.split('+')[0] == '2.5.1'
    assert torch.cuda.is_available(), 'Select a GPU runtime'
    write_json(DRIVE / 'environment/audit.json', audit)
    print(torch.cuda.get_device_name(0), torch.__version__)
    """)]

PREPARE = [md("""
    # 01 — Stage, audit, preview and convert TempleRAIL
    No dataset downloads, GPU setup, weights, training or Ultralytics. A standard
    Colab CPU session with its bundled Pillow is sufficient. Read one archive from
    Drive, verify MD5, extract to fast temporary disk and cache completed audits.
    Original Drive archive and extracted data are never modified or deleted.
    Staging/cache writes are automatic; dataset conversion still requires explicit
    confirmation in cell 13. No input annotations.json is needed.
    """), md("""
    ## 1. Repository and Drive
    Upload and extract this repository's code to `/content/aquafina-yolo-detector`.
    Include src, configs, scripts, requirements and pyproject.toml; no raw data or
    weights in the repository ZIP. Mount Drive below. Do not use Run all to approve
    conversion: review audit and previews first. Cell numbers include markdown.
    """), code("""
    import sys
    sys.dont_write_bytecode = True
    from pathlib import Path
    from google.colab import drive
    drive.mount('/content/drive')
    REPO = Path('/content/aquafina-yolo-detector')
    assert (REPO / 'pyproject.toml').is_file(), 'Extract repository code first'
    sys.path.insert(0, str(REPO / 'src'))
    ARCHIVE = Path('/content/drive/MyDrive/aquafina-yolo/raw/temple/detection_dataset.tar.gz')
    EXPECTED_MD5 = 'fca7260d4785af1dec18aa320fa9fc4a'
    STAGE_DIR = Path('/content/temple_stage')
    TEMPLE_ROOT = Path('/content/temple_stage/detection_dataset')
    CACHE_DIR = Path('/content/drive/MyDrive/aquafina-yolo/cache/temple_audit')
    PROCESSED_ROOT = Path('/content/drive/MyDrive/aquafina-yolo/processed/temple')
    from aquafina_detector.temple import preview_temple, convert_temple
    from aquafina_detector.temple_stage import stage_archive, audit_staged
    STAGE = None
    AUDIT = None
    PREVIEW_SHOWN = False
    CONVERSION_RESULT = None
    print('Original archive (read-only):', ARCHIVE)
    print('Temporary audit input:', TEMPLE_ROOT)
    print('Proposed output (not created):', PROCESSED_ROOT)
    """), md("""
    ## 2. Verify archive and stage on temporary disk
    Copy the single existing archive from Drive while calculating MD5 and SHA256.
    A mismatch stops before extraction. Extract locally, never by copying thousands
    of individual source files from Drive. Safe extraction rejects path traversal,
    links and special files. An existing complete stage is checked locally before
    reuse; interrupted/conflicting stages require a fresh runtime or stage path.
    No original archive or extracted Drive dataset is deleted or modified.
    """), code("""
    AUDIT = None
    PREVIEW_SHOWN = False
    CONVERSION_RESULT = None
    STAGE = None
    STAGE = stage_archive(ARCHIVE, EXPECTED_MD5, STAGE_DIR,
                          progress=lambda message: print(message, flush=True))
    assert STAGE.root == TEMPLE_ROOT
    print('Verified local input:', STAGE.root)
    """), md("""
    ## 3. Audit locally or reuse the completed verified cache
    Decode every image; check image/XML/TXT pairing, dimensions, boxes and classes.
    All per-file checks now use temporary disk. Progress prints after every 250
    images and on completion. Set REUSE_AUDIT_CACHE=False to force a fresh audit.
    A completed report is cached in Drive under the archive MD5; reuse requires
    matching archive hashes, audit-code signature, policy and staged-file hashes.
    Incomplete, corrupt, stale or blocked results never bypass a fresh audit.
    Ignore train.txt for membership: preserve unique val.txt IDs and use
    all JPEGImages IDs minus validation: initially 4,000 train / 870 val.
    The real audit reports 37 conflicting groups, including 9 crossing splits.
    Conflicts remain blocked: 4,000 train, zero excluded, 9 content overlaps.
    The 3,991/870 target applies only after a separately reviewed resolution.
    Known dog and corrupt-train-list warnings do not alone block conversion.
    Multiple validation representatives, ambiguous groups, conflicting complete XML
    or Darknet annotations, and unexpected counts still block conversion.
    The changed code/policy compatibility key invalidates older audit caches.
    """), code("""
    import json
    AUDIT = None
    PREVIEW_SHOWN = False
    CONVERSION_RESULT = None
    assert STAGE is not None, 'Run staging cell 5 first'
    REUSE_AUDIT_CACHE = True
    AUDIT = audit_staged(STAGE, CACHE_DIR, reuse=REUSE_AUDIT_CACHE,
                         progress=lambda message: print(message, flush=True))
    print(json.dumps(AUDIT.audit, indent=2))
    print(json.dumps(AUDIT.class_counts, indent=2))
    print(json.dumps(AUDIT.anomalies, indent=2))
    print('Completed audit cache:', CACHE_DIR / (STAGE.archive_md5 + '.json'))
    """), md("""
    ## 4. Read-only duplicate-conflict diagnostics
    Run even when the audit is blocked. Summarize cross-split, train-only and
    validation-only conflicting groups, then display EVERY cross-split conflict
    side-by-side (expected 9). Each panel overlays that file's Darknet boxes in cyan
    and XML boxes in magenta. Printed JSON includes original label lines, classes,
    coordinates, matched-box deltas, IoU and unmatched objects for every pair.
    Matching is a diagnostic heuristic; inspect ties and unmatched objects manually.
    Tiny differences are rounding candidates, not approval to relax the audit.
    Filename-only XML differences are already ignored by the conversion policy.
    Only the diagnostic JSON is saved under Drive diagnostics; no images, raw files,
    audit decisions or processed dataset are changed. Stop after diagnostics and
    optional preview for this investigation. Do not enable conversion.
    """), code("""
    from IPython.display import display
    from aquafina_detector.temple_diagnostics import (
        duplicate_conflict_report, render_duplicate_group, save_duplicate_conflict_report,
    )
    assert AUDIT is not None, 'Run audit cell 7 first'
    CONFLICT_REPORT = duplicate_conflict_report(AUDIT)
    print(json.dumps(CONFLICT_REPORT['summary'], indent=2))
    print('Total conflicting groups:', CONFLICT_REPORT['total_conflicting_groups'])
    for group in CONFLICT_REPORT['groups']:
        if group['scope'] == 'cross_split' and group['annotation_conflict']:
            print(json.dumps(group, indent=2))
            display(render_duplicate_group(AUDIT, group))
    REPORT_PATH = Path('/content/drive/MyDrive/aquafina-yolo/diagnostics/temple') / STAGE.archive_md5 / 'duplicate_conflict_report.json'
    save_duplicate_conflict_report(AUDIT, CONFLICT_REPORT, REPORT_PATH, PROCESSED_ROOT)
    print('Diagnostic report:', REPORT_PATH)
    print('Audit remains:', AUDIT.audit['status'], 'Blocking errors:', AUDIT.audit['blocking_errors'])
    """), md("""
    ## 5. Read-only preview from temporary disk
    Green: Aquafina boxes kept. Orange: competitor boxes become background.
    Red: XML dog box excluded. Samples prioritize the dog image and available
    positive/mixed/negative subsets. Inspect all reported anomalies, not just these
    samples. Source annotations do not prove visible brand identity in every image.
    """), code("""
    from IPython.display import display
    PREVIEW_SHOWN = False
    assert AUDIT is not None, 'Run audit cell 7 first'
    previews = preview_temple(AUDIT, limit=8)
    for caption, image in previews:
        print(caption)
        display(image)
    PREVIEW_SHOWN = bool(previews)
    print('Preview displayed; nothing saved. Blocking errors:', AUDIT.audit['blocking_errors'])
    """), md("""
    ## 6. Explicit confirmation before converted-dataset writes
    After reviewing audit and preview, change CONFIRM_CONVERSION to True and run
    cell 13. Keep it False to skip conversion. A clean audit and completed preview
    are required. Conversion writes exclusively under PROCESSED_ROOT, copies images
    (no raw symlinks), and checks raw hashes before/after. Existing output is never
    overwritten; use a new child/version path for another conversion.

    Aquafina Darknet class 0 becomes model class 0 / COCO category 1. Keep every
    image except the verified training duplicates listed in excluded_training_images; competitor-only images have empty annotations, and mixed images retain
    Aquafina boxes only. Dog objects are excluded. No test split is manufactured.
    """), code("""
    CONFIRM_CONVERSION = False
    if not CONFIRM_CONVERSION:
        print('Conversion not confirmed; no converted dataset written. Staging/audit cache are retained.')
    else:
        assert AUDIT is not None and PREVIEW_SHOWN, 'Run audit cell 7 and preview cell 11 first'
        assert AUDIT.audit['blocking_errors'] == 0, 'Resolve blocking audit errors before conversion'
        CONVERSION_RESULT = convert_temple(AUDIT, PROCESSED_ROOT, confirm=True)
        print(json.dumps(CONVERSION_RESULT, indent=2))
    """), md("""
    ## 7. Read back completed outputs
    audit.json, anomaly_report.json, class_counts.json, train_manifest.json,
    val_manifest.json, split_manifest.json and conversion.json accompany the
    annotations/train.json, annotations/val.json and train2017/val2017 images.
    The conversion marker is written last; its absence means an incomplete output.
    Stop here. Training is a separate future action, not part of this notebook.
    """), code("""
    if CONVERSION_RESULT is None:
        print('No conversion performed in this session.')
    else:
        from aquafina_detector.common import read_json
        marker = read_json(PROCESSED_ROOT / 'conversion.json')
        from aquafina_detector.data import verify_prepared
        verify_prepared(PROCESSED_ROOT)
        assert marker['train_images'] == 3991 and marker['val_images'] == 870
        assert marker['excluded_training_images'] == 9
        assert marker['cross_split_duplicate_content'] == 0
        assert marker['validation_ids_preserved'] and marker['raw_sha256_before_after_equal']
        print(marker | {'source_file_sha256': '(omitted from display)'})
        print('Files:', sorted(path.name for path in PROCESSED_ROOT.iterdir()))
    """)]

TRAIN = [md("""
    # 02 — Train YOLOX-S on Colab
    GPU work happens here. Weights, checkpoints, logs and run metadata persist in
    Google Drive. Run cells deliberately; full training and weight downloading
    have explicit switches. No Ultralytics package, CLI, trainer or weights.
    """), *SETUP, *GPU_SETUP, md("""
    ## Verify actual negative and mixed GPU batches
    Synthetic fixtures test loading, Mosaic, class mapping and finite backward
    passes. These are software tests, not a training dataset or accuracy evidence.
    """), code("""
    subprocess.run([sys.executable, '-m', 'pytest', '-q', '-m', 'gpu'], check=True)
    """), md("""
    ## Official pretrained weights — explicit download to Drive
    Set DOWNLOAD_WEIGHTS only when ready. Existing weights are reused with their
    provenance sidecar. Never use Ultralytics checkpoints or generic `yolo` commands.
    """), code("""
    from aquafina_detector.bootstrap import download_pretrained
    WEIGHTS = DRIVE / 'weights/pretrained/yolox_s.pth'
    DOWNLOAD_WEIGHTS = False
    if DOWNLOAD_WEIGHTS:
        download_pretrained(WEIGHTS)
    assert WEIGHTS.is_file(), 'Enable the explicit download once, then reuse the weights'
    """), md("""
    ## Dataset and checkpoint adaptation check
    DATA defaults to prepared files on Drive. For faster reads, optionally copy
    PREPARED to `/content/aquafina-data` using shutil.copytree and set DATA there.
    Only temporary Colab storage may hold this copy; checkpoints still use Drive.
    """), code("""
    DATA = PREPARED
    from yolox.utils import load_ckpt
    from aquafina_detector.experiment import AquafinaExp
    original = torch.load(WEIGHTS, map_location='cpu', weights_only=False)['model']
    model = AquafinaExp().get_model()
    mismatched = [k for k, v in model.state_dict().items() if k in original and v.shape != original[k].shape]
    assert mismatched and all('cls_preds' in k for k in mismatched), mismatched
    load_ckpt(model, original)
    assert model.head.num_classes == 1
    del model, original
    print('One-class adaptation verified:', mismatched)
    """), md("""
    ## One-epoch smoke run, then resume the second epoch
    Set RUN_SMOKE=True after preparing data. The two commands share a two-epoch
    schedule; stopping after epoch one exercises a real checkpoint resume.
    Use a new SMOKE name for a repeat. Check logs for finite losses.
    """), code("""
    RUN_SMOKE = False
    SMOKE = DRIVE / 'runs/smoke-v1'
    if RUN_SMOKE:
        command = [sys.executable, '-m', 'aquafina_detector.train', '--data', str(DATA),
                   '--run', str(SMOKE), '--batch-size', '2', '--epochs', '2']
        subprocess.run(command + ['--checkpoint', str(WEIGHTS), '--stop-after-epochs', '1'], check=True)
        saved = torch.load(SMOKE / 'latest_ckpt.pth', map_location='cpu', weights_only=False)
        assert saved['start_epoch'] == 1 and 'scaler' in saved and 'training_model' in saved
        del saved
        subprocess.run(command + ['--checkpoint', str(SMOKE / 'latest_ckpt.pth'), '--resume'], check=True)
        saved = torch.load(SMOKE / 'latest_ckpt.pth', map_location='cpu', weights_only=False)
        assert saved['start_epoch'] == 2
        del saved
        print('Checkpoint save and resume passed')
    """), md("""
    ## Baseline
    Set RUN_BASELINE=True after smoke verification. Start at batch 8; if CUDA runs
    out of memory, use a NEW run name and batch 4, then 2. Learning rate scales
    with batch size. Do not change batch/data/schedule while resuming.
    """), code("""
    RUN_BASELINE = False
    RUN = DRIVE / 'runs/baseline-v1'
    BATCH = 8
    if RUN_BASELINE:
        subprocess.run([sys.executable, '-m', 'aquafina_detector.train', '--data', str(DATA),
                        '--run', str(RUN), '--checkpoint', str(WEIGHTS), '--batch-size', str(BATCH)], check=True)
    """), md("""
    ## Resume after disconnection
    Remount Drive, restore repository/source and environment, then enable this
    cell. Use the same RUN, BATCH and DATA as the original baseline. Checkpoints
    are saved each epoch; work since the last completed epoch is lost.
    """), code("""
    RESUME_BASELINE = False
    if RESUME_BASELINE:
        subprocess.run([sys.executable, '-m', 'aquafina_detector.train', '--data', str(DATA),
                        '--run', str(RUN), '--checkpoint', str(RUN / 'latest_ckpt.pth'),
                        '--batch-size', str(BATCH), '--resume'], check=True)
    """)]

EVALUATE = [md("""
    # 03 — Evaluate rejection and predict
    Choose the best validation checkpoint. Generate low-confidence predictions
    for COCO AP, then select a threshold using validation only. The test cells
    remain disabled until you freeze the model and operating point.
    """), *SETUP, *GPU_SETUP, code("""
    from aquafina_detector.predict import Predictor, predict_coco, predict_media
    from aquafina_detector.evaluate import evaluate_files
    from aquafina_detector.common import read_json
    CHECKPOINT = DRIVE / 'runs/baseline-v1/best_ckpt.pth'
    REPORTS = DRIVE / 'reports/baseline-v1'
    REPORTS.mkdir(parents=True, exist_ok=True)
    predictor = Predictor(CHECKPOINT, confidence=0.001, nms=0.65)
    val_predictions = REPORTS / 'val_predictions.json'
    predict_coco(predictor, PREPARED / 'annotations/val.json', PREPARED / 'val2017', val_predictions)
    operating_point = REPORTS / 'operating_point.json'
    result = evaluate_files(PREPARED / 'annotations/val.json', val_predictions, operating_point)
    print(result)
    """), md("""
    If status is `no_qualifying_threshold`, do not claim successful brand rejection.
    Review false positives on validation and collect additional independent training
    examples. Never copy test images into training. Reports include counts, AP,
    image-level false positive rates and separate subset results. Precision/FPR
    constraints are provisional targets, not guarantees; inspect sample size.
    """), code("""
    from PIL import Image, ImageDraw
    from IPython.display import display
    report = result.get('operating_report') or result.get('diagnostic_report')
    if report:
        for error in report['errors'][:12]:
            print(error)
            display(Image.open(PREPARED / 'val2017' / error['file_name']))
    else:
        print('No report available.')
    """), md("""
    ## Held-out test — explicit final evaluation
    Enable only after model/threshold freeze. The evaluator checks checkpoint,
    preprocessing and NMS provenance. It never searches thresholds on test.
    """), code("""
    RUN_FINAL_TEST = False
    if RUN_FINAL_TEST:
        assert (PREPARED / 'annotations/test.json').is_file(), 'TempleRAIL has train/val only; supply an independent test set first'
        assert read_json(operating_point)['status'] == 'qualified'
        test_predictions = REPORTS / 'test_predictions.json'
        predict_coco(predictor, PREPARED / 'annotations/test.json', PREPARED / 'test2017', test_predictions)
        print(evaluate_files(PREPARED / 'annotations/test.json', test_predictions,
                             REPORTS / 'test_report.json', operating_point))
    """), md("""
    ## Image or saved-video inference
    Set INPUT to your Drive file and RUN_PREDICTION=True. Outputs include annotated
    media and original-coordinate detections; video has frame indices/timestamps.
    Output videos omit audio. Use a new output directory for each invocation.
    """), code("""
    RUN_PREDICTION = False
    INPUT = DRIVE / 'inference/example.jpg'
    if RUN_PREDICTION:
        subprocess.run([sys.executable, '-m', 'aquafina_detector.predict', '--checkpoint', str(CHECKPOINT),
                        '--input', str(INPUT), '--output', str(DRIVE / 'predictions/example-v1'),
                        '--operating-point', str(operating_point)], check=True)
    """)]


def main():
    directory = ROOT / "notebooks"
    directory.mkdir(exist_ok=True)
    for name, cells in [("01_prepare_data", PREPARE), ("02_train_colab", TRAIN), ("03_evaluate_predict", EVALUATE)]:
        for i, cell in enumerate(cells):
            cell["id"] = f"cell-{i:03d}"
        notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}}, "cells": cells}
        (directory / f"{name}.ipynb").write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
