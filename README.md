# Aquafina bottle detector

Single-class **YOLOX-S** using the official Megvii implementation. Detects bottles
with identifiable Aquafina branding and learns to reject competitor/ambiguous
bottles from negative examples. This is a training pipeline, **not a trained model**.
There are no datasets or weights in this repository and no measured accuracy yet.

## Start here

1. Read [the TempleRAIL policy](docs/temple_dataset.md).
2. Keep the extracted raw dataset at
   `/content/drive/MyDrive/aquafina-yolo/raw/temple/extracted/detection_dataset`.
3. Open [notebook 01](notebooks/01_prepare_data.ipynb) on Colab CPU. Run cells
   **3 → 5 → 7** for setup, read-only audit and preview (counting markdown cells).
4. Review the results, then set `CONFIRM_CONVERSION = True` in **cell 9** only when
   ready to write processed data. Cell **11** reads back the completion marker.
5. Stop there for this dataset-support stage. [Notebook 02](notebooks/02_train_colab.ipynb)
   is for future training; [notebook 03](notebooks/03_evaluate_predict.ipynb) is for
   future evaluation and inference. See [the Colab workflow](docs/colab_workflow.md).

Notebook 01 reads `JPEGImages`, `Annotations` (XML), `Labels` (Darknet TXT), and
`ImageSets/Main`. It no longer requires an input `annotations.json`. Nothing under
the raw directory is modified, including the XML object named `dog`.

Conversion writes to `/content/drive/MyDrive/aquafina-yolo/processed/temple`:
`audit.json`, `anomaly_report.json`, `class_counts.json`, `train_manifest.json`,
`val_manifest.json`, canonical COCO annotations, copied images, a combined split
manifest, and a final `conversion.json` marker with raw-file hashes. Existing output
is never overwritten. Audit/preview results stay in memory until confirmation.

The notebooks contain explicit switches before downloading pretrained weights,
training, and final test evaluation. They do not download datasets. Before a Git
remote exists, upload a code-only repository ZIP into Colab; no commit/push is
needed. Notebook outputs must be cleared before copying edits back here.

## Boundaries

| Local repository / Codex | Google Colab | Google Drive |
|---|---|---|
| Code, clean notebooks, configuration, docs, software tests | GPU setup, training, evaluation, media inference, temporary data copies | Raw/prepared data, annotations, split manifests, weights, logs, reports, predictions |

No `ultralytics` package, CLI, trainers, utilities or model weights are used.
[Upstream provenance](docs/upstream_provenance.md) records the official source and
commit. Training requires Drive paths for checkpoint inputs and persistent outputs.
Do not place real datasets/weights in the repository even if ignored by Git.

## Defaults and outputs

- YOLOX-S, one class `aquafina_bottle` (model index 0; COCO category ID 1).
- Full visible bottle boxes; positive only when branding is identifiable.
- 640×640, batch 8, seed 42, FP16, 100 epochs; final 15 epochs without Mosaic.
- MixUp and horizontal flipping disabled to preserve brand evidence.
- TempleRAIL: retain the 870 unique original validation IDs and reconstruct 4,000
  training IDs as all 4,870 image IDs minus validation. Ignore corrupt `train.txt`
  membership. Assert zero ID overlap; flag identical image content across splits.
  There is no TempleRAIL test split. The generic COCO `prepare` API still supports
  its original 70/15/15 grouped split for other datasets.
- Only source Darknet class 0 (Aquafina) becomes a positive. Deer, Kirkland and
  Nestle remain background, including competitor-only images with empty annotations.
  Mixed-brand images retain only Aquafina boxes. The dog exception is reported and
  excluded. Source brand visibility requires human review; it is not inferred here.
- Epoch checkpoints include training model, EMA model, optimizer, AMP scaler and
  RNG states. Resume is epoch-based, not bitwise-identical worker replay.
- AP50 and AP50–95 from COCO, plus precision/recall at IoU 0.5 and negative-image
  false-positive metrics. Fractions in reports are in [0, 1].
- Validation selects highest recall subject to precision ≥0.95 and competitor-only
  image FPR ≤0.05. Missing evidence or no qualifying threshold is a failure, not success.
- Saved-video inference emits zero-based frame indices, nominal timestamps,
  original-coordinate boxes and annotated MP4 without audio.

## Local verification

Python 3.10+ is sufficient for the lightweight tests. Training environment pins
target Python 3.10–3.12 on Colab. In a virtual environment:

```text
python -m pip install -e ".[test]"
python -m pytest -q
python scripts/check_repository.py
```

OpenCV is needed for the media IO test; it skips when unavailable. Actual GPU tests
skip unless PyTorch, CUDA and the pinned YOLOX source are present. On restricted
Windows systems, make a `.test-tmp` directory and use a fresh
`--basetemp=.test-tmp/<run-name>` if the system temporary directory is inaccessible.

**Verification status:** local data/metrics/media and TempleRAIL fixture tests have run.
The real TempleRAIL files are on Drive and have not been audited by local Codex. Fresh Colab
installation, real GPU tests, weight adaptation, Drive reconnect and training have
not been executed locally. See [verification notes](docs/verification.md).

## Repository layout

```text
configs/       Thin experiment entrypoint and example storage paths
requirements/  Colab dependency pins
notebooks/     Preparation, training, evaluation/inference
src/aquafina_detector/
               Data validation/splitting, trainer, evaluation, inference, bootstrap
tests/         Local behavior tests and optional actual YOLOX GPU tests
scripts/       Clean notebook generation and repository checks
docs/          Data protocol, workflow, provenance and verification
```

No Git initialization, commit, or push is performed by setup scripts. These remain
separate user-approved actions. No deployment service, live camera, OCR stage, or
second bottle-brand class is included.
