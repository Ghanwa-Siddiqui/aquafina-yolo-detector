# Upstream provenance

- Source: https://github.com/Megvii-BaseDetection/YOLOX.git
- Release: `0.3.0`
- Immutable commit: `419778480ab6ec0590e5d3831b3afb3b46ab2aa3`
- Official pretrained YOLOX-S:
  https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
- Custom training documentation:
  https://github.com/Megvii-BaseDetection/YOLOX/blob/0.3.0/docs/train_custom_data.md
- Upstream license remains in its source checkout. This project does not relicense
  upstream files or vendor another detector implementation.

The release commit was resolved from the official Git remote. Bootstrap checks the
remote URL, exact HEAD, clean tracked source and imports under `yolox/`. It rejects
imports rooted at `ultralytics`/`yolov5`, and runtime checks reject an installed or
importable `ultralytics` package. Mentioning another project in upstream comments
is not itself executable use of that project. These checks are practical provenance
controls, not a formal proof of every transitive dependency's historical origin.

Local inspection of the pinned checkout audited all 52 Python files under `yolox/`:
no executable imports rooted at `ultralytics` or `yolov5` were found. Runtime and
resolved-dependency audits still run again in Colab before model use.

Use official YOLOX model, preprocessing, postprocessing and trainer modules. The
project adds a configuration and a trainer subclass for full-state resume and exact
final no-Mosaic epochs. A standard pycocotools training evaluator avoids upstream's
single-batch timing division and optional compiled COCO extension, and clips boxes
consistently with final inference. The source itself remains unmodified. Optional deployment
dependencies/ONNX tools are not installed because export is outside this version.

Direct GPU dependencies are pinned in `requirements/colab.txt`; torch/torchvision
are pinned in the bootstrap installer. This is **not** a fully transitive lock.
Every runtime audit records all installed package versions and the installer prints
`pip check` findings. Colab can contain unrelated package conflicts; review any
conflict involving this project's stack before training. Never silently substitute
an Ultralytics-based wrapper to resolve an installation issue.

The weight download is explicit and Drive-only. It records its source URL and file
SHA256 and initial training checks the sidecar against the file. That is a local
integrity record, not an independently authenticated published checksum. Only load
trusted official weights or this project's own checkpoints; checkpoint deserialization
uses PyTorch's full-state format for resume.

No dataset or weight download was performed during local implementation. Colab
environment compatibility and actual pretrained adaptation must be verified by the
notebook before training.
