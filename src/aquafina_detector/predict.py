"""Image/video inference and low-score COCO predictions using official YOLOX."""
import argparse
import json
import math
from pathlib import Path

from . import CLASS_NAME, UPSTREAM_COMMIT
from .common import contained, read_json, sha256, write_json


def restore_boxes(boxes, ratio, width, height):
    """YOLOX pads on bottom/right; only divide by the resize ratio, then clip."""
    result = []
    for box in boxes:
        x1, y1, x2, y2 = box
        result.append([max(0.0, min(width, x1 / ratio)), max(0.0, min(height, y1 / ratio)),
                       max(0.0, min(width, x2 / ratio)), max(0.0, min(height, y2 / ratio))])
    return result


class Predictor:
    def __init__(self, checkpoint, confidence=0.25, nms=0.65, device="cuda"):
        import torch
        from .experiment import AquafinaExp
        from .bootstrap import audit_runtime
        audit_runtime()
        if not (0 <= confidence <= 1 and 0 < nms <= 1):
            raise ValueError("Invalid confidence/NMS thresholds")
        self.device, self.confidence, self.nms = device, confidence, nms
        self.exp = AquafinaExp()
        self.model = self.exp.get_model().to(device).eval()
        # Only official or locally trained trusted checkpoints are supported.
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model"], strict=True)
        self.checkpoint_hash = sha256(checkpoint)

    def __call__(self, frame):
        import numpy as np
        import torch
        from yolox.data.data_augment import preproc
        from yolox.utils import postprocess
        if frame is None or frame.ndim != 3:
            raise ValueError("Expected a decoded BGR image")
        pixels, ratio = preproc(frame, self.exp.test_size)
        tensor = torch.from_numpy(np.ascontiguousarray(pixels)).unsqueeze(0).float().to(self.device)
        with torch.inference_mode():
            detections = postprocess(self.model(tensor), 1, self.confidence, self.nms, class_agnostic=True)[0]
        if detections is None:
            return []
        rows = detections.cpu().numpy()
        height, width = frame.shape[:2]
        boxes = restore_boxes(rows[:, :4].tolist(), ratio, width, height)
        return [{"bbox_xyxy": b, "class_id": 0, "class_name": CLASS_NAME,
                 "confidence": float(r[4] * r[5])}
                for r, b in zip(rows, boxes) if b[2] > b[0] and b[3] > b[1]]


def draw(frame, detections):
    import cv2
    frame = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = map(round, d["bbox_xyxy"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 210, 0), 2)
        cv2.putText(frame, f"Aquafina {d['confidence']:.2f}", (x1, max(18, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 210, 0), 1)
    return frame


def predict_media(predictor, source, output):
    import cv2
    source, output = Path(source), Path(output)
    if not source.is_file():
        raise FileNotFoundError(source)
    output.mkdir(parents=True, exist_ok=False)
    if source.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        frame = cv2.imread(str(source))
        detections = predictor(frame)
        if not cv2.imwrite(str(output / "annotated.jpg"), draw(frame, detections)):
            raise OSError("Could not save annotated image")
        write_json(output / "predictions.json", {"source": str(source), "detections": detections})
        return
    cap = cv2.VideoCapture(str(source))
    writer = None
    try:
        if not cap.isOpened():
            raise ValueError("Cannot open video")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video has no valid FPS; re-encode before inference")
        index = 0
        with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(str(output / "annotated.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                    if not writer.isOpened():
                        raise OSError("MP4 encoder unavailable")
                detections = predictor(frame)
                stream.write(json.dumps({"frame_index": index, "timestamp_seconds": index / fps,
                                          "detections": detections}) + "\n")
                writer.write(draw(frame, detections))
                index += 1
        if index == 0:
            raise ValueError("Video contains no decodable frames")
    finally:
        cap.release()
        if writer is not None:
            writer.release()


def predict_coco(predictor, annotations, images, output):
    import cv2
    from .data import validate
    data = read_json(annotations)
    validate(data, images)
    if Path(output).exists():
        raise FileExistsError(output)
    records = []
    for im in data["images"]:
        for d in predictor(cv2.imread(str(contained(images, im["file_name"])))):
            x1, y1, x2, y2 = d["bbox_xyxy"]
            records.append({"image_id": im["id"], "category_id": 1,
                            "bbox": [x1, y1, x2 - x1, y2 - y1], "score": d["confidence"]})
    write_json(output, records)
    write_json(str(output) + ".meta.json", {"annotations_sha256": sha256(annotations),
        "checkpoint_sha256": predictor.checkpoint_hash, "input_size": list(predictor.exp.test_size),
        "score_floor": predictor.confidence, "nms": predictor.nms, "upstream_commit": UPSTREAM_COMMIT})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", required=True, help="Media file or COCO split image directory")
    p.add_argument("--output", required=True)
    p.add_argument("--annotations", help="Emit COCO results for this split")
    p.add_argument("--confidence", type=float, default=None)
    p.add_argument("--nms", type=float, default=0.65)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--operating-point", help="Use a frozen validation threshold for media")
    a = p.parse_args()
    conf = a.confidence if a.confidence is not None else (0.001 if a.annotations else 0.25)
    if a.operating_point:
        if a.annotations or a.confidence is not None:
            p.error("Operating point is for media inference; cannot combine with confidence/annotations")
        point = read_json(a.operating_point)
        if point["status"] != "qualified" or point["provenance"]["checkpoint_sha256"] != sha256(a.checkpoint):
            raise ValueError("Operating point is unqualified or belongs to a different checkpoint")
        conf, a.nms = point["threshold"], point["provenance"]["nms"]
    predictor = Predictor(a.checkpoint, conf, a.nms, a.device)
    if a.annotations:
        predict_coco(predictor, a.annotations, a.input, a.output)
    else:
        predict_media(predictor, a.input, a.output)


if __name__ == "__main__":
    main()
