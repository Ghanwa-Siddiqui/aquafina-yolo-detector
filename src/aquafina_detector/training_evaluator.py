"""Small-dataset-safe validation using standard pycocotools and YOLOX NMS."""
import contextlib
import io

from .evaluate import coco_ap
from .predict import restore_boxes


class TrainingEvaluator:
    def __init__(self, dataloader, exp):
        self.dataloader, self.exp = dataloader, exp

    def evaluate(self, model, distributed=False, half=False):
        import torch
        from yolox.utils import postprocess
        if distributed:
            raise ValueError("Single-GPU evaluation only")
        device = next(model.parameters()).device
        records = []
        with torch.inference_mode():
            for inputs, _, sizes, ids in self.dataloader:
                inputs = inputs.to(device=device, dtype=torch.float32)
                with torch.autocast(device.type, enabled=half):
                    outputs = postprocess(model(inputs), 1, self.exp.test_conf, self.exp.nmsthre, class_agnostic=True)
                for output, height, width, image_id in zip(outputs, sizes[0], sizes[1], ids):
                    if output is None:
                        continue
                    height, width = int(height), int(width)
                    ratio = min(self.exp.test_size[0] / height, self.exp.test_size[1] / width)
                    rows = output.detach().cpu().tolist()
                    boxes = restore_boxes([row[:4] for row in rows], ratio, width, height)
                    for row, box in zip(rows, boxes):
                        x1, y1, x2, y2 = box
                        if x2 > x1 and y2 > y1:
                            records.append({"image_id": int(image_id), "category_id": 1,
                                            "bbox": [x1, y1, x2 - x1, y2 - y1], "score": row[4] * row[5]})
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            metrics = coco_ap(self.dataloader.dataset.coco.dataset, records)
        return metrics["ap50_95"] or 0.0, metrics["ap50"] or 0.0, buffer.getvalue()
