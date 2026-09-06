"""Run explicitly in Colab to verify the actual upstream GPU path."""
import importlib.util

import pytest


@pytest.mark.gpu
def test_yolox_negative_and_mixed_augmentation_loss(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or importlib.util.find_spec("yolox") is None:
        pytest.skip("Needs CUDA and pinned official YOLOX; run the Colab smoke cell")
    import numpy as np
    from PIL import Image
    from aquafina_detector.bootstrap import audit_runtime
    from aquafina_detector.common import write_json
    from aquafina_detector.experiment import AquafinaExp
    from yolox.data import COCODataset, TrainTransform, MosaicDetection
    audit_runtime()
    images = tmp_path / "train2017"
    images.mkdir()
    records = []
    for i in range(4):
        Image.new("RGB", (64, 64), (i * 30, 50, 100)).save(images / f"{i}.png")
        records.append({"id": i, "file_name": f"{i}.png", "width": 64, "height": 64})
    coco = {"images": records, "categories": [{"id": 1, "name": "aquafina_bottle"}],
            "annotations": []}
    exp = AquafinaExp()
    exp.input_size = exp.test_size = (64, 64)
    model = exp.get_model().cuda().train()
    for mixed in (False, True):
        if mixed:
            coco["annotations"] = [{"id": 1, "image_id": 0, "category_id": 1,
                                    "bbox": [10, 10, 20, 30], "area": 600, "iscrowd": 0}]
        write_json(tmp_path / "annotations" / "train.json", coco)
        base = COCODataset(str(tmp_path), "train.json", img_size=(64, 64),
                           preproc=TrainTransform(flip_prob=0))
        assert len(base) == 4
        assert base.load_anno(1).shape == (0, 5)
        if mixed:
            assert base.load_anno(0)[0, 4] == 0
        mosaic = MosaicDetection(base, (64, 64), mosaic=True,
                                 preproc=TrainTransform(max_labels=120, flip_prob=0), enable_mixup=False)
        for loader in (base, mosaic):
            samples = [loader[i] for i in range(2)]
            inputs = torch.from_numpy(np.stack([s[0] for s in samples])).float().cuda()
            targets = torch.from_numpy(np.stack([s[1] for s in samples])).float().cuda()
            model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                result = model(inputs, targets)
            assert torch.isfinite(result["total_loss"])
            result["total_loss"].backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.gpu
def test_full_checkpoint_restores_training_weights_and_optimizer(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or importlib.util.find_spec("yolox") is None:
        pytest.skip("Needs configured CUDA/YOLOX")
    from types import SimpleNamespace
    from aquafina_detector.trainer import DriveTrainer
    from yolox.utils import ModelEMA
    trainer = object.__new__(DriveTrainer)
    trainer.model = torch.nn.Linear(2, 1).cuda()
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01, momentum=0.9)
    trainer.model(torch.ones(2, 2, device="cuda")).sum().backward()
    trainer.optimizer.step()
    trainer.scaler = torch.amp.GradScaler("cuda")
    trainer.ema_model = ModelEMA(trainer.model)
    trainer.use_model_ema, trainer.rank = True, 0
    trainer.file_name, trainer.epoch, trainer.best_ap = str(tmp_path), 0, 0.2
    raw = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    with torch.no_grad():
        for p in trainer.ema_model.ema.parameters():
            p.add_(1)
    trainer.save_ckpt("latest", True)
    path = tmp_path / "latest_ckpt.pth"
    saved = torch.load(path, weights_only=False)
    assert saved["start_epoch"] == 1 and saved["best_ap"] == 0.2
    assert not torch.equal(saved["model"]["weight"], saved["training_model"]["weight"])
    replacement = torch.nn.Linear(2, 1).cuda()
    trainer.optimizer = torch.optim.SGD(replacement.parameters(), lr=0.01, momentum=0.9)
    trainer.device, trainer.args = "cuda:0", SimpleNamespace(ckpt=str(path), resume=True)
    trainer.resume_train(replacement)
    assert all(torch.equal(v, raw[k]) for k, v in replacement.state_dict().items())
    assert trainer.optimizer.state_dict()["state"]
    assert trainer.start_epoch == 1
