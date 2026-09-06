"""Official trainer extension for complete, atomic epoch-boundary checkpoints."""
import os
from pathlib import Path
import random
import shutil

import numpy as np
import torch
from yolox.core import Trainer
from yolox.utils import load_ckpt


class DriveTrainer(Trainer):
    def train_in_epoch(self):
        end = min(self.max_epoch, self.args.stop_after_epochs or self.max_epoch)
        if end <= self.start_epoch:
            raise ValueError("Stop epoch must be later than the checkpoint epoch")
        for self.epoch in range(self.start_epoch, end):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def resume_train(self, model):
        ckpt = torch.load(self.args.ckpt, map_location=self.device, weights_only=False)
        self.start_epoch = 0
        self._resume_state = None
        if self.args.resume:
            if ckpt.get("aquafina_checkpoint_version") != 1:
                raise ValueError("Resume requires this project's full-state checkpoint")
            model.load_state_dict(ckpt["training_model"], strict=True)
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scaler.load_state_dict(ckpt["scaler"])
            self.best_ap = ckpt["best_ap"]
            self.start_epoch = ckpt["start_epoch"]
            self._resume_state = ckpt
        else:
            # Megvii loader skips incompatible COCO class heads; backbone and
            # compatible box/objectness parameters are retained.
            model = load_ckpt(model, ckpt["model"])
        return model

    def before_train(self):
        super().before_train()
        if self._resume_state is not None:
            ckpt = self._resume_state
            if self.use_model_ema:
                self.ema_model.ema.load_state_dict(ckpt["model"])
                self.ema_model.updates = ckpt["ema_updates"]
            random.setstate(ckpt["python_rng"])
            np.random.set_state(ckpt["numpy_rng"])
            torch.set_rng_state(ckpt["torch_rng"].cpu())
            torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["cuda_rng"]])
            self._resume_state = None
        if self.start_epoch >= self.max_epoch:
            raise ValueError("Checkpoint already completed the configured epoch count")

    def before_epoch(self):
        # Exactly the final no_aug_epochs use no mosaic, including resumed runs.
        if self.epoch >= self.max_epoch - self.exp.no_aug_epochs:
            self.train_loader.close_mosaic()
            self.model.head.use_l1 = True
            self.no_aug = True

    def after_iter(self):
        for name, meter in self.meter.get_filtered_meter("loss").items():
            if not np.isfinite(float(meter.latest)):
                raise FloatingPointError(f"Non-finite {name}; stop and inspect data")
        super().after_iter()

    def after_epoch(self):
        super().after_epoch()
        # Upstream saves latest before evaluation; refresh it with current best AP.
        self.save_ckpt("latest")

    def save_ckpt(self, ckpt_name, update_best_ckpt=False):
        if self.rank != 0:
            return
        eval_model = self.ema_model.ema if self.use_model_ema else self.model
        state = {"aquafina_checkpoint_version": 1, "start_epoch": self.epoch + 1,
                 "model": eval_model.state_dict(), "training_model": self.model.state_dict(),
                 "optimizer": self.optimizer.state_dict(), "scaler": self.scaler.state_dict(),
                 "best_ap": self.best_ap, "ema_updates": self.ema_model.updates if self.use_model_ema else 0,
                 "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                 "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}
        path = Path(self.file_name) / f"{ckpt_name}_ckpt.pth"
        temp = path.with_suffix(".tmp")
        torch.save(state, temp)
        os.replace(temp, path)
        best = path.parent / "best_ckpt.pth"
        if update_best_ckpt or not best.exists():
            temporary_best = best.with_suffix(".tmp")
            shutil.copyfile(path, temporary_best)
            os.replace(temporary_best, best)
