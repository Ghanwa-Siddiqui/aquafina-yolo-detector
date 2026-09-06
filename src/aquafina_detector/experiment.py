"""One-class configuration extending the official YOLOX experiment."""
from yolox.exp import Exp


class AquafinaExp(Exp):
    def __init__(self):
        super().__init__()
        self.num_classes = 1
        self.depth, self.width = 0.33, 0.50
        self.exp_name = "aquafina_yolox_s"
        self.input_size = self.test_size = (640, 640)
        self.multiscale_range = 0
        self.train_ann, self.val_ann, self.test_ann = "train.json", "val.json", "test.json"
        self.seed = 42
        self.max_epoch = 100
        self.no_aug_epochs = 15
        self.eval_interval = 1
        self.data_num_workers = 2
        self.enable_mixup = False
        self.mixup_prob = 0.0
        # Horizontal flipping mirrors the printed brand; avoid it for this task.
        self.flip_prob = 0.0
        self.mosaic_scale = (0.5, 1.5)
        self.test_conf, self.nmsthre = 0.001, 0.65
        self.save_history_ckpt = False

    def get_trainer(self, args):
        from .trainer import DriveTrainer
        return DriveTrainer(self, args)

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        loader = super().get_data_loader(batch_size, is_distributed, no_aug, cache_img)
        # Four valid 50-bottle images must not be silently truncated by Mosaic.
        self.dataset.preproc.max_labels = 200
        return loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        if is_distributed:
            raise ValueError("This project supports one Colab GPU")
        from .training_evaluator import TrainingEvaluator
        return TrainingEvaluator(self.get_eval_loader(batch_size, False, testdev, legacy), self)
