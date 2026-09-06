"""Colab single-GPU training entrypoint; outputs and weights stay in Drive."""
import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import random
from types import SimpleNamespace

from . import WEIGHTS_URL
from .common import read_json, require_drive, sha256, write_json
from .data import verify_prepared


def run(data_dir, run_dir, checkpoint, batch_size=8, epochs=100, resume=False, workers=2, stop_after_epochs=None):
    import numpy as np
    import torch
    from .bootstrap import audit_runtime
    from .experiment import AquafinaExp
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a Google Colab CUDA GPU")
    if batch_size not in (2, 4, 8) or epochs < 2:
        raise ValueError("Batch size must be 8, 4, or 2; epochs must be >= 2")
    run_dir, checkpoint = require_drive(run_dir), require_drive(checkpoint)
    data_dir = Path(data_dir).resolve()
    env = audit_runtime()
    manifests = verify_prepared(data_dir)
    code_hash = hashlib.sha256()
    for source in sorted(Path(__file__).parent.glob("*.py")):
        code_hash.update(source.name.encode())
        code_hash.update(source.read_bytes())
    config = {"batch_size": batch_size, "epochs": epochs, "seed": 42,
              "input_size": 640, "annotation_hashes": manifests, "workers": workers,
              "project_code_sha256": code_hash.hexdigest()}
    if resume:
        previous = read_json(run_dir / "run.json")
        if previous["config"] != config:
            raise ValueError("Resume must keep data, epochs, batch size and workers unchanged")
    else:
        if run_dir.exists():
            raise FileExistsError("Choose a new run directory, or explicitly resume")
        provenance = read_json(str(checkpoint) + ".provenance.json")
        if provenance.get("url") != WEIGHTS_URL or provenance.get("sha256") != sha256(checkpoint):
            raise ValueError("Initial weights must match the official download provenance")
        run_dir.mkdir(parents=True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    exp = AquafinaExp()
    exp.data_dir, exp.output_dir = str(data_dir), str(run_dir.parent)
    exp.max_epoch = epochs
    exp.data_num_workers = workers
    if epochs < 20:  # Short smoke runs must leave room for warmup and the schedule.
        exp.warmup_epochs, exp.no_aug_epochs = 0, 1
    args = SimpleNamespace(fp16=True, batch_size=batch_size, resume=resume, ckpt=str(checkpoint),
                           start_epoch=None, cache=False, occupy=False, logger="tensorboard", opts=[],
                           experiment_name=run_dir.name, stop_after_epochs=stop_after_epochs)
    event = {"timestamp": datetime.now(timezone.utc).isoformat(), "config": config,
             "environment": env, "checkpoint_sha256": sha256(checkpoint),
             "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda, "resume": resume}
    write_json(run_dir / ("resume-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".json" if resume else "run.json"), event)
    try:
        exp.get_trainer(args).train()
    except torch.cuda.OutOfMemoryError as error:
        write_json(run_dir / "failure.json", {"error": str(error), "suggestion":
            "Start a new run at batch 4, then 2. Learning rate scales automatically; do not change batch on resume."})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--batch-size", type=int, default=8, choices=(2, 4, 8))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop-after-epochs", type=int, help="Stop cleanly at this epoch without changing the schedule")
    a = p.parse_args()
    run(a.data, a.run, a.checkpoint, a.batch_size, a.epochs, a.resume, a.workers, a.stop_after_epochs)


if __name__ == "__main__":
    main()
