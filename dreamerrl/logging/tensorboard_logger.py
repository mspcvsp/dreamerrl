# lstmppo/logging/tensorboard_logger.py

from __future__ import annotations

from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger:
    def __init__(self, logdir: str, run_name: str):
        self.logdir = logdir
        self.run_name = run_name
        self.writer = SummaryWriter(log_dir=f"{logdir}/{run_name}")

    def close(self):
        self.writer.close()
