import numpy as np

from dreamerrl.training.trainer import CosineWarmupScheduler
from dreamerrl.utils.types import LRScheduleConfig


def test_lr_scheduler_warmup_and_decay():
    cfg = LRScheduleConfig(base_lr=1e-3, warmup_steps=10, total_steps=100, lr_floor=0.1)
    sch = CosineWarmupScheduler(cfg)

    lrs = np.array([sch(t) for t in range(0, 100)])

    assert np.isclose(lrs[0], 0.0)
    assert lrs[5] > lrs[0]
    assert lrs[10] <= cfg.base_lr + 1e-8
    assert lrs[-1] > 0.0
    assert lrs[-1] < cfg.base_lr
