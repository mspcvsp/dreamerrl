import torch

from lstmppo.learning_sch import EntropySchdeduler, LearningRateScheduler
from lstmppo.types import Config


def test_lr_scheduler_monotonicity():
    cfg = Config()
    sched = LearningRateScheduler(cfg)
    sched.reset(total_updates=100)

    # Sample progress from 0 → 1
    pct = torch.linspace(0, 1, 50).tolist()
    lrs = [sched.lr_at(p) for p in pct]

    warmup_end = cfg.sched.lr_warmup_pct / 100

    # LR increases during warmup
    assert lrs[1] >= lrs[0]
    assert lrs[int(len(pct) * warmup_end)] >= lrs[0]

    # LR decreases after warmup
    assert lrs[-1] <= lrs[int(len(pct) * warmup_end)]


def test_entropy_scheduler_monotonicity():
    cfg = Config()
    sched = EntropySchdeduler(cfg)

    pct = torch.linspace(0, 1, 50).tolist()
    ents = [sched.entropy_coef_at(p) for p in pct]

    # Entropy must monotonically decrease
    for i in range(1, len(ents)):
        assert ents[i] <= ents[i - 1]
