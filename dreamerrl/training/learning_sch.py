class DreamerLRScheduler:
    def __init__(self, base_lr: float, warmup_steps: int):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps

    def __call__(self, update_idx: int):
        if update_idx < self.warmup_steps:
            return self.base_lr * (update_idx / self.warmup_steps)
        return self.base_lr
