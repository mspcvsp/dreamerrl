import math

from .types import Config


class EntropySchdeduler(object):
    def __init__(self, cfg: Config):
        self.anneal_entropy_flag = cfg.sched.anneal_entropy_flag
        self.debug_mode = cfg.trainer.debug_mode
        self.start_entropy_coef = cfg.sched.start_entropy_coef
        self.end_entropy_coef = cfg.sched.end_entropy_coef

    def reset(self, total_updates: int):
        if self.anneal_entropy_flag and self.debug_mode is False:
            self.slope = (self.end_entropy_coef - self.start_entropy_coef) / max(total_updates - 1, 1)

            self.bias = self.start_entropy_coef
        else:
            self.slope = 0
            self.bias = self.start_entropy_coef

    def __call__(self, update_idx):
        return self.bias + update_idx * self.slope

    def entropy_coef_at(self, pct: float) -> float:
        """
        Monotonic linear decay from start → end.
        """
        return self.start_entropy_coef + pct * (self.end_entropy_coef - self.start_entropy_coef)


class LearningRateScheduler(object):
    """
    Warmup + cosien decay learning rate schedule

    Microsoft Copilot explanation of why this is an optimal strategy:
    ----------------------------------------------------------------
    "Warmup solves a very specific problem. Early in training, the
    policy logits and value estimates are garbage.

    A high LR causes catastrophic updates. With LSTMs, this is even
    worse because:

    - hidden states are untrained
    - DropConnect amplifies noise
    - AR/TAR gradients are large early on
    - TBPTT chunks propagate instability

    Warmup gives the network time to “settle” before applying
    full‑strength PPO updates.

    Typical warmup:
    - 5% of total updates
    - LR ramps from 0 → base LR

    This alone often cuts early‑training variance in half.

    Linear decay is simple, but it has a flaw: It keeps the LR too
    high for too long, then too low too early. Cosine annealing
    instead:

    - decays slowly at first
    - decays rapidly near the end
    - gives you a long “productive plateau”
    - ends with a gentle landing

    This is ideal for PPO because:

    - early updates need stability
    - mid‑training needs exploration of parameter space
    - late training benefits from fine‑grained adjustments

    Cosine annealing also interacts beautifully with entropy
    annealing:

    - entropy ↓ encourages exploitation
    - LR ↓ encourages fine‑tuning
    - both curves taper together"
    """

    def __init__(self, cfg: Config):
        self.lr_warmup_pct = cfg.sched.lr_warmup_pct
        self.lr_final_pct = cfg.sched.lr_final_pct

        self.debug_mode = cfg.trainer.debug_mode
        self.base_lr = cfg.sched.base_lr
        self.end_lr = self.base_lr * self.lr_final_pct / 100
        self.perc_warmup_updates = self.lr_warmup_pct
        self.warmup_updates = None
        self.total_updates = None

    def reset(self, total_updates: int):
        self.total_updates = total_updates

        self.warmup_updates = max(int((self.total_updates * self.perc_warmup_updates / 100) + 0.5), 1)

    def __call__(self, update_idx: int):
        if self.debug_mode:
            return self.base_lr
        else:
            if self.warmup_updates is None or self.total_updates is None:
                raise ValueError("LearningRateScheduler not initialized. Call reset() before using.")

            # Zero learning rate is expected for 1st update
            if update_idx < self.warmup_updates:
                lr = self.base_lr * (update_idx / self.warmup_updates)
            # Learning rate cosine decay afterwards
            else:
                denom = max(self.total_updates - self.warmup_updates, 1)
                progress = (update_idx - self.warmup_updates) / denom
                lr = self.end_lr + 0.5 * (self.base_lr - self.end_lr) * (1 + math.cos(math.pi * progress))

            return lr

    def lr_at(self, pct: float) -> float:
        """
        pct ∈ [0,1] representing progress through training.
        Warmup increases LR linearly, then decays linearly.
        """
        warmup_end = self.lr_warmup_pct / 100
        final_end = self.lr_final_pct / 100

        if pct < warmup_end:
            # Linear warmup from 0 → base_lr
            return self.base_lr * (pct / warmup_end)

        if pct > final_end:
            # Linear decay from base_lr → 0
            return self.base_lr * (1 - (pct - final_end) / (1 - final_end))

        # Plateau region (if any)
        return self.base_lr
