from typing import Protocol

import torch

from dreamerrl.types import LSTMGates


class EvalOutputLike(Protocol):
    """
    Minimal structural type required by trainer.compute_lstm_unit_diagnostics().
    """

    gates: LSTMGates
    new_hxs: torch.Tensor
    new_cxs: torch.Tensor


class EvalOutProxy:
    """
    Lightweight proxy implementing EvalOutputLike for stitching TBPTT chunks.
    """

    def __init__(self, gates: LSTMGates, new_hxs: torch.Tensor, new_cxs: torch.Tensor):
        self.gates = gates
        self.new_hxs = new_hxs
        self.new_cxs = new_cxs
