# dreamerrl/tests/world_model/state/test_state_cloning_invariants.py
import torch

from dreamerrl.models.world_model import WorldModelState


def test_state_clone_independence():
    h = torch.randn(3, 5)
    z = torch.randn(3, 7)
    s = WorldModelState(h=h, z=z)

    c = s.clone()

    # Same values
    assert torch.allclose(s.h, c.h)
    assert torch.allclose(s.z, c.z)

    # Different storage (mutating clone does not affect original)
    c.h[0, 0] += 1.0
    c.z[0, 0] += 1.0
    assert not torch.allclose(s.h, c.h)
    assert not torch.allclose(s.z, c.z)


def test_state_detach_breaks_grad():
    h = torch.randn(3, 5, requires_grad=True)
    z = torch.randn(3, 7, requires_grad=True)
    s = WorldModelState(h=h, z=z)

    d = s.detach()

    assert not d.h.requires_grad
    assert not d.z.requires_grad
    # Values preserved
    assert torch.allclose(s.h, d.h)
    assert torch.allclose(s.z, d.z)
