import pytest
import torch


@pytest.mark.invariants
@pytest.mark.rssm
def test_rssm_deterministic_transition_invariance(world_model, dummy_actor):
    """
    RSSMCore is purely deterministic: same inputs → same outputs.
    This test ensures no hidden nondeterminism in the transition.
    """
    B = 4
    state = world_model.init_state(B)

    out1 = world_model.imagine_step(state, dummy_actor, deterministic_imagination=True)
    out2 = world_model.imagine_step(state, dummy_actor, deterministic_imagination=True)

    torch.testing.assert_close(out1.h, out2.h)
    torch.testing.assert_close(out1.z, out2.z)
