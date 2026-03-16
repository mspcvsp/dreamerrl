# dreamerrl/tests/world_model/imagine_step/test_no_grad_path.py
import torch

from dreamerrl.models.world_model import WorldModelState


def test_imagine_step_no_grad(world_model, imagine_input):
    wm = world_model.to("cpu")
    s0: WorldModelState = imagine_input

    with torch.no_grad():
        s1 = wm.imagine_step(s0)
        s2 = wm.imagine_step(s1)

    for s in (s1, s2):
        assert not s.h.requires_grad
        assert not s.z.requires_grad


def test_imagination_rollout_no_grad(world_model, imagine_input):
    wm = world_model.to("cpu")
    s0: WorldModelState = imagine_input

    with torch.no_grad():
        rollout = wm.imagination_rollout(s0, horizon=5)

    for s in rollout:
        assert not s.h.requires_grad
        assert not s.z.requires_grad
