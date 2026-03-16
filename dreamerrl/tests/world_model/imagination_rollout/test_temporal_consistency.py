# dreamerrl/tests/world_model/imagination_rollout/test_temporal_consistency.py
import torch

from dreamerrl.models.world_model import WorldModelState
from dreamerrl.tools.rollout_inspector import check_rollout_consistency


def test_temporal_consistency(world_model, imagine_input):
    wm = world_model.to("cpu")
    h0: WorldModelState = imagine_input.to(torch.device("cpu"))

    with torch.no_grad():
        rollout = wm.imagination_rollout(h0, horizon=7)

    # 1) shapes consistent across time
    check_rollout_consistency(rollout)

    # 2) deterministic vs manual unroll equivalence
    manual = []
    s = h0
    with torch.no_grad():
        for _ in range(7):
            s = wm.imagine_step(s)
            manual.append(s)

    for t in range(7):
        rt = rollout[t]
        mt = manual[t]
        assert torch.allclose(rt.h, mt.h)
        assert torch.allclose(rt.z, mt.z)
