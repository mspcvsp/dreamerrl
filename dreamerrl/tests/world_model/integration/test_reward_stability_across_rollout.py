import torch


def test_reward_stability_across_rollout(world_model, imagine_input):
    wm = world_model.to("cpu")

    with torch.no_grad():
        rollout = wm.imagination_rollout(imagine_input, horizon=5)
        rewards = [wm.reward_head(s["h"]) for s in rollout]

    for r in rewards:
        assert torch.isfinite(r).all()
