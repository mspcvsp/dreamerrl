import torch


def test_reward_stability_across_rollout(world_model, imagine_input):
    wm = world_model.to("cpu")

    with torch.no_grad():
        rollout = wm.imagination_rollout(imagine_input, horizon=5)
        rewards = [wm.reward_head(s.h, s.z) for s in rollout]

    assert len(rewards) == 5
    for r in rewards:
        assert r.shape[-1] == 1
