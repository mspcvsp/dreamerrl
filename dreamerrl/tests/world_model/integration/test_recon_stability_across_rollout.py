import torch


def test_recon_stability_across_rollout(world_model, imagine_input):
    wm = world_model.to("cpu")

    with torch.no_grad():
        rollout = wm.imagination_rollout(imagine_input, horizon=5)
        recons = [wm.decoder(s.h, s.z) for s in rollout]

    # keep whatever stability assertions you had, e.g. variance bounds, etc.
    assert len(recons) == 5
    for r in recons:
        assert r.shape[-1] == wm.flat_obs_dim
