import torch


def test_recon_stability_across_rollout(world_model, imagine_input):
    wm = world_model.to("cpu")

    with torch.no_grad():
        rollout = wm.imagination_rollout(imagine_input, horizon=5)
        recons = [wm.decoder(s["h"]) for s in rollout]

    for r in recons:
        assert torch.isfinite(r).all()
