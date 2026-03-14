import torch


def test_imagine_step_deterministic_latent(world_model, imagine_input):
    wm = world_model.to("cpu")

    with torch.no_grad():
        out1 = wm.imagine_step(imagine_input, stochastic=False)
        out2 = wm.imagine_step(imagine_input, stochastic=False)

    for k in out1:
        assert torch.allclose(out1[k], out2[k])
