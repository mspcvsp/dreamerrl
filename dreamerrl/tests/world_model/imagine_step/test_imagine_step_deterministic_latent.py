import torch


def test_imagine_step_deterministic_latent(world_model, imagine_input):
    wm = world_model.to("cpu")

    with torch.no_grad():
        out1 = wm.imagine_step(imagine_input, stochastic=False)
        out2 = wm.imagine_step(imagine_input, stochastic=False)

    assert torch.allclose(out1.h, out2.h)
    assert torch.allclose(out1.z, out2.z)
