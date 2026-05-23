import torch


def test_imagination_observe_equivalence(world_model):
    """
    One-step imagine should match observe_step when using posterior z.
    """
    B = 4
    obs = torch.rand(B, 8)
    action = torch.nn.functional.one_hot(
        torch.randint(0, world_model.net_cfg.action_dim, (B,)), num_classes=world_model.net_cfg.action_dim
    ).float()

    state0 = world_model.init_state(B)

    # Observe step
    out = world_model.observe_step(
        prev_state=state0,
        obs=obs,
        action=action,
        reward=None,
        is_first=None,
        is_last=None,
        is_terminal=None,
    )
    post = out["post"]

    # Imagine step using posterior z
    class PosteriorActor(torch.nn.Module):
        def forward(self, h, z):
            return torch.zeros(B, world_model.net_cfg.action_dim)

    actor = PosteriorActor()
    imagined = world_model.imagine_step(state0, actor, stochastic=False)

    torch.testing.assert_close(imagined.h, post.h, atol=1e-4, rtol=1e-4)
