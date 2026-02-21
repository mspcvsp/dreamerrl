import torch

from dreamerrl.policy import LSTMPPOPolicy
from dreamerrl.trainer_state import TrainerState
from dreamerrl.types import PolicyEvalInput, PolicyInput


def test_pre_post_state_flow_alignment(trainer_state: TrainerState):
    assert trainer_state.env_info is not None
    trainer_state.env_info.flat_obs_dim = 4
    trainer_state.env_info.action_dim = 3

    B, T = 3, 7
    H = trainer_state.cfg.lstm.lstm_hidden_size

    policy = LSTMPPOPolicy(trainer_state)

    obs = torch.randn(T, B, trainer_state.env_info.flat_obs_dim)
    actions = torch.randint(0, trainer_state.env_info.action_dim, (T, B))
    h0 = torch.zeros(B, H)
    c0 = torch.zeros(B, H)

    # Full sequence
    full = policy.evaluate_actions_sequence(
        PolicyEvalInput(
            obs=obs,
            hxs=h0,
            cxs=c0,
            actions=actions,
        )
    )

    # Step-mode reconstruction of PRE/POST
    h_step = h0.clone()
    c_step = c0.clone()
    pre_h_list = []
    post_h_list = []
    pre_c_list = []
    post_c_list = []

    for t in range(T):
        pre_h_list.append(h_step)  # PRE-STEP
        pre_c_list.append(c_step)

        out = policy.forward(
            PolicyInput(
                obs=obs[t],  # (B, F)
                hxs=h_step,
                cxs=c_step,
            )
        )

        eval_step = policy.evaluate_actions(
            out,
            actions[t],
            pre_h=h_step,
            pre_c=c_step,
        )

        # Squeeze the fake time dimension
        h_step = eval_step.new_hxs.squeeze(0)  # (B, H)
        c_step = eval_step.new_cxs.squeeze(0)  # (B, H)

        post_h_list.append(h_step)  # POST-STEP
        post_c_list.append(c_step)

    pre_h = torch.stack(pre_h_list, dim=0)  # (T, B, H)
    post_h = torch.stack(post_h_list, dim=0)  # (T, B, H)
    pre_c = torch.stack(pre_c_list, dim=0)  # (T, B, H)
    post_c = torch.stack(post_c_list, dim=0)  # (T, B, H)

    assert torch.allclose(full.pre_hxs, pre_h, atol=1e-6)
    assert torch.allclose(full.new_hxs, post_h, atol=1e-6)

    assert torch.allclose(full.pre_cxs, pre_c, atol=1e-6)
    assert torch.allclose(full.new_cxs, post_c, atol=1e-6)
