import torch

from lstmppo.trainer import LSTMPPOTrainer
from lstmppo.types import LSTMStates, PolicyEvalInput


def test_rollout_boundary_lstm_state_handoff():
    trainer = LSTMPPOTrainer.for_validation()
    policy = trainer.policy
    state = trainer.state
    device = trainer.device

    T = 16
    B = 1
    H = state.cfg.lstm.lstm_hidden_size
    obs_dim = state.env_info.flat_obs_dim

    # First rollout
    obs1 = torch.randn(T, B, obs_dim, device=device)
    actions1 = torch.randint(0, state.env_info.action_dim, (T, B), device=device)
    h0 = torch.zeros(B, H, device=device)
    c0 = torch.zeros(B, H, device=device)

    full1 = policy.evaluate_actions_sequence(PolicyEvalInput(obs=obs1, hxs=h0, cxs=c0, actions=actions1))

    h_T = full1.new_hxs[-1]  # POST-STEP state at final timestep
    c_T = full1.new_cxs[-1]

    # Simulate buffer handoff
    handoff = LSTMStates(hxs=h_T.detach(), cxs=c_T.detach())

    # Second rollout
    obs2 = torch.randn(T, B, obs_dim, device=device)
    actions2 = torch.randint(0, state.env_info.action_dim, (T, B), device=device)

    full2 = policy.evaluate_actions_sequence(
        PolicyEvalInput(obs=obs2, hxs=handoff.hxs, cxs=handoff.cxs, actions=actions2)
    )

    # Full unroll over concatenated sequence
    obs_cat = torch.cat([obs1, obs2], dim=0)
    actions_cat = torch.cat([actions1, actions2], dim=0)

    full_cat = policy.evaluate_actions_sequence(PolicyEvalInput(obs=obs_cat, hxs=h0, cxs=c0, actions=actions_cat))

    # Compare PRE-STEP states of rollout 2 to the corresponding segment of full unroll
    pre2 = full2.pre_hxs
    pre_cat_segment = full_cat.pre_hxs[T : 2 * T]

    assert torch.allclose(pre2, pre_cat_segment, atol=1e-5)
