import torch


def make_fake_world_model_batch(
    batch_size: int,
    seq_len: int,
    obs_dim: int,
    device: torch.device,
):
    torch.manual_seed(0)

    state = torch.randn(batch_size, seq_len, obs_dim, device=device)
    reward = torch.randn(batch_size, seq_len, device=device)
    is_first = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    is_last = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    is_terminal = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

    return {
        "state": state,
        "reward": reward,
        "is_first": is_first,
        "is_last": is_last,
        "is_terminal": is_terminal,
    }
