import torch


def test_mask_correctness(fake_buffer_loader):
    buf = fake_buffer_loader()

    terminated = buf.terminated
    truncated = buf.truncated
    masks = buf.masks

    T, B = terminated.shape

    # 1. Shapes must match
    assert masks.shape == (T, B)
    assert terminated.shape == truncated.shape == masks.shape

    # 2. mask = 1 - (terminated | truncated)
    expected = 1.0 - (terminated | truncated).float()
    assert torch.allclose(masks, expected), "mask must equal 1 - (terminated | truncated)"

    # 3. mask must be 0 wherever terminated or truncated is True
    assert torch.all(masks[terminated] == 0)
    assert torch.all(masks[truncated] == 0)

    # 4. mask must be in [0, 1]
    assert masks.min() >= 0
    assert masks.max() <= 1

    # 5. mask must be float32
    assert masks.dtype == torch.float32
