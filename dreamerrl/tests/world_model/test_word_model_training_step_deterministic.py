import copy

import torch


def test_world_model_training_step_deterministic(make_world_model, fake_batch):
    torch.manual_seed(0)
    wm1 = make_world_model()
    wm2 = make_world_model()

    batch1 = copy.deepcopy(fake_batch)
    batch2 = copy.deepcopy(fake_batch)

    loss1, _ = wm1.training_step(batch1)
    loss2, _ = wm2.training_step(batch2)

    torch.testing.assert_close(loss1, loss2)
