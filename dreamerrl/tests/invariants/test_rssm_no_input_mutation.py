import torch
import torch.nn.functional as F

from dreamerrl.models.world_model_core import RSSMCore


def test_rssm_no_input_mutation(latent, net):
    rssm = RSSMCore(latent=latent, net=net)
    h = torch.randn(4, latent.deter_size)
    a = F.one_hot(torch.randint(0, net.action_dim, (4,)), net.action_dim).float()

    h_clone = h.clone()
    a_clone = a.clone()

    _ = rssm(h, a)

    torch.testing.assert_close(h, h_clone)
    torch.testing.assert_close(a, a_clone)
