"""Identity-aware Deep Sets encoder for defender-visible CAGE2 host beliefs."""

import torch
from torch import nn

from dreamerrl.models.base_obs_encoder import BaseEncoder
from dreamerrl.utils.transforms import symlog


class HostObsEncoder(BaseEncoder):
    """Pool complete host records; identities must move with fields when permuted.

    Missing/padded hosts use mask=False. A healthy all-zero record and the
    Compromised=Unknown category are valid observations, never missingness.
    """

    host_ids: torch.Tensor
    subnet_ids: torch.Tensor

    def __init__(self, host_order, subnet_ids, embed_dim=200):
        super().__init__()
        if not host_order or len(set(host_order)) != len(host_order):
            raise ValueError("Host identities must be nonempty and unique")
        if len(subnet_ids) != len(host_order) or set(subnet_ids) != {0, 1, 2}:
            raise ValueError("Each host needs a visible subnet group, with exactly three groups")
        self.host_order = tuple(host_order)
        self.register_buffer("host_ids", torch.arange(len(host_order)))
        self.register_buffer("subnet_ids", torch.tensor(subnet_ids, dtype=torch.long))
        self.host_embedding = nn.Embedding(len(host_order), 8)
        self.subnet_embedding = nn.Embedding(3, 4)
        self.host_mlp = nn.Sequential(nn.Linear(16, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU())
        self.context = nn.Sequential(nn.Linear(3 * 64 + 3, 64), nn.SiLU())
        self.project = nn.Sequential(nn.Linear(64 * 3, embed_dim), nn.SiLU())
        self._output_size = embed_dim

    @property
    def output_size(self):
        return self._output_size

    def encode_hosts(self, fields, host_ids, subnet_ids, mask=None):
        if fields.ndim < 2 or fields.shape[-1] != 4:
            raise ValueError("Expected (..., hosts, 4) visible host fields")
        shape = fields.shape[:-1]
        if host_ids.shape != fields.shape[-2:-1] or subnet_ids.shape != host_ids.shape:
            raise ValueError("Identity/subnet IDs must align with the host axis")
        if mask is None:
            mask = torch.ones(shape, device=fields.device, dtype=torch.bool)
        if mask.shape != shape or mask.dtype != torch.bool:
            raise ValueError("Host mask must be boolean with shape (..., hosts)")
        # Mask before nonlinear layers: missing NaN/sentinel fields must not leak.
        values = symlog(torch.where(mask.unsqueeze(-1), fields, 0))
        identities = self.host_embedding(host_ids).expand(*shape, 8)
        subnets = self.subnet_embedding(subnet_ids).expand(*shape, 4)
        tokens = self.host_mlp(torch.cat((values, identities, subnets), -1))
        masked = torch.where(mask.unsqueeze(-1), tokens, 0)
        count = mask.sum(-1, keepdim=True)
        mean = masked.sum(-2) / count.clamp_min(1)
        maximum = tokens.masked_fill(~mask.unsqueeze(-1), -torch.inf).amax(-2)
        maximum = torch.where(count > 0, maximum, 0)
        subnet_means, coverage = [], []
        for group in range(3):
            members = mask & (subnet_ids == group)
            group_count = members.sum(-1, keepdim=True)
            subnet_means.append(torch.where(members.unsqueeze(-1), tokens, 0).sum(-2) / group_count.clamp_min(1))
            # Fraction of actor-visible schema hosts present, not true host status.
            coverage.append(group_count.to(fields.dtype) / (self.subnet_ids == group).sum().clamp_min(1))
        context = self.context(torch.cat((*subnet_means, *coverage), -1))
        return self.project(torch.cat((mean, maximum, context), -1))

    def forward(self, obs, mask=None):
        if obs.ndim < 2 or obs.shape[-1] != 4 * len(self.host_order):
            raise ValueError("Observation dimension does not match the visible host schema")
        fields = obs.reshape(*obs.shape[:-1], len(self.host_order), 4)
        return self.encode_hosts(fields, self.host_ids, self.subnet_ids, mask)
