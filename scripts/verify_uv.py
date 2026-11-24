"""Check the maintained uv interpreter and a real Dreamer encoder backward pass."""

import argparse
import importlib.metadata
import json
import os
import sys
from pathlib import Path

import torch

from dreamerrl.models.mlp_obs_encoder import MLPObsEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    assert Path(sys.executable).parent == root / ".venv/bin"
    assert sys.prefix != sys.base_prefix
    if args.gpu:
        assert torch.cuda.is_available(), "GPU execution is required"
    device = torch.device("cuda" if args.gpu else "cpu")
    torch.manual_seed(0)
    encoder = MLPObsEncoder(52, 200).to(device)
    inputs = torch.randn(4, 52, device=device, requires_grad=True)
    loss = encoder(inputs).square().mean()
    loss.backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    grads = [p.grad for p in encoder.parameters()]
    assert all(g is not None and g.device == inputs.device and torch.isfinite(g).all() for g in grads)
    gradient_l1 = sum(float(g.abs().sum()) for g in grads if g is not None)
    assert gradient_l1 > 0 and torch.isfinite(loss)
    if args.gpu:
        torch.cuda.synchronize()
    print(
        json.dumps(
            dict(
                python=sys.executable,
                base_prefix=sys.base_prefix,
                torch=torch.__version__,
                cuda=torch.version.cuda,
                device=str(inputs.device),
                gpu=torch.cuda.get_device_name(0) if args.gpu else None,
                gymnasium=importlib.metadata.version("gymnasium"),
                finite_gradients=True,
                gradient_l1=gradient_l1,
                inherited_conda_prefix=os.environ.get("CONDA_PREFIX"),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
