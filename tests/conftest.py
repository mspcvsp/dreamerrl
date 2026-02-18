import pytest
import torch


def pytest_runtest_setup(item):
    if "gpu" in item.keywords and not torch.cuda.is_available():
        pytest.skip("Skipping GPU test because CUDA is not available")
