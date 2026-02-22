import pytest
import torch


def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="cpu", help="cpu or cuda")


@pytest.fixture(scope="session")
def device(request):
    dev = request.config.getoption("--device")
    if dev == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(dev)
