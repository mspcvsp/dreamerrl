import torch


def assert_close_cpu_gpu(fn, *args, atol=1e-5, rtol=1e-5):
    """
    fn: a function that takes tensors and returns tensors
    args: tensors on CPU
    """
    cpu_args = [a.cpu() for a in args]
    gpu_args = [a.cuda() for a in args]

    cpu_out = fn(*cpu_args)
    gpu_out = fn(*gpu_args)

    if isinstance(cpu_out, torch.Tensor):
        torch.testing.assert_close(cpu_out.cpu(), gpu_out.cpu(), atol=atol, rtol=rtol)
    elif isinstance(cpu_out, (list, tuple)):
        for c, g in zip(cpu_out, gpu_out):
            torch.testing.assert_close(c.cpu(), g.cpu(), atol=atol, rtol=rtol)
    else:
        raise TypeError("Unsupported output type for numerical equivalence test")
