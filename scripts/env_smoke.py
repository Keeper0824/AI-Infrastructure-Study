"""Small correctness check for the actual CUDA and Triton execution paths."""

import importlib.metadata
import json

import torch
import triton
import triton.language as tl


@triton.jit
def add_one(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(X + offsets, mask=offsets < N, other=0)
    tl.store(Y + offsets, values + 1, mask=offsets < N)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA")

    torch.manual_seed(0)
    left = torch.randn(64, 64, device="cuda", dtype=torch.float32)
    right = torch.randn(64, 64, device="cuda", dtype=torch.float32)
    actual = (left @ right).cpu()
    expected = left.cpu() @ right.cpu()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

    # A non-multiple length also checks the final block's mask.
    values = torch.arange(4099, device="cuda", dtype=torch.float32)
    output = torch.empty_like(values)
    for _ in range(3):
        add_one[(triton.cdiv(values.numel(), 256),)](
            values, output, values.numel(), BLOCK=256
        )
    torch.cuda.synchronize()
    torch.testing.assert_close(output, values + 1, rtol=0, atol=0)

    print(json.dumps({
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "triton": triton.__version__,
        "vllm": importlib.metadata.version("vllm"),
        "pytorch_cuda_matmul": "passed",
        "triton_add_one_tail_mask": "passed",
        "purpose": "correctness smoke test, not a performance benchmark",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
