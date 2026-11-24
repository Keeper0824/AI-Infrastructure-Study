"""Validate the installed FlashInfer wheels with a tiny real-GPU RMSNorm."""

import json
from importlib import metadata

import torch
from flashinfer.norm import rmsnorm


def main():
    torch.manual_seed(0)
    values = torch.randn(4, 256, device="cuda", dtype=torch.float16)
    weights = torch.ones(256, device="cuda", dtype=torch.float16)
    actual = rmsnorm(values, weights, eps=1e-6)
    reference = values.float() * torch.rsqrt(
        values.float().square().mean(dim=-1, keepdim=True) + 1e-6
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(
        actual, reference.to(values.dtype), rtol=2e-3, atol=2e-3
    )
    packages = {
        name: metadata.version(name)
        for name in (
            "flashinfer-python", "flashinfer-cubin", "flashinfer-jit-cache",
            "flashinfer-jit-cache-sm120f",
        )
    }
    print(json.dumps({
        "gpu": torch.cuda.get_device_name(),
        "torch_cuda_runtime": torch.version.cuda,
        "packages": packages,
        "rmsnorm_shape": list(values.shape),
        "rmsnorm_correctness": "passed",
        "purpose": "correctness smoke test, not a performance benchmark",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
