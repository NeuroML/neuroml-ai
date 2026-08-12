#!/usr/bin/env python3

# Copyright 2026 Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
# File : test_torch.py
#
# Verify that the installed PyTorch build actually works on this machine.
# Prints an environment snapshot and runs a real CUDA compute op, which
# catches builds that report the GPU but contain no kernels for it (e.g.
# a +cu130 wheel on a Pascal GPU).

import argparse
import sys


def main() -> int:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-only",
        action="store_true",
        help="Only print the environment snapshot; skip the GPU compute test",
    )
    args = parser.parse_args()

    print(f"PyTorch version   : {torch.__version__}")
    print(f"CUDA build        : {torch.version.cuda}")

    if not torch.cuda.is_available():
        print("CUDA is not available: running on CPU only")
        return 0

    print(f"CUDA device       : {torch.cuda.get_device_name(0)}")
    capability = torch.cuda.get_device_capability(0)
    print(f"Compute capability: {capability[0]}.{capability[1]}")
    if args.env_only:
        return 0

    try:
        a = torch.rand(1024, 1024, device="cuda")
        b = torch.rand(1024, 1024, device="cuda")
        result = a @ b
        torch.cuda.synchronize()
        if result.shape != (1024, 1024):
            raise AssertionError(f"unexpected shape {result.shape}")
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: GPU compute test failed: {exc}")
        print(
            "The installed torch build likely has no kernels for this GPU. "
            "See requirements-torch.txt to install the matching CUDA build."
        )
        return 1

    torchvision = None
    try:
        import torchvision
    except ImportError:
        pass

    if torchvision is not None:
        try:
            boxes = torch.tensor(
                [[0, 0, 10, 10], [5, 5, 15, 15], [30, 30, 40, 40]],
                dtype=torch.float32,
                device="cuda",
            )
            scores = torch.tensor([0.9, 0.75, 0.8], device="cuda")
            torchvision.ops.nms(boxes, scores, 0.5)
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL: torchvision GPU compute test failed: {exc}")
            print(
                "torch and torchvision must come from the same CUDA build. "
                "See requirements-torch.txt to install the matching pair."
            )
            return 1

    print("GPU compute OK: 1024x1024 matmul ran on the GPU")
    return 0


if __name__ == "__main__":
    sys.exit(main())
