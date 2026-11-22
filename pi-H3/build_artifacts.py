"""Build and persist pi-H3 regional packages on ZeroGPU xlarge."""

import spaces
import torch
import torch._dynamo.utils

from app import build_demo, launch_demo

if __name__ == '__main__':
    torch._dynamo.utils.common_constant_types.add(torch._C.device)
    launch_demo(build_demo(compile_artifacts=True))
