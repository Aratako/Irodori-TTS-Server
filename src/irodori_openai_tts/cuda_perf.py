from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def configure_cuda_performance() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    logger.info("cuda performance flags enabled: tf32_matmul=true tf32_cudnn=true cudnn_benchmark=true")
