---
title: "StreamDiffusionV2 (Repository Snapshot)"
source_url: https://github.com/chenfengxu714/StreamDiffusionV2
fetch_date: 2026-02-22
source_type: code
author: chenfengxu714
---

# StreamDiffusionV2 (Repository Snapshot)

## File Tree (raw/repo)

````
README.md
causvid/data.py
causvid/dmd.py
causvid/loss.py
causvid/models/__init__.py
causvid/models/model_interface.py
causvid/models/wan/bidirectional_inference.py
causvid/models/wan/causal_inference.py
causvid/models/wan/causal_model.py
causvid/models/wan/causal_stream_inference.py
causvid/models/wan/flow_match.py
causvid/models/wan/wan_base/README.md
causvid/models/wan/wan_base/distributed/__init__.py
causvid/models/wan/wan_base/distributed/fsdp.py
causvid/models/wan/wan_base/distributed/xdit_context_parallel.py
causvid/models/wan/wan_base/image2video.py
causvid/models/wan/wan_base/modules/attention.py
causvid/models/wan/wan_base/modules/model.py
causvid/models/wan/wan_base/modules/vae.py
causvid/models/wan/wan_base/text2video.py
causvid/models/wan/wan_wrapper.py
causvid/scheduler.py
causvid/util.py
configs/sdxl_8node_dmd_config.yaml
configs/wan_bidirectional_dmd.yaml
configs/wan_bidirectional_dmd_from_scratch.yaml
configs/wan_causal_dmd_v2v.yaml
configs/wan_causal_dmd_v2v_14b.yaml
configs/wan_causal_dmd_warp_4step_cfg2.yaml
configs/wan_causal_ode.yaml
demo/README.md
demo/frontend/README.md
streamv2v/communication/__init__.py
streamv2v/communication/buffer_manager.py
streamv2v/communication/data_containers.py
streamv2v/communication/distributed_communicator.py
streamv2v/communication/kv_cache_manager.py
streamv2v/communication/model_data_transfer.py
streamv2v/communication/utils.py
streamv2v/inference.py
streamv2v/inference_pipe.py
streamv2v/inference_wo_batch.py
````

## File: README.md

````markdown
# StreamDiffusionV2: A Streaming System for Dynamic and Interactive Video Generation (MLSys 2026)

[Tianrui Feng](https://jerryfeng2003.github.io/)<sup>1</sup>, [Zhi Li](https://scholar.google.com/citations?user=C6kPjgwAAAAJ&hl)<sup>2</sup>, [Shuo Yang](https://andy-yang-1.github.io/)<sup>2</sup>, [Haocheng Xi](https://haochengxi.github.io/)<sup>2</sup>, [Muyang Li](https://lmxyy.me/)<sup>3</sup>, [Xiuyu Li](https://xiuyuli.com/)<sup>1</sup>, [Lvmin Zhang](https://lllyasviel.github.io/lvmin_zhang/)<sup>4</sup>, [Keting Yang](https://www.linkedin.com/in/kellyzpeng/)<sup>5</sup>, [Kelly Peng](https://www.linkedin.com/in/kellyzpeng/)<sup>6</sup>, [Song Han](https://hanlab.mit.edu/songhan)<sup>7</sup>, [Maneesh Agrawala](https://graphics.stanford.edu/~maneesh/)<sup>4</sup>, [Kurt Keutzer](https://people.eecs.berkeley.edu/~keutzer/)<sup>2</sup>, [Akio Kodaira](https://scholar.google.com/citations?hl=ja&user=15X3cioAAAAJ)<sup>8</sup>, [Chenfeng Xu](https://www.chenfengx.com/)<sup>†,1</sup>

<sup>1</sup>UT Austin, <sup>2</sup>UC Berkeley, <sup>3</sup>Nunchaku AI, <sup>4</sup>Stanford University, <sup>5</sup>Independent Researcher, <sup>6</sup>First Intelligence, <sup>7</sup>MIT, <sup>8</sup>Shizhuku AI

<sup>†</sup> Project lead, corresponding to [xuchenfeng@utexas.edu](mailto:xuchenfeng@utexas.edu)

[![Project](https://img.shields.io/badge/Homepage-project-orange.svg?logo=googlehome)](https://streamdiffusionv2.github.io/) [![arXiv](https://img.shields.io/badge/Arxiv-2511.07399-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2511.07399) [![Hugging Face](https://img.shields.io/badge/HuggingFace-Space-blue.svg?logo=huggingface)](https://huggingface.co/jerryfeng/StreamDiffusionV2)

<p align="center">
  <image src="./assets/demo-1.gif" controls width="800">
  <image src="./assets/demo-2.gif" controls width="800">
  <image src="./assets/demo-3.gif" controls width="800">
</p>

## Overview

StreamDiffusionV2 is an open-source interactive diffusion pipeline for real-time streaming applications. It scales across diverse GPU setups, supports flexible denoising steps, and delivers high FPS for creators and platforms. Further details are available on our project [homepage](https://streamdiffusionv2.github.io/).

## News
- **[2026-01-26]** 🎉 [StreamDiffusionV2](https://arxiv.org/abs/2511.07399) is accepted by MLSys 2026!
- **[2025-11-10]** 🚀 We have released our [paper](https://arxiv.org/abs/2511.07399) at arXiv. Check it for more details!
- **[2025-10-18]** Release our model checkpoint on [huggingface](https://huggingface.co/jerryfeng/StreamDiffusionV2/).
- **[2025-10-06]** 🔥 Our [StreamDiffusionV2](https://github.com/chenfengxu714/StreamDiffusionV2) is publicly released! Check our project [homepage](https://streamdiffusionv2.github.io/) for more details.

## Prerequisites

- OS: Linux with NVIDIA GPU
- CUDA-compatible GPU and drivers

## Installation

```shell
conda create -n stream python=3.10.0
conda activate stream
# Require CUDA 12.4 or above, please check via `nvcc -V`
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt 
python setup.py develop
```

## Download Checkpoints

```shell
# 1.3B Model
huggingface-cli download --resume-download Wan-AI/Wan2.1-T2V-1.3B --local-dir wan_models/Wan2.1-T2V-1.3B
huggingface-cli download --resume-download jerryfeng/StreamDiffusionV2 --local-dir ./ckpts --include "wan_causal_dmd_v2v/*"

# 14B Model
huggingface-cli download --resume-download Wan-AI/Wan2.1-T2V-14B --local-dir wan_models/Wan2.1-T2V-14B
huggingface-cli download --resume-download jerryfeng/StreamDiffusionV2 --local-dir ./ckpts --include "wan_causal_dmd_v2v_14b/*"
```
We use the 14B model from [CausVid-Plus](https://github.com/GoatWu/CausVid-Plus) for offline inference demo.

## Offline Inference

### Single GPU

```shell
python streamv2v/inference.py \
--config_path configs/wan_causal_dmd_v2v.yaml \
--checkpoint_folder ckpts/wan_causal_dmd_v2v \
--output_folder outputs/ \
--prompt_file_path examples/original.mp4 \
--video_path examples/original.mp4 \
--height 480 \
--width 832 \
--fps 16 \
--step 2
```
Note: `--step` sets how many denoising steps are used during inference.

### Multi-GPU

```shell
torchrun --nproc_per_node=2 --master_port=29501 streamv2v/inference_pipe.py \
--config_path configs/wan_causal_dmd_v2v.yaml \
--checkpoint_folder ckpts/wan_causal_dmd_v2v \
--output_folder outputs/ \
--prompt_file_path examples/original.mp4 \
--video_path examples/original.mp4 \
--height 480 \
--width 832 \
--fps 16 \
--step 2
# --schedule_block  # optional: enable block scheduling
```
Note: `--step` sets how many denoising steps are used during inference. Enabling `--schedule_block` can provide optimal throughput.

Adjust `--nproc_per_node` to your GPU count. For different resolutions or FPS, change `--height`, `--width`, and `--fps` accordingly.

## Online Inference (Web UI)
A minimal web demo is available under `demo/`. For setup and startup, please refer to [demo](demo/README.md).
- Access in a browser after startup: `http://0.0.0.0:7860` or `http://localhost:7860`


## To-do List

- [x] Demo and inference pipeline.
- [ ] Dynamic scheduler for various workload.
- [ ] Training code.
- [ ] FP8 support.
- [ ] TensorRT support.

## Acknowledgements
StreamDiffusionV2 is inspired by the prior works [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) and [StreamV2V](https://github.com/Jeff-LiangF/streamv2v). Our Causal DiT builds upon [CausVid](https://github.com/tianweiy/CausVid), and the rolling KV cache design is inspired by [Self-Forcing](https://github.com/guandeh17/Self-Forcing).

We are grateful to the team members of [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) for their support. We also thank [First Intelligence](https://first-intelligence.com) and [Daydream](https://docs.daydream.live/) team for their great feedback.

We also especially thank DayDream team for the great collaboration and incorporating our StreamDiffusionV2 pipeline into their cool [Demo UI](https://github.com/daydreamlive/scope). 

## Citation

If you find this repository useful in your research, please consider giving a star ⭐ or a citation.
```BibTeX
@article{feng2025streamdiffusionv2,
  title={StreamDiffusionV2: A Streaming System for Dynamic and Interactive Video Generation},
  author={Feng, Tianrui and Li, Zhi and Yang, Shuo and Xi, Haocheng and Li, Muyang and Li, Xiuyu and Zhang, Lvmin and Yang, Keting and Peng, Kelly and Han, Song and others},
  journal={arXiv preprint arXiv:2511.07399},
  year={2025}
}
```
````

## File: streamv2v/communication/__init__.py

````python
"""
Communication module for distributed inference pipeline.

This module provides abstractions for distributed communication operations,
model data transfer, and buffer management in the StreamDiffusionV2 pipeline.
"""

from .distributed_communicator import DistributedCommunicator
from .model_data_transfer import ModelDataTransfer
from .buffer_manager import BufferManager
from .data_containers import LatentData, KVCacheData, CommunicationConfig
from .kv_cache_manager import KVCacheManager
from .utils import CommunicationTags, init_distributed, setup_logging, compute_balanced_split

__all__ = [
    'DistributedCommunicator',
    'ModelDataTransfer', 
    'BufferManager',
    'LatentData',
    'KVCacheData',
    'CommunicationConfig',
    'KVCacheManager',
    'CommunicationTags',
    'init_distributed',
    'setup_logging',
    'compute_balanced_split'
]
````

## File: streamv2v/communication/buffer_manager.py

````python
"""
Buffer manager for efficient GPU memory management.

This module provides a buffer pool manager to avoid repeated GPU memory allocations
during distributed communication operations.
"""

import torch
from typing import Dict, List, Tuple, Optional
import threading
import logging
from .data_containers import CommunicationConfig


class BufferManager:
    """
    Manages GPU buffer pools to avoid repeated allocations.
    
    This class maintains pools of pre-allocated GPU tensors that can be reused
    across communication operations, reducing memory allocation overhead.
    """
    
    def __init__(self, device: torch.device, config: Optional[CommunicationConfig] = None):
        """
        Initialize the buffer manager.
        
        Args:
            device: GPU device for buffer allocation
            config: Communication configuration
        """
        self.device = device
        self.config = config or CommunicationConfig()
        
        # Buffer pools: {shape: [tensor1, tensor2, ...]}
        self.free_buffers = {}  # For latent tensors
        self.free_buffers_origin = {}  # For original latent tensors
        self.free_buffers_kv = {}  # For KV cache tensors
        self.free_buffers_misc = {}  # For headers, shapes, index vectors (int64 etc.)
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Statistics
        self.allocation_count = 0
        self.reuse_count = 0
        self.total_allocated_memory = 0
        
        # Setup logging
        self.logger = logging.getLogger(f"BufferManager_{device}")
        self.logger.propagate = False
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            # handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                f'[BufferManager {device}] %(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        # self.logger.setLevel(logging.DEBUG)
    
    def get_buffer(self, shape: Tuple[int, ...], dtype: torch.dtype, 
                   buffer_type: str = "latent") -> torch.Tensor:
        """
        Get or allocate a buffer with the specified shape and dtype.
        
        Args:
            shape: Tensor shape
            dtype: Tensor data type
            buffer_type: Type of buffer ("latent", "origin", "kv")
            
        Returns:
            Tensor buffer
        """
        with self._lock:
            # Select the appropriate buffer pool
            if buffer_type == "latent":
                buffer_pool = self.free_buffers
            elif buffer_type == "origin":
                buffer_pool = self.free_buffers_origin
            elif buffer_type == "kv":
                buffer_pool = self.free_buffers_kv
            elif buffer_type == "misc":
                buffer_pool = self.free_buffers_misc
            else:
                raise ValueError(f"Unknown buffer type: {buffer_type}")
            
            # Try to reuse existing buffer
            if self.config.enable_buffer_reuse and shape in buffer_pool and len(buffer_pool[shape]) > 0:
                buffer = buffer_pool[shape].pop()
                self.reuse_count += 1
                self.logger.debug(f"Reused buffer of shape {shape}, type {buffer_type}")
                return buffer
            
            # Allocate new buffer
            buffer = torch.empty(shape, dtype=dtype, device=self.device)
            self.allocation_count += 1
            self.total_allocated_memory += buffer.numel() * buffer.element_size()
            
            self.logger.debug(f"Allocated new buffer of shape {shape}, type {buffer_type}")
            return buffer
    
    def return_buffer(self, tensor: torch.Tensor, buffer_type: str = "latent") -> None:
        """
        Return a buffer to the pool for reuse.
        
        Args:
            tensor: Tensor to return
            buffer_type: Type of buffer ("latent", "origin", "kv")
        """
        if not self.config.enable_buffer_reuse:
            return
        
        with self._lock:
            # Select the appropriate buffer pool
            if buffer_type == "latent":
                buffer_pool = self.free_buffers
            elif buffer_type == "origin":
                buffer_pool = self.free_buffers_origin
            elif buffer_type == "kv":
                buffer_pool = self.free_buffers_kv
            elif buffer_type == "misc":
                buffer_pool = self.free_buffers_misc
            else:
                raise ValueError(f"Unknown buffer type: {buffer_type}")
            
            shape = tuple(tensor.shape)
            
            # Initialize pool for this shape if it doesn't exist
            if shape not in buffer_pool:
                buffer_pool[shape] = []
            
            # Add buffer to pool if not at capacity
            if len(buffer_pool[shape]) < self.config.buffer_pool_size:
                # Clear the tensor to free memory
                tensor.zero_()
                buffer_pool[shape].append(tensor)
                self.logger.debug(f"Returned buffer of shape {shape}, type {buffer_type}")
            else:
                self.logger.debug(f"Buffer pool full for shape {shape}, type {buffer_type}, discarding")
    
    def clear_buffers(self, buffer_type: Optional[str] = None) -> None:
        """
        Clear buffer pools to free memory.
        
        Args:
            buffer_type: Specific buffer type to clear, or None to clear all
        """
        with self._lock:
            if buffer_type is None:
                # Clear all buffer pools
                self.free_buffers.clear()
                self.free_buffers_origin.clear()
                self.free_buffers_kv.clear()
                self.free_buffers_misc.clear()
                self.logger.info("Cleared all buffer pools")
            else:
                # Clear specific buffer pool
                if buffer_type == "latent":
                    self.free_buffers.clear()
                elif buffer_type == "origin":
                    self.free_buffers_origin.clear()
                elif buffer_type == "kv":
                    self.free_buffers_kv.clear()
                elif buffer_type == "misc":
                    self.free_buffers_misc.clear()
                else:
                    raise ValueError(f"Unknown buffer type: {buffer_type}")
                self.logger.info(f"Cleared {buffer_type} buffer pool")
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get buffer manager statistics.
        
        Returns:
            Dictionary containing statistics
        """
        with self._lock:
            total_free_buffers = sum(len(pool) for pool in self.free_buffers.values())
            total_free_buffers_origin = sum(len(pool) for pool in self.free_buffers_origin.values())
            total_free_buffers_kv = sum(len(pool) for pool in self.free_buffers_kv.values())
            total_free_buffers_misc = sum(len(pool) for pool in self.free_buffers_misc.values())

            return {
                "allocation_count": self.allocation_count,
                "reuse_count": self.reuse_count,
                "total_allocated_memory_bytes": self.total_allocated_memory,
                "total_free_buffers": total_free_buffers,
                "total_free_buffers_origin": total_free_buffers_origin,
                "total_free_buffers_kv": total_free_buffers_kv,
                "total_free_buffers_misc": total_free_buffers_misc,
                "reuse_rate": self.reuse_count / max(1, self.allocation_count),
                "buffer_pool_size": self.config.buffer_pool_size,
                "enable_buffer_reuse": self.config.enable_buffer_reuse
            }
    
    def print_statistics(self) -> None:
        """Print buffer manager statistics."""
        stats = self.get_statistics()
        self.logger.info("Buffer Manager Statistics:")
        for key, value in stats.items():
            self.logger.info(f"  {key}: {value}")
    
    def preallocate_buffers(self, common_shapes: List[Tuple[Tuple[int, ...], torch.dtype, str]], 
                          count_per_shape: int = 5) -> None:
        """
        Preallocate buffers for common shapes to reduce allocation overhead.
        
        Args:
            common_shapes: List of (shape, dtype, buffer_type) tuples
            count_per_shape: Number of buffers to preallocate per shape
        """
        with self._lock:
            for shape, dtype, buffer_type in common_shapes:
                for _ in range(count_per_shape):
                    buffer = torch.empty(shape, dtype=dtype, device=self.device)
                    
                    # Select the appropriate buffer pool
                    if buffer_type == "latent":
                        buffer_pool = self.free_buffers
                    elif buffer_type == "origin":
                        buffer_pool = self.free_buffers_origin
                    elif buffer_type == "kv":
                        buffer_pool = self.free_buffers_kv
                    elif buffer_type == "misc":
                        buffer_pool = self.free_buffers_misc
                    else:
                        raise ValueError(f"Unknown buffer type: {buffer_type}")
                    
                    # Initialize pool for this shape if it doesn't exist
                    if shape not in buffer_pool:
                        buffer_pool[shape] = []
                    
                    buffer_pool[shape].append(buffer)
                    self.allocation_count += 1
                    self.total_allocated_memory += buffer.numel() * buffer.element_size()
            
            self.logger.info(f"Preallocated {len(common_shapes) * count_per_shape} buffers")
    
    def __del__(self):
        """Cleanup when the buffer manager is destroyed."""
        self.clear_buffers()
````

## File: streamv2v/communication/data_containers.py

````python
"""
Data containers for communication operations.

This module defines data structures used for communication between distributed ranks.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import torch


@dataclass
class LatentData:
    """
    Container for latent data and related information.
    
    This class encapsulates all the data that needs to be transferred between ranks
    during the inference pipeline.
    """
    chunk_idx: int
    latents: torch.Tensor
    original_latents: torch.Tensor
    current_start: torch.Tensor
    current_end: torch.Tensor
    current_step: int
    patched_x_shape: torch.Tensor
    
    def __post_init__(self):
        """Validate tensor shapes and types after initialization."""
        if not isinstance(self.latents, torch.Tensor):
            raise TypeError("latents must be a torch.Tensor")
        if not isinstance(self.original_latents, torch.Tensor):
            raise TypeError("original_latents must be a torch.Tensor")
        if not isinstance(self.current_start, torch.Tensor):
            raise TypeError("current_start must be a torch.Tensor")
        if not isinstance(self.current_end, torch.Tensor):
            raise TypeError("current_end must be a torch.Tensor")
        if not isinstance(self.patched_x_shape, torch.Tensor):
            raise TypeError("patched_x_shape must be a torch.Tensor")


@dataclass
class KVCacheData:
    """
    Container for KV cache data.
    
    This class encapsulates key-value cache information for transformer blocks.
    """
    block_index: int
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    global_end_index: torch.Tensor
    local_end_index: torch.Tensor
    
    def __post_init__(self):
        """Validate tensor shapes and types after initialization."""
        if not isinstance(self.k_cache, torch.Tensor):
            raise TypeError("k_cache must be a torch.Tensor")
        if not isinstance(self.v_cache, torch.Tensor):
            raise TypeError("v_cache must be a torch.Tensor")
        if not isinstance(self.global_end_index, torch.Tensor):
            raise TypeError("global_end_index must be a torch.Tensor")
        if not isinstance(self.local_end_index, torch.Tensor):
            raise TypeError("local_end_index must be a torch.Tensor")


@dataclass
class CommunicationConfig:
    """
    Configuration for communication operations.
    
    This class holds configuration parameters for distributed communication.
    """
    max_outstanding: int = 1
    buffer_pool_size: int = 10
    enable_buffer_reuse: bool = True
    communication_timeout: float = 30.0
    enable_async_communication: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.max_outstanding < 1:
            raise ValueError("max_outstanding must be at least 1")
        if self.buffer_pool_size < 1:
            raise ValueError("buffer_pool_size must be at least 1")
        if self.communication_timeout <= 0:
            raise ValueError("communication_timeout must be positive")


@dataclass
class BlockInterval:
    """
    Container for block interval information.
    
    This class represents a block interval [start, end) for a specific rank.
    """
    start: int
    end: int
    rank: int
    
    def __post_init__(self):
        """Validate block interval parameters."""
        if self.start < 0:
            raise ValueError("start must be non-negative")
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        if self.rank < 0:
            raise ValueError("rank must be non-negative")
    
    @property
    def size(self) -> int:
        """Get the size of the block interval."""
        return self.end - self.start
    
    def contains(self, block_index: int) -> bool:
        """Check if the block interval contains the given block index."""
        return self.start <= block_index < self.end


@dataclass
class PerformanceMetrics:
    """
    Container for performance metrics.
    
    This class holds timing and performance information for communication operations.
    """
    dit_time: float
    total_time: float
    communication_time: float
    buffer_allocation_time: float
    
    def __post_init__(self):
        """Validate performance metrics."""
        if self.dit_time < 0:
            raise ValueError("dit_time must be non-negative")
        if self.total_time < 0:
            raise ValueError("total_time must be non-negative")
        if self.communication_time < 0:
            raise ValueError("communication_time must be non-negative")
        if self.buffer_allocation_time < 0:
            raise ValueError("buffer_allocation_time must be non-negative")
    
    @property
    def efficiency(self) -> float:
        """Calculate communication efficiency (computation time / total time)."""
        if self.total_time == 0:
            return 0.0
        return (self.total_time - self.communication_time) / self.total_time
````

## File: streamv2v/communication/distributed_communicator.py

````python
"""
Distributed communication abstraction layer.

This module provides a high-level interface for distributed communication operations,
encapsulating the low-level PyTorch distributed primitives.
"""

import torch
import torch.distributed as dist
from typing import List, Tuple, Optional, Any
import logging
import time
from .utils import CommunicationTags, get_next_rank, get_prev_rank, CommunicationTimer
from .data_containers import CommunicationConfig


class DistributedCommunicator:
    """
    High-level interface for distributed communication operations.
    
    This class encapsulates all distributed communication operations, providing
    a clean interface for sending and receiving tensors between ranks.
    """
    
    def __init__(self, rank: int, world_size: int, device: torch.device, 
                 config: Optional[CommunicationConfig] = None):
        """
        Initialize the distributed communicator.
        
        Args:
            rank: Current rank
            world_size: Total number of ranks
            device: GPU device for communication
            config: Communication configuration
        """
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.config = config or CommunicationConfig()
        
        # Track outstanding operations
        self.outstanding_operations: List[Any] = []
        
        # Setup logging
        self.logger = logging.getLogger(f"DistributedCommunicator_rank_{rank}")
        self.logger.propagate = False
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'[Rank {rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Validate distributed is initialized
        if not dist.is_initialized():
            raise RuntimeError("Distributed not initialized. Call init_distributed() first.")
    
    def send_tensor_async(self, tensor: torch.Tensor, dst: int, tag: int) -> Any:
        """
        Asynchronously send a tensor to the specified destination.
        
        Args:
            tensor: Tensor to send
            dst: Destination rank
            tag: Communication tag
            
        Returns:
            Work object for the send operation
        """
        if tensor.device != self.device:
            raise ValueError(f"Tensor device {tensor.device} doesn't match communicator device {self.device}")
        
        work = dist.isend(tensor, dst=dst, tag=tag)
        self.outstanding_operations.append(work)
        
        self.logger.debug(f"Started async send to rank {dst} with tag {tag}, tensor shape: {tensor.shape}")
        return work
    
    def recv_tensor(self, src: int, tag: int, shape: Tuple[int, ...], 
                   dtype: torch.dtype) -> torch.Tensor:
        """
        Receive a tensor from the specified source.
        
        Args:
            src: Source rank
            tag: Communication tag
            shape: Expected tensor shape
            dtype: Expected tensor dtype
            
        Returns:
            Received tensor
        """
        tensor = torch.empty(shape, dtype=dtype, device=self.device)
        
        with CommunicationTimer(f"recv_tensor from rank {src}", self.logger):
            dist.recv(tensor, src=src, tag=tag)
        
        self.logger.debug(f"Received tensor from rank {src} with tag {tag}, shape: {tensor.shape}")
        return tensor
    
    def send_header_and_tensor_async(self, header: torch.Tensor, tensor: torch.Tensor,
                                   dst: int, tag_header: int, tag_tensor: int) -> Tuple[Any, Any]:
        """
        Asynchronously send a header and tensor pair.
        
        Args:
            header: Header tensor containing metadata
            tensor: Data tensor
            dst: Destination rank
            tag_header: Tag for header
            tag_tensor: Tag for tensor
            
        Returns:
            Tuple of (header_work, tensor_work)
        """
        if header.device != self.device or tensor.device != self.device:
            raise ValueError("Header and tensor must be on the same device as communicator")
        
        header_work = dist.isend(header, dst=dst, tag=tag_header)
        tensor_work = dist.isend(tensor, dst=dst, tag=tag_tensor)
        
        self.outstanding_operations.extend([header_work, tensor_work])
        
        self.logger.debug(f"Started async send of header+tensor to rank {dst}, "
                         f"header shape: {header.shape}, tensor shape: {tensor.shape}")
        return header_work, tensor_work
    
    def recv_header_and_tensor(self, src: int, tag_header: int, tag_tensor: int, header_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Receive a header and tensor pair.
        
        Args:
            src: Source rank
            tag_header: Tag for header
            tag_tensor: Tag for tensor
            header_len: Length of header tensor to receive
            
        Returns:
            Tuple of (header, tensor)
        """
        with CommunicationTimer(f"recv_header_and_tensor from rank {src}", self.logger):
            # First receive the header to get tensor shape (length can vary)
            header = torch.empty(header_len, dtype=torch.int64, device=self.device)
            dist.recv(header, src=src, tag=tag_header)
            
            # Parse header to get tensor shape
            chunk_idx, shape = self._parse_header(header)
            
            # Receive the tensor
            tensor = torch.empty(shape, dtype=torch.bfloat16, device=self.device)
            dist.recv(tensor, src=src, tag=tag_tensor)
        
        self.logger.debug(f"Received header+tensor from rank {src}, "
                         f"header: {header.tolist()}, tensor shape: {tensor.shape}")
        return header, tensor
    
    def send_latent_data_async(self, chunk_idx: int, latents: torch.Tensor,
                             original_latents: torch.Tensor, patched_x_shape: torch.Tensor,
                             current_start: torch.Tensor, current_end: torch.Tensor,
                             current_step: int) -> List[Any]:
        """
        Asynchronously send all latent data components.
        
        Args:
            chunk_idx: Chunk index
            latents: Latent tensor
            original_latents: Original latent tensor
            patched_x_shape: Patched x shape tensor
            current_start: Current start indices
            current_end: Current end indices
            current_step: Current step
            
        Returns:
            List of work objects for all send operations
        """
        dst = get_next_rank(self.rank, self.world_size)
        work_objects = []
        
        # Create headers
        latent_header = self._create_header(chunk_idx, latents.shape)
        origin_header = self._create_header(chunk_idx, original_latents.shape)
        
        # Create start/end/step tensor
        start_end_step = torch.cat([
            current_start, 
            current_end, 
            torch.tensor([current_step], dtype=torch.int64, device=self.device)
        ], dim=0)
        
        # Send all components asynchronously
        work_objects.extend(self.send_header_and_tensor_async(
            latent_header, latents, dst, CommunicationTags.LATENT_HDR, CommunicationTags.LATENT_PAY
        ))
        
        work_objects.extend(self.send_header_and_tensor_async(
            origin_header, original_latents, dst, 
            CommunicationTags.LATENT_ORIGIN_HDR, CommunicationTags.LATENT_ORIGIN_PAY
        ))
        
        work_objects.append(self.send_tensor_async(
            patched_x_shape, dst, CommunicationTags.PATCHED_X_SHAPE
        ))
        
        work_objects.append(self.send_tensor_async(
            start_end_step, dst, CommunicationTags.START_END_STEP
        ))
        
        self.logger.debug(f"Started async send of latent data to rank {dst}, chunk_idx: {chunk_idx}")
        return work_objects
    
    def recv_latent_data_async(self, num_steps: int, buffer_manager) -> Tuple[int, torch.Tensor, torch.Tensor, 
                                                                           torch.Tensor, torch.Tensor, int, torch.Tensor]:
        """
        Asynchronously receive all latent data components.
        
        Args:
            num_steps: Number of denoising steps
            buffer_manager: Buffer manager for tensor allocation
            
        Returns:
            Tuple of (chunk_idx, latents, original_latents, current_start, current_end, current_step, patched_x_shape)
        """
        src = get_prev_rank(self.rank, self.world_size)
        
        with CommunicationTimer(f"recv_latent_data_async from rank {src}", self.logger):
            # Receive latent header (length 4): [i, bsz, slen, cch]
            latent_header = buffer_manager.get_buffer((4,), torch.int64, "misc")
            dist.recv(latent_header, src=src, tag=CommunicationTags.LATENT_HDR)
            chunk_idx, latent_shape = self._parse_header(latent_header)
            # header no longer needed
            buffer_manager.return_buffer(latent_header, "misc")
            # Allocate or reuse buffer for latents: shape (bsz, slen, cch)
            latents = buffer_manager.get_buffer(tuple(latent_shape), torch.bfloat16, "latent")
            dist.recv(latents, src=src, tag=CommunicationTags.LATENT_PAY)

            # Receive original latent header (length 6): [i, bsz, cch, tlen, hh, ww]
            origin_header = buffer_manager.get_buffer((6,), torch.int64, "misc")
            dist.recv(origin_header, src=src, tag=CommunicationTags.LATENT_ORIGIN_HDR)
            _, origin_shape = self._parse_header(origin_header)
            # header no longer needed
            buffer_manager.return_buffer(origin_header, "misc")
            # Allocate or reuse buffer for original latents: shape (bsz, cch, tlen, hh, ww)
            original_latents = buffer_manager.get_buffer(tuple(origin_shape), torch.bfloat16, "origin")
            dist.recv(original_latents, src=src, tag=CommunicationTags.LATENT_ORIGIN_PAY)

            # Receive patched_x_shape (length 5, int64)
            patched_x_shape = buffer_manager.get_buffer((5,), torch.int64, "misc")
            dist.recv(patched_x_shape, src=src, tag=CommunicationTags.PATCHED_X_SHAPE)

            # Receive start_end_step (length 2*num_steps+1, int64)
            start_end_step = buffer_manager.get_buffer((2 * num_steps + 1,), torch.int64, "misc")
            dist.recv(start_end_step, src=src, tag=CommunicationTags.START_END_STEP)

            # Parse start/end/step into dedicated misc buffers, then release the combined vector
            current_start = buffer_manager.get_buffer((num_steps,), torch.int64, "misc")
            current_end = buffer_manager.get_buffer((num_steps,), torch.int64, "misc")
            current_start.copy_(start_end_step[:num_steps])
            current_end.copy_(start_end_step[num_steps:-1])
            current_step = int(start_end_step[-1].item())
            # Release the temporary combined buffer
            buffer_manager.return_buffer(start_end_step, "misc")
        
        self.logger.debug(f"Received latent data from rank {src}, chunk_idx: {chunk_idx}")
        return chunk_idx, latents, original_latents, current_start, current_end, current_step, patched_x_shape

    def send_prompt_async(self, prompt: str, device: torch.device) -> List[Any]:
        work_objects = []
        dst = get_next_rank(self.rank, self.world_size)

        # Encode to bytes
        encoded = prompt.encode("utf-8")
        data = torch.ByteTensor(list(encoded)).to(device)

        # Send length first
        length = torch.tensor([len(data)], dtype=torch.int64, device=data.device)
        work_objects.append(dist.isend(length, dst=dst, tag=CommunicationTags.UPDATED_PROMPT_LENGTH))

        # Then send the content
        work_objects.append(dist.isend(data, dst=dst, tag=CommunicationTags.UPDATED_PROMPT))

        return work_objects

    def recv_prompt_async(self) -> str:
        src = get_prev_rank(self.rank, self.world_size)

        # Receive length first
        length = torch.empty(1, dtype=torch.int64, device=self.device)
        dist.recv(length, src=src, tag=CommunicationTags.UPDATED_PROMPT_LENGTH)

        # Then receive the content
        prompt = torch.empty(length.item(), dtype=torch.uint8, device=self.device)
        dist.recv(prompt, src=src, tag=CommunicationTags.UPDATED_PROMPT)

        return bytes(prompt.cpu().tolist()).decode("utf-8")    
    
    def broadcast_tensor(self, tensor: torch.Tensor, src: int) -> None:
        """
        Broadcast a tensor from source to all ranks.
        
        Args:
            tensor: Tensor to broadcast
            src: Source rank
        """
        with CommunicationTimer(f"broadcast_tensor from rank {src}", self.logger):
            dist.broadcast(tensor, src=src)
        
        self.logger.debug(f"Broadcasted tensor from rank {src}, shape: {tensor.shape}")
    
    def all_gather_tensors(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """
        Gather tensors from all ranks.
        
        Args:
            tensor: Local tensor to gather
            
        Returns:
            List of tensors from all ranks
        """
        with CommunicationTimer("all_gather_tensors", self.logger):
            gather_list = [torch.zeros_like(tensor) for _ in range(self.world_size)]
            dist.all_gather(gather_list, tensor)
        
        self.logger.debug(f"Gathered tensors from all ranks, local shape: {tensor.shape}")
        return gather_list
    
    def wait_for_outstanding(self, max_outstanding: Optional[int] = None) -> None:
        """
        Wait for outstanding operations to complete.
        
        Args:
            max_outstanding: Maximum number of outstanding operations to keep
        """
        max_outstanding = max_outstanding or self.config.max_outstanding
        
        while len(self.outstanding_operations) >= max_outstanding:
            if not self.outstanding_operations:
                break
            
            # Wait for the oldest operation
            oldest_operations = self.outstanding_operations.pop(0)
            
            # Handle both single work objects and lists of work objects
            if isinstance(oldest_operations, (list, tuple)):
                for work in oldest_operations:
                    try:
                        work.wait()
                    except Exception as e:
                        self.logger.error(f"Error waiting for outstanding operation: {e}")
                        raise
            else:
                try:
                    oldest_operations.wait()
                except Exception as e:
                    self.logger.error(f"Error waiting for outstanding operation: {e}")
                    raise
        
        self.logger.debug(f"Outstanding operations: {len(self.outstanding_operations)}")
    
    def barrier(self) -> None:
        """Synchronize all ranks."""
        with CommunicationTimer("barrier", self.logger):
            dist.barrier()
    
    def _create_header(self, chunk_idx: int, shape: Tuple[int, ...]) -> torch.Tensor:
        """Create a header tensor for communication."""
        header_data = [chunk_idx] + list(shape)
        return torch.tensor(header_data, dtype=torch.int64, device=self.device)
    
    def _parse_header(self, header: torch.Tensor) -> Tuple[int, Tuple[int, ...]]:
        """Parse a header tensor to extract metadata."""
        header_list = header.tolist()
        chunk_idx = int(header_list[0])
        shape = tuple(int(x) for x in header_list[1:])
        return chunk_idx, shape
    
    def get_statistics(self) -> dict:
        """Get communication statistics."""
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "outstanding_operations": len(self.outstanding_operations),
            "max_outstanding": self.config.max_outstanding,
            "device": str(self.device)
        }
    
    def print_statistics(self) -> None:
        """Print communication statistics."""
        stats = self.get_statistics()
        self.logger.info("Distributed Communicator Statistics:")
        for key, value in stats.items():
            self.logger.info(f"  {key}: {value}")
````

## File: streamv2v/communication/kv_cache_manager.py

````python
"""
KV Cache management for distributed inference.

This module provides functionality for managing and rebalancing KV caches
across distributed ranks during inference.
"""

import torch
import torch.distributed as dist
from typing import List, Dict, Tuple, Optional
import logging
from .utils import CommunicationTags, CommunicationTimer
from .data_containers import KVCacheData, BlockInterval


class KVCacheManager:
    """
    Manages KV cache operations for distributed inference.
    
    This class handles KV cache broadcasting, rebalancing, and ownership
    management across distributed ranks.
    """
    
    def __init__(self, pipeline, device: torch.device):
        """
        Initialize the KV cache manager.
        
        Args:
            pipeline: The inference pipeline containing KV caches
            device: GPU device for operations
        """
        self.pipeline = pipeline
        self.device = device
        self.frame_seq_length = pipeline.frame_seq_length
        self.time_step_length = len(pipeline.denoising_step_list)
        
        # Setup logging
        self.logger = logging.getLogger(f"KVCacheManager_{device}")
        self.logger.propagate = False
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'[KVCacheManager {device}] %(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def broadcast_kv_blocks(self, block_indices: List[int], donor_rank: int) -> None:
        """
        Broadcast kv_cache1 entries for the specified block indices from donor_rank to all ranks.
        
        This ensures the receiver rank has the up-to-date KV cache when ownership moves.
        
        Args:
            block_indices: List of block indices to broadcast
            donor_rank: Rank that owns the KV cache data
        """
        if len(block_indices) == 0:
            return
        
        rank = dist.get_rank()
        
        with CommunicationTimer(f"broadcast_kv_blocks from rank {donor_rank}", self.logger):
            for bi in block_indices:
                # Broadcast key cache
                if self.pipeline.kv_cache1[bi]['k'].device != self.device:
                    self.pipeline.kv_cache1[bi]['k'] = self.pipeline.kv_cache1[bi]['k'].to(self.device)
                    self.pipeline.kv_cache1[bi]['v'] = self.pipeline.kv_cache1[bi]['v'].to(self.device)
                
                dist.barrier()

                dist.broadcast(self.pipeline.kv_cache1[bi]['k'], src=donor_rank)
                # Broadcast value cache
                dist.broadcast(self.pipeline.kv_cache1[bi]['v'], src=donor_rank)
                # Broadcast global end index
                dist.broadcast(self.pipeline.kv_cache1[bi]['global_end_index'], src=donor_rank)
                # Broadcast local end index
                dist.broadcast(self.pipeline.kv_cache1[bi]['local_end_index'], src=donor_rank)
                
                # Adjust global_end_index for the receiving rank
                if donor_rank > rank:
                    self.pipeline.kv_cache1[bi]['global_end_index'] += self.frame_seq_length * (donor_rank - rank) * self.time_step_length
        
        self.logger.debug(f"Broadcasted KV cache for blocks {block_indices} from rank {donor_rank}")
    
    def compute_block_owners(self, block_intervals: torch.Tensor, total_blocks: int) -> torch.Tensor:
        """
        Given block intervals in [start, end) format for all ranks, return a tensor
        where each entry is the owner rank of that block index.
        
        Args:
            block_intervals: Block intervals for all ranks [world_size, 2]
            total_blocks: Total number of blocks
            
        Returns:
            Tensor of length total_blocks with owner ranks
        """
        world_size = block_intervals.shape[0]
        owners = torch.full((total_blocks,), -1, dtype=torch.int64, device=block_intervals.device)
        
        for r in range(world_size):
            s = int(block_intervals[r, 0].item())
            e = int(block_intervals[r, 1].item())
            if e > s:
                owners[s:e] = r
        
        self.logger.debug(f"Computed block owners: {owners.tolist()}")
        return owners
    
    def rebalance_kv_cache_by_diff(self, old_block_intervals: torch.Tensor, 
                                  new_block_intervals: torch.Tensor, total_blocks: int) -> None:
        """
        Compare ownership from old to new intervals and broadcast KV cache for blocks whose owner changes.
        
        For each moved block i, use the previous owner's rank as src to broadcast
        pipeline.kv_cache1[i]['k'/'v'/...] to all ranks so the new owner has the correct state.
        
        Args:
            old_block_intervals: Previous block intervals [world_size, 2]
            new_block_intervals: New block intervals [world_size, 2]
            total_blocks: Total number of blocks
        """
        with CommunicationTimer("rebalance_kv_cache_by_diff", self.logger):
            old_owners = self.compute_block_owners(old_block_intervals, total_blocks)
            new_owners = self.compute_block_owners(new_block_intervals, total_blocks)
            
            # Find blocks that changed ownership
            moved_by_src = {}
            for i in range(total_blocks):
                o = int(old_owners[i].item())
                n = int(new_owners[i].item())
                if o != n and o >= 0:
                    if o not in moved_by_src:
                        moved_by_src[o] = []
                    moved_by_src[o].append(i)
            
            # Synchronize before broadcasting
            dist.barrier()
            
            # Broadcast per donor rank (can batch multiple blocks per src)
            for src, blocks in moved_by_src.items():
                self.broadcast_kv_blocks(blocks, donor_rank=src)
        
        self.logger.info(f"Rebalanced KV cache: {len(moved_by_src)} ranks had ownership changes")
    
    def get_kv_cache_statistics(self, block_intervals: torch.Tensor, total_blocks: int) -> Dict[str, any]:
        """
        Get statistics about KV cache distribution.
        
        Args:
            block_intervals: Current block intervals [world_size, 2]
            total_blocks: Total number of blocks
            
        Returns:
            Dictionary containing KV cache statistics
        """
        owners = self.compute_block_owners(block_intervals, total_blocks)
        
        # Count blocks per rank
        block_counts = {}
        for rank in range(block_intervals.shape[0]):
            block_counts[rank] = int((owners == rank).sum().item())
        
        # Calculate memory usage per rank (approximate)
        memory_per_block = 0
        if hasattr(self.pipeline, 'kv_cache1') and len(self.pipeline.kv_cache1) > 0:
            # Estimate memory per block based on first block
            first_block = self.pipeline.kv_cache1[0]
            if 'k' in first_block and 'v' in first_block:
                k_memory = first_block['k'].numel() * first_block['k'].element_size()
                v_memory = first_block['v'].numel() * first_block['v'].element_size()
                memory_per_block = k_memory + v_memory
        
        memory_usage = {rank: block_counts[rank] * memory_per_block for rank in block_counts}
        
        return {
            "block_counts": block_counts,
            "memory_usage_bytes": memory_usage,
            "total_blocks": total_blocks,
            "memory_per_block_bytes": memory_per_block,
            "frame_seq_length": self.frame_seq_length
        }
    
    def print_kv_cache_statistics(self, block_intervals: torch.Tensor, total_blocks: int) -> None:
        """
        Print KV cache statistics.
        
        Args:
            block_intervals: Current block intervals [world_size, 2]
            total_blocks: Total number of blocks
        """
        stats = self.get_kv_cache_statistics(block_intervals, total_blocks)
        
        self.logger.info("KV Cache Statistics:")
        self.logger.info(f"  Total blocks: {stats['total_blocks']}")
        self.logger.info(f"  Memory per block: {stats['memory_per_block_bytes']} bytes")
        self.logger.info(f"  Frame sequence length: {stats['frame_seq_length']}")
        
        self.logger.info("  Block distribution:")
        for rank, count in stats['block_counts'].items():
            memory_mb = stats['memory_usage_bytes'][rank] / (1024 * 1024)
            self.logger.info(f"    Rank {rank}: {count} blocks, {memory_mb:.2f} MB")
    
    def validate_kv_cache_consistency(self, block_intervals: torch.Tensor, total_blocks: int) -> bool:
        """
        Validate that KV cache ownership is consistent with block intervals.
        
        Args:
            block_intervals: Current block intervals [world_size, 2]
            total_blocks: Total number of blocks
            
        Returns:
            True if consistent, False otherwise
        """
        owners = self.compute_block_owners(block_intervals, total_blocks)
        
        # Check that all blocks have owners
        unowned_blocks = (owners == -1).sum().item()
        if unowned_blocks > 0:
            self.logger.error(f"Found {unowned_blocks} unowned blocks")
            return False
        
        # Check that block intervals are contiguous and non-overlapping
        for rank in range(block_intervals.shape[0]):
            start = int(block_intervals[rank, 0].item())
            end = int(block_intervals[rank, 1].item())
            
            if start < 0 or end > total_blocks or start >= end:
                self.logger.error(f"Invalid block interval for rank {rank}: [{start}, {end})")
                return False
            
            # Check that all blocks in this interval are owned by this rank
            for block_idx in range(start, end):
                if int(owners[block_idx].item()) != rank:
                    self.logger.error(f"Block {block_idx} not owned by rank {rank}")
                    return False
        
        self.logger.debug("KV cache consistency validation passed")
        return True
    
    def cleanup_kv_cache(self, block_intervals: torch.Tensor, total_blocks: int) -> None:
        """
        Clean up KV cache for blocks not owned by current rank.
        
        Args:
            block_intervals: Current block intervals [world_size, 2]
            total_blocks: Total number of blocks
        """
        rank = dist.get_rank()
        owners = self.compute_block_owners(block_intervals, total_blocks)
        
        cleaned_blocks = 0
        for block_idx in range(total_blocks):
            if int(owners[block_idx].item()) != rank:
                # Clear KV cache for blocks not owned by this rank
                if hasattr(self.pipeline, 'kv_cache1') and block_idx < len(self.pipeline.kv_cache1):
                    if 'k' in self.pipeline.kv_cache1[block_idx]:
                        self.pipeline.kv_cache1[block_idx]['k'].zero_()
                    if 'v' in self.pipeline.kv_cache1[block_idx]:
                        self.pipeline.kv_cache1[block_idx]['v'].zero_()
                    cleaned_blocks += 1
        
        self.logger.info(f"Cleaned up KV cache for {cleaned_blocks} blocks not owned by rank {rank}")
````

## File: streamv2v/communication/model_data_transfer.py

````python
"""
Model data transfer abstraction layer.

This module provides high-level interfaces for transferring model data
between distributed ranks during inference.
"""

import torch
from typing import List, Tuple, Optional, Any
import logging
from .distributed_communicator import DistributedCommunicator
from .buffer_manager import BufferManager
from .kv_cache_manager import KVCacheManager
from .data_containers import LatentData, CommunicationConfig, PerformanceMetrics
from .utils import CommunicationTimer


class ModelDataTransfer:
    """
    High-level interface for model data transfer operations.
    
    This class encapsulates all model-related data transfer operations,
    providing a clean interface for sending and receiving latent data,
    KV caches, and other model state between ranks.
    """
    
    def __init__(self, communicator: DistributedCommunicator, 
                 buffer_manager: BufferManager,
                 kv_cache_manager: Optional[KVCacheManager] = None,
                 config: Optional[CommunicationConfig] = None):
        """
        Initialize the model data transfer manager.
        
        Args:
            communicator: Distributed communicator instance
            buffer_manager: Buffer manager for tensor allocation
            kv_cache_manager: KV cache manager (optional)
            config: Communication configuration
        """
        self.comm = communicator
        self.buffer_mgr = buffer_manager
        self.kv_cache_mgr = kv_cache_manager
        self.config = config or CommunicationConfig()
        
        # Setup logging
        self.logger = logging.getLogger(f"ModelDataTransfer_rank_{communicator.rank}")
        self.logger.propagate = False
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'[Rank {communicator.rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Performance tracking
        self.transfer_count = 0
        self.total_transfer_time = 0.0
    
    def send_latent_data_async(self, chunk_idx: int, latents: torch.Tensor,
                             original_latents: torch.Tensor, patched_x_shape: torch.Tensor,
                             current_start: torch.Tensor, current_end: torch.Tensor,
                             current_step: int) -> List[Any]:
        """
        Asynchronously send latent data to the next rank.
        
        Args:
            chunk_idx: Chunk index
            latents: Latent tensor
            original_latents: Original latent tensor
            patched_x_shape: Patched x shape tensor
            current_start: Current start indices
            current_end: Current end indices
            current_step: Current step
            
        Returns:
            List of work objects for all send operations
        """
        with CommunicationTimer(f"send_latent_data_async chunk_{chunk_idx}", self.logger):
            work_objects = self.comm.send_latent_data_async(
                chunk_idx=chunk_idx,
                latents=latents,
                original_latents=original_latents,
                patched_x_shape=patched_x_shape,
                current_start=current_start,
                current_end=current_end,
                current_step=current_step
            )
        
        self.transfer_count += 1
        self.logger.debug(f"Sent latent data for chunk {chunk_idx}")
        return work_objects
    
    def receive_latent_data_async(self, num_steps: int) -> LatentData:
        """
        Asynchronously receive latent data from the previous rank.
        
        Args:
            num_steps: Number of denoising steps
            
        Returns:
            LatentData object containing all received data
        """
        with CommunicationTimer("receive_latent_data_async", self.logger):
            chunk_idx, latents, original_latents, current_start, current_end, current_step, patched_x_shape = \
                self.comm.recv_latent_data_async(num_steps, self.buffer_mgr)
        
        self.transfer_count += 1
        self.logger.debug(f"Received latent data for chunk {chunk_idx}")
        
        return LatentData(
            chunk_idx=chunk_idx,
            latents=latents,
            original_latents=original_latents,
            current_start=current_start,
            current_end=current_end,
            current_step=current_step,
            patched_x_shape=patched_x_shape
        )

    def send_prompt_async(self, prompt: str, device: torch.device) -> List[Any]:
        return self.comm.send_prompt_async(prompt, device)

    def recv_prompt_async(self) -> str:
        return self.comm.recv_prompt_async()
    
    def send_kv_cache_blocks(self, block_indices: List[int], donor_rank: int) -> None:
        """
        Send KV cache blocks to all ranks.
        
        Args:
            block_indices: List of block indices to send
            donor_rank: Rank that owns the KV cache data
        """
        if self.kv_cache_mgr is None:
            raise RuntimeError("KV cache manager not initialized")
        
        with CommunicationTimer(f"send_kv_cache_blocks {len(block_indices)} blocks", self.logger):
            self.kv_cache_mgr.broadcast_kv_blocks(block_indices, donor_rank)
        
        self.logger.debug(f"Sent KV cache blocks {block_indices} from rank {donor_rank}")
    
    def rebalance_kv_cache(self, old_intervals: torch.Tensor, 
                          new_intervals: torch.Tensor, total_blocks: int) -> None:
        """
        Rebalance KV cache ownership based on new block intervals.
        
        Args:
            old_intervals: Previous block intervals [world_size, 2]
            new_intervals: New block intervals [world_size, 2]
            total_blocks: Total number of blocks
        """
        if self.kv_cache_mgr is None:
            raise RuntimeError("KV cache manager not initialized")
        
        with CommunicationTimer("rebalance_kv_cache", self.logger):
            self.kv_cache_mgr.rebalance_kv_cache_by_diff(old_intervals, new_intervals, total_blocks)
        
        self.logger.info("Rebalanced KV cache ownership")
    
    def broadcast_tensor(self, tensor: torch.Tensor, src: int) -> None:
        """
        Broadcast a tensor from source to all ranks.
        
        Args:
            tensor: Tensor to broadcast
            src: Source rank
        """
        with CommunicationTimer(f"broadcast_tensor from rank {src}", self.logger):
            self.comm.broadcast_tensor(tensor, src)
        
        self.logger.debug(f"Broadcasted tensor from rank {src}, shape: {tensor.shape}")
    
    def all_gather_tensors(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """
        Gather tensors from all ranks.
        
        Args:
            tensor: Local tensor to gather
            
        Returns:
            List of tensors from all ranks
        """
        with CommunicationTimer("all_gather_tensors", self.logger):
            gather_list = self.comm.all_gather_tensors(tensor)
        
        self.logger.debug(f"Gathered tensors from all ranks, local shape: {tensor.shape}")
        return gather_list
    
    def wait_for_outstanding(self, max_outstanding: Optional[int] = None) -> None:
        """
        Wait for outstanding operations to complete.
        
        Args:
            max_outstanding: Maximum number of outstanding operations to keep
        """
        with CommunicationTimer("wait_for_outstanding", self.logger):
            self.comm.wait_for_outstanding(max_outstanding)
    
    def barrier(self) -> None:
        """Synchronize all ranks."""
        with CommunicationTimer("barrier", self.logger):
            self.comm.barrier()
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """
        Get performance metrics for data transfer operations.
        
        Returns:
            PerformanceMetrics object containing timing information
        """
        # This is a simplified version - in practice, you'd want to track
        # more detailed timing information
        avg_transfer_time = self.total_transfer_time / max(1, self.transfer_count)
        
        return PerformanceMetrics(
            dit_time=0.0,  # Would be filled by caller
            total_time=0.0,  # Would be filled by caller
            communication_time=avg_transfer_time,
            buffer_allocation_time=0.0  # Would be tracked by buffer manager
        )
    
    def get_statistics(self) -> dict:
        """
        Get transfer statistics.
        
        Returns:
            Dictionary containing transfer statistics
        """
        return {
            "transfer_count": self.transfer_count,
            "total_transfer_time": self.total_transfer_time,
            "avg_transfer_time": self.total_transfer_time / max(1, self.transfer_count),
            "communicator_stats": self.comm.get_statistics(),
            "buffer_manager_stats": self.buffer_mgr.get_statistics() if self.buffer_mgr else None
        }
    
    def print_statistics(self) -> None:
        """Print transfer statistics."""
        stats = self.get_statistics()
        self.logger.info("Model Data Transfer Statistics:")
        for key, value in stats.items():
            if key == "communicator_stats" or key == "buffer_manager_stats":
                if value:
                    self.logger.info(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        self.logger.info(f"    {sub_key}: {sub_value}")
            else:
                self.logger.info(f"  {key}: {value}")
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.buffer_mgr:
            self.buffer_mgr.clear_buffers()
        self.logger.info("Model data transfer cleanup completed")
    
    def __del__(self):
        """Cleanup when the transfer manager is destroyed."""
        self.cleanup()
````

## File: streamv2v/communication/utils.py

````python
"""
Utility functions and constants for communication operations.

This module provides utility functions and constants used across the communication module.
"""

import torch
import torch.distributed as dist
from typing import List, Tuple, Optional
import time
import logging

# Communication tags for different types of data
class CommunicationTags:
    """Constants for communication tags."""
    LATENT_HDR = 11001
    LATENT_PAY = 11002
    START_END_STEP = 11003
    PATCHED_X_SHAPE = 11004
    LATENT_ORIGIN_HDR = 11005
    LATENT_ORIGIN_PAY = 11006
    KV_CACHE_K = 11007
    KV_CACHE_V = 11008
    KV_CACHE_GLOBAL_END = 11009
    KV_CACHE_LOCAL_END = 11010
    BLOCK_INTERVALS = 11011
    PERFORMANCE_METRICS = 11012
    UPDATED_PROMPT_LENGTH = 11013
    UPDATED_PROMPT = 11014


def init_distributed():
    """
    Initialize distributed communication.
    
    This function initializes the distributed process group if not already initialized.
    """
    if not dist.is_initialized():
        backend = "nccl"
        dist.init_process_group(backend=backend)


def get_rank_info() -> Tuple[int, int]:
    """
    Get current rank and world size.
    
    Returns:
        Tuple of (rank, world_size)
    """
    if not dist.is_initialized():
        raise RuntimeError("Distributed not initialized")
    return dist.get_rank(), dist.get_world_size()


def get_next_rank(rank: int, world_size: int) -> int:
    """
    Get the next rank in the ring topology.
    
    Args:
        rank: Current rank
        world_size: Total number of ranks
        
    Returns:
        Next rank in the ring
    """
    return (rank + 1) % world_size


def get_prev_rank(rank: int, world_size: int) -> int:
    """
    Get the previous rank in the ring topology.
    
    Args:
        rank: Current rank
        world_size: Total number of ranks
        
    Returns:
        Previous rank in the ring
    """
    return (rank - 1) % world_size


def create_tensor_header(shape: Tuple[int, ...], dtype: torch.dtype, 
                        chunk_idx: int, device: torch.device) -> torch.Tensor:
    """
    Create a header tensor for communication.
    
    Args:
        shape: Shape of the tensor to be sent
        dtype: Data type of the tensor
        chunk_idx: Chunk index
        device: Device where the header will be created
        
    Returns:
        Header tensor containing metadata
    """
    header_data = [chunk_idx] + list(shape)
    return torch.tensor(header_data, dtype=torch.int64, device=device)


def parse_tensor_header(header: torch.Tensor) -> Tuple[int, Tuple[int, ...]]:
    """
    Parse a header tensor to extract metadata.
    
    Args:
        header: Header tensor
        
    Returns:
        Tuple of (chunk_idx, shape)
    """
    header_list = header.tolist()
    chunk_idx = int(header_list[0])
    shape = tuple(int(x) for x in header_list[1:])
    return chunk_idx, shape


def validate_tensor_for_communication(tensor: torch.Tensor, 
                                    expected_device: torch.device,
                                    expected_dtype: torch.dtype) -> None:
    """
    Validate tensor properties for communication.
    
    Args:
        tensor: Tensor to validate
        expected_device: Expected device
        expected_dtype: Expected data type
        
    Raises:
        ValueError: If tensor properties don't match expectations
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Input must be a torch.Tensor")
    
    if tensor.device != expected_device:
        raise ValueError(f"Tensor device {tensor.device} doesn't match expected {expected_device}")
    
    if tensor.dtype != expected_dtype:
        raise ValueError(f"Tensor dtype {tensor.dtype} doesn't match expected {expected_dtype}")


def compute_balanced_split(total_blocks: int, rank_times: List[float], 
                          dit_times: List[float], 
                          current_block_nums: List[List[int]]) -> List[List[int]]:
    """
    Compute new block splits for all ranks to balance total rank times.
    
    This function is moved from the original file to provide better organization.
    
    Args:
        total_blocks: Total number of DiT blocks
        rank_times: List of total iteration times for each rank [t_rank0, t_rank1, ..., t_rankN] (DiT + VAE time)
        dit_times: List of pure DiT inference times for each rank [dit_rank0, dit_rank1, ..., dit_rankN] (DiT time only)
        current_block_nums: List of current block_num format for each rank [[rank0_blocks], [rank1_blocks], ...]
        
    Returns:
        List of new block_num format for each rank, matching the original format:
        - For world_size == 2: [[end_idx_rank0], [start_idx_rank1]]
        - For world_size > 2: [[end_idx_rank0], [start1, end1], [start2, end2], ..., [start_idx_last]]
        Note: Numbers are shared across ranks (rank0_end = rank1_start, rank1_end = rank2_start, etc.)
    """
    num_ranks = len(rank_times)
    if num_ranks == 0 or num_ranks != len(current_block_nums) or num_ranks != len(dit_times):
        return current_block_nums
    
    # Edge case: if we have more ranks than blocks, we can't guarantee 1 block per rank
    if num_ranks > total_blocks:
        # Fall back to original behavior for this edge case
        return current_block_nums

    # Step 1: Calculate total DiT time and per-block DiT time
    total_dit_time = sum(dit_times)
    dit_time_per_block = total_dit_time / total_blocks
    
    # Step 2: Calculate average rank time
    avg_rank_time = sum(rank_times) / num_ranks
    
    # Step 3: Extract current block counts from current_block_nums (all ranks use [start, end) now)
    current_block_counts = []
    for block_num in current_block_nums:
        # block_num: [start, end) exclusive end
        start_idx, end_idx = int(block_num[0]), int(block_num[1])
        current_block_counts.append(max(0, end_idx - start_idx))
    
    # Step 4: Calculate target block counts based on time differences
    target_blocks = []
    for i in range(num_ranks):
        time_diff = avg_rank_time - rank_times[i]  # positive = needs more time, negative = needs less time
        block_adjustment = time_diff / dit_time_per_block  # convert time difference to block count
        target_count = current_block_counts[i] + block_adjustment
        # Ensure each rank gets at least 1 block (minimum allocation)
        target_count = max(1, int(round(target_count)))
        target_blocks.append(target_count)
    
    # Step 5: Adjust to ensure total blocks sum to total_blocks while maintaining minimum 1 block per rank
    current_total = sum(target_blocks)
    if current_total != total_blocks:
        diff = total_blocks - current_total
        # When adding, give to ranks with smallest counts first; when removing, take from largest counts first
        if diff > 0:
            order = sorted(range(num_ranks), key=lambda i: (target_blocks[i], i))
        else:
            order = sorted(range(num_ranks), key=lambda i: (target_blocks[i], i), reverse=True)
        i = 0
        while diff != 0 and num_ranks > 0:
            idx = order[i % num_ranks]
            if diff > 0:
                target_blocks[idx] += 1
                diff -= 1
            else:
                # Only remove blocks if rank has more than 1 block (maintain minimum allocation)
                if target_blocks[idx] > 1:
                    target_blocks[idx] -= 1
                    diff += 1
            i += 1
    
    # Step 6: Convert target block counts to contiguous [start, end) intervals from 0 to total_blocks
    new_block_nums = []
    running_start = 0
    for i in range(num_ranks):
        block_count = int(target_blocks[i])
        start_idx = running_start
        end_idx = start_idx + block_count
        # Guard (should not trigger if sums are correct)
        if end_idx > total_blocks:
            end_idx = total_blocks
        new_block_nums.append([start_idx, end_idx])
        running_start = end_idx
    
    return new_block_nums


def setup_logging(rank: int, log_level: int = logging.INFO) -> logging.Logger:
    """
    Setup logging for the current rank.
    
    Args:
        rank: Current rank
        log_level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(f"rank_{rank}")
    logger.setLevel(log_level)
    # Prevent messages from propagating to the root logger (avoid double prints)
    logger.propagate = False
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            f'[Rank {rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


class CommunicationTimer:
    """
    Timer for measuring communication performance.
    
    This class provides context manager functionality for timing communication operations.
    """
    
    def __init__(self, operation_name: str, logger: Optional[logging.Logger] = None):
        self.operation_name = operation_name
        self.logger = logger
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        if self.logger:
            self.logger.info(f"{self.operation_name} took {duration:.4f} seconds")
    
    @property
    def duration(self) -> float:
        """Get the duration of the timed operation."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time
````

## File: streamv2v/inference.py

````python
"""
Single GPU Inference Pipeline - Refactored from inference_pipe.py

This file extracts core logic from multi-GPU inference code to implement a complete 
inference pipeline on a single GPU:
1. VAE encode input video
2. DiT inference (using input mode, processing all 30 blocks)
3. VAE decode output video
"""

from causvid.models.wan.causal_stream_inference import CausalStreamInferencePipeline
from diffusers.utils import export_to_video
from causvid.data import TextDataset
from omegaconf import OmegaConf
import argparse
import torch
import os
import time
import numpy as np
import logging

import torchvision
import torchvision.transforms.functional as TF
from einops import rearrange


def load_mp4_as_tensor(
    video_path: str,
    max_frames: int = None,
    resize_hw: tuple[int, int] = None,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Load an .mp4 video and return it as a PyTorch tensor with shape [C, T, H, W].

    Args:
        video_path (str): Path to the input .mp4 video file
        max_frames (int, optional): Maximum number of frames to load. If None, loads all frames
        resize_hw (tuple, optional): Target (height, width) to resize each frame. If None, no resizing
        normalize (bool, optional): Whether to normalize pixel values to [-1, 1]

    Returns:
        torch.Tensor: Tensor of shape [C, T, H, W], dtype=torch.float32
    """
    assert os.path.exists(video_path), f"Video file not found: {video_path}"

    video, _, _ = torchvision.io.read_video(video_path, output_format="TCHW")
    if max_frames is not None:
        video = video[:max_frames]

    video = rearrange(video, "t c h w -> c t h w")
    if resize_hw is not None:
        c, t, h0, w0 = video.shape
        video = torch.stack([
            TF.resize(video[:, i], resize_hw, antialias=True)
            for i in range(t)
        ], dim=1)
    if video.dtype != torch.float32:
        video = video.float()
    if normalize:
        video = video / 127.5 - 1.0

    return video  # [C, T, H, W]

def compute_noise_scale_and_step(input_video_original: torch.Tensor, end_idx: int, chunk_size: int, noise_scale: float, init_noise_scale: float):
    """Compute adaptive noise scale and current step based on video content."""
    l2_dist=(input_video_original[:,:,end_idx-chunk_size:end_idx]-input_video_original[:,:,end_idx-chunk_size-1:end_idx-1])**2
    l2_dist = (torch.sqrt(l2_dist.mean(dim=(0,1,3,4))).max()/0.2).clamp(0,1)
    new_noise_scale = (init_noise_scale-0.1*l2_dist.item())*0.9+noise_scale*0.1
    current_step = int(1000*new_noise_scale)-100
    return new_noise_scale, current_step

class SingleGPUInferencePipeline:
    """
    Single GPU Inference Pipeline Manager
    
    This class encapsulates the complete inference logic on a single GPU, 
    including encoding, inference, and decoding.
    """
    
    def __init__(self, config, device: torch.device):
        """
        Initialize the single GPU inference pipeline manager.
        
        Args:
            config: Configuration object
            device: GPU device
        """
        self.config = config
        self.device = device
        
        # Setup logging
        self.logger = logging.getLogger("SingleGPUInference")
        self.logger.setLevel(logging.INFO)
        # Prevent messages from propagating to the root logger (avoid double prints)
        self.logger.propagate = False
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Initialize pipeline
        self.pipeline = CausalStreamInferencePipeline(config, device=str(device))
        self.pipeline.to(device=str(device), dtype=torch.bfloat16)
        
        # Performance tracking
        self.t_dit = 100.0
        self.t_total = 100.0
        self.processed = 0
        self.processed_offset = 3
        self.base_chunk_size = 4
        self.t_refresh = 50

        
        self.logger.info("Single GPU inference pipeline manager initialized")
    
    def load_model(self, checkpoint_folder: str):
        """Load the model from checkpoint."""
        ckpt_path = os.path.join(checkpoint_folder, "model.pt")
        self.logger.info(f"Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")

        # Decide which key holds the generator state dict
        if isinstance(ckpt, dict):
            if 'generator' in ckpt:
                state_dict = ckpt['generator']
            elif 'generator_ema' in ckpt:
                state_dict = ckpt['generator_ema']
            elif 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            else:
                # Assume the checkpoint itself is a state dict
                state_dict = ckpt
        else:
            state_dict = ckpt

        # Load into the pipeline generator
        try:
            self.pipeline.generator.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            # Try non-strict load as a fallback and report
            self.logger.warning(f"Strict load_state_dict failed: {e}; retrying with strict=False")
            self.pipeline.generator.load_state_dict(state_dict, strict=False)
    
    def prepare_pipeline(self, text_prompts: list, noise: torch.Tensor, 
                        current_start: int, current_end: int):
        """Prepare the pipeline for inference."""
        # Use the original prepare method which now handles distributed environment gracefully
        denoised_pred = self.pipeline.prepare(
            text_prompts=text_prompts,
            device=self.device,
            dtype=torch.bfloat16,
            block_mode='input',
            noise=noise,
            current_start=current_start,
            current_end=current_end
        )
        return denoised_pred
    
    def run_inference(
        self, 
        input_video_original: torch.Tensor, 
        prompts: list, 
        num_chunks: int, 
        chunk_size: int, 
        noise_scale: float, 
        output_folder: str, 
        fps: int, 
        target_fps:int,  
        num_steps: int,
        ):
        """
        Run the complete single GPU inference pipeline.
        
        This method integrates the complete encoding, inference, and decoding pipeline.
        """
        self.logger.info("Starting single GPU inference pipeline")
        
        os.makedirs(output_folder, exist_ok=True)
        results = {}
        save_results = 0

        fps_list = []
        dit_fps_list = []
        
        # Initialize variables
        start_idx = 0
        end_idx = 1 + chunk_size
        current_start = 0
        current_end = self.pipeline.frame_seq_length * (1+chunk_size//4)
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        # Process first chunk (initialization)
        if input_video_original is not None:
            inp = input_video_original[:, :, start_idx:end_idx]
            
            # VAE encoding
            latents = self.pipeline.vae.stream_encode(inp)
            latents = latents.transpose(2, 1).contiguous().to(dtype=torch.bfloat16)
            
            noise = torch.randn_like(latents)
            noisy_latents = noise * noise_scale + latents * (1 - noise_scale)
        else:
            noisy_latents = torch.randn(1,1+self.pipeline.num_frame_per_block,16,self.pipeline.height,self.pipeline.width, device=self.device, dtype=torch.bfloat16)
        
            
        # Prepare pipeline
        denoised_pred = self.prepare_pipeline(
            text_prompts=prompts,
            noise=noisy_latents,
            current_start=current_start,
            current_end=current_end
        )
        
        # Save first result - only start decoding after num_steps
        video = self.pipeline.vae.stream_decode_to_pixel(denoised_pred)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        video = video[0].permute(0, 2, 3, 1).contiguous()
        results[save_results] = video.cpu().float().numpy()
        save_results += 1
        
        init_noise_scale = noise_scale
        
        # Process remaining chunks
        while self.processed < num_chunks + num_steps - 1:
            # Update indices
            start_idx = end_idx
            end_idx = end_idx + chunk_size
            current_start = current_end
            current_end = current_end + (chunk_size // 4) * self.pipeline.frame_seq_length

            if input_video_original is not None and end_idx <= input_video_original.shape[2]:
                inp = input_video_original[:, :, start_idx:end_idx]
                
                noise_scale, current_step = compute_noise_scale_and_step(
                    input_video_original, end_idx, chunk_size, noise_scale, init_noise_scale
                )
                
                # VAE encoding
                latents = self.pipeline.vae.stream_encode(inp)
                latents = latents.transpose(2, 1).contiguous().to(dtype=torch.bfloat16)
                
                noise = torch.randn_like(latents)
                noisy_latents = noise * noise_scale + latents * (1 - noise_scale)
            else:
                noisy_latents = torch.randn(1,self.pipeline.num_frame_per_block,16,self.pipeline.height,self.pipeline.width, device=self.device, dtype=torch.bfloat16)
                current_step = None # Use default steps

            # if current_start//self.pipeline.frame_seq_length >= self.t_refresh:
            #     current_start = self.pipeline.kv_cache_length - self.pipeline.frame_seq_length
            #     current_end = current_start + (chunk_size // 4) * self.pipeline.frame_seq_length
            
            torch.cuda.synchronize()
            dit_start_time = time.time()
                
            # DiT inference - using input mode to process all 30 blocks
            denoised_pred = self.pipeline.inference_stream(
                noise=noisy_latents,
                current_start=current_start,
                current_end=current_end,
                current_step=current_step,
            )

            if self.processed > self.processed_offset:
                torch.cuda.synchronize()
                dit_fps_list.append(chunk_size/(time.time()-dit_start_time))
            
            self.processed += 1
            
            # VAE decoding - only start decoding after num_steps
            if self.processed >= num_steps:
                video = self.pipeline.vae.stream_decode_to_pixel(denoised_pred[[-1]])
                video = (video * 0.5 + 0.5).clamp(0, 1)
                video = video[0].permute(0, 2, 3, 1).contiguous()
                
                results[save_results] = video.cpu().float().numpy()
                save_results += 1
            
                # Update timing
                torch.cuda.synchronize()
                end_time = time.time()
                t = end_time - start_time
                fps_test = chunk_size/t
                fps_list.append(fps_test)
                self.logger.info(f"Processed {self.processed}, time: {t:.4f} s, FPS: {fps_test:.4f}")

                if self.processed==num_steps+self.processed_offset and target_fps is not None and fps_test<target_fps:
                    max_chunk_size = (self.pipeline.num_kv_cache - self.pipeline.num_sink_tokens - 1) * self.base_chunk_size
                    num_chunks=(num_chunks-self.processed-num_steps+1)//(max_chunk_size//chunk_size)+self.processed-num_steps+1
                    self.pipeline.hidden_states=self.pipeline.hidden_states.repeat(1,max_chunk_size//chunk_size,1,1,1)
                    chunk_size = max_chunk_size
                    self.logger.info(f"Adjust chunk size to {chunk_size}")

                start_time = end_time
        
        # Save final video
        video_list = [results[i] for i in range(num_chunks)]
        video = np.concatenate(video_list, axis=0)
        fps_avg = np.mean(np.array(fps_list))
        self.logger.info(f"DiT Average FPS: {np.mean(np.array(dit_fps_list)):.4f}")
        self.logger.info(f"Video shape: {video.shape}, Average FPS: {fps_avg:.4f}")
        
        output_path = os.path.join(output_folder, f"output_{0:03d}.mp4")
        export_to_video(video, output_path, fps=fps)
        self.logger.info(f"Video saved to: {output_path}")
        
        self.logger.info("Single GPU inference pipeline completed")


def main():
    """Main function for the single GPU inference pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Configuration file path")
    parser.add_argument("--checkpoint_folder", type=str, required=True, help="Checkpoint folder path")
    parser.add_argument("--output_folder", type=str, required=True, help="Output folder path")
    parser.add_argument("--prompt_file_path", type=str, required=True, help="Prompt file path")
    parser.add_argument("--video_path", type=str, required=False, default=None, help="Input video path")
    parser.add_argument("--noise_scale", type=float, default=0.700, help="Noise scale")
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--width", type=int, default=832, help="Video width")
    parser.add_argument("--fps", type=int, default=16, help="Output video fps")
    parser.add_argument("--step", type=int, default=2, help="Step")
    parser.add_argument("--model_type", type=str, default="T2V-1.3B", help="Model type (e.g., T2V-1.3B)")
    parser.add_argument("--num_frames", type=int, default=81, help="Video length (number of frames)")
    parser.add_argument("--fixed_noise_scale", action="store_true", default=False)
    parser.add_argument("--target_fps", type=int, required=False, default=None, help="Video length (number of frames)")
    args = parser.parse_args()
    
    torch.set_grad_enabled(False)
    
    # Auto-detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load configuration
    config = OmegaConf.load(args.config_path)
    config = OmegaConf.merge(config, OmegaConf.create(vars(args)))
    # Derive denoising_step_list from step if provided
    # Always base on the canonical full list to ensure --step overrides YAML
    full_denoising_list = [700, 600, 500, 400, 0]
    step_value = int(args.step)
    # Preserve historical mappings for 1..4
    if step_value <= 1:
        config.denoising_step_list = [700, 0]
    elif step_value == 2:
        config.denoising_step_list = [700, 500, 0]
    elif step_value == 3:
        config.denoising_step_list = [700, 600, 400, 0]
    else:
        config.denoising_step_list = full_denoising_list
    
    # Load input video
    if args.video_path is not None:
        input_video_original = load_mp4_as_tensor(args.video_path, resize_hw=(args.height, args.width)).unsqueeze(0)
        print(f"Input video tensor shape: {input_video_original.shape}")
        b, c, t, h, w = input_video_original.shape
        if input_video_original.dtype != torch.bfloat16:
            input_video_original = input_video_original.to(dtype=torch.bfloat16).to(device)
    else:
        input_video_original = None
        t = args.num_frames
    
    # Calculate number of chunks
    chunk_size = 4 * config.num_frame_per_block
    num_chunks = (t - 1) // chunk_size
    
    # Initialize pipeline manager
    pipeline_manager = SingleGPUInferencePipeline(config, device)
    pipeline_manager.load_model(args.checkpoint_folder)
    
    # Load prompts
    dataset = TextDataset(args.prompt_file_path)
    prompts = [dataset[0]]
    num_steps = len(pipeline_manager.pipeline.denoising_step_list)
    
    # Run inference
    try:
        pipeline_manager.run_inference(
            input_video_original, prompts, num_chunks, chunk_size, 
            args.noise_scale, args.output_folder, args.fps, args.target_fps, num_steps
        )
    except Exception as e:
        print(f"Error occurred during inference: {e}")
        raise


if __name__ == "__main__":
    main()
````

## File: streamv2v/inference_pipe.py

````python
"""
Refactored multi-rank inference pipeline with communication abstractions.

This is a refactored version of inference_pipe_multi.py that uses the new
communication abstraction layers for better code organization and maintainability.
"""

from causvid.models.wan.causal_stream_inference import CausalStreamInferencePipeline
from streamv2v.inference import compute_noise_scale_and_step
from diffusers.utils import export_to_video
from causvid.data import TextDataset
from omegaconf import OmegaConf
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import time
import numpy as np

import torchvision
import torchvision.transforms.functional as TF
from einops import rearrange

# Import our new communication abstractions
from streamv2v.communication import (
    DistributedCommunicator,
    ModelDataTransfer,
    BufferManager,
    KVCacheManager,
    CommunicationConfig,
    init_distributed,
    setup_logging,
    compute_balanced_split
)


def load_mp4_as_tensor(
    video_path: str,
    max_frames: int = None,
    resize_hw: tuple[int, int] = None,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Loads an .mp4 video and returns it as a PyTorch tensor with shape [C, T, H, W].

    Args:
        video_path (str): Path to the input .mp4 video file.
        max_frames (int, optional): Maximum number of frames to load. If None, loads all.
        resize_hw (tuple, optional): Target (height, width) to resize each frame. If None, no resizing.
        normalize (bool, optional): Whether to normalize pixel values to [-1, 1].

    Returns:
        torch.Tensor: Tensor of shape [C, T, H, W], dtype=torch.float32
    """
    assert os.path.exists(video_path), f"Video file not found: {video_path}"

    video, _, _ = torchvision.io.read_video(video_path, output_format="TCHW")
    if max_frames is not None:
        video = video[:max_frames]

    video = rearrange(video, "t c h w -> c t h w")
    if resize_hw is not None:
        c, t, h0, w0 = video.shape
        video = torch.stack([
            TF.resize(video[:, i], resize_hw, antialias=True)
            for i in range(t)
        ], dim=1)
    if video.dtype != torch.float32:
        video = video.float()
    if normalize:
        video = video / 127.5 - 1.0

    return video  # [C, T, H, W]



class InferencePipelineManager:
    """
    Manages the inference pipeline with communication abstractions.
    
    This class encapsulates the main inference logic and uses the communication
    abstractions for distributed operations.
    """
    
    def __init__(self, config, device: torch.device, rank: int, world_size: int):
        """
        Initialize the inference pipeline manager.
        
        Args:
            config: Configuration object
            device: GPU device
            rank: Current rank
            world_size: Total number of ranks
        """
        self.config = config
        self.device = device
        self.rank = rank
        self.world_size = world_size

        self.com_stream = torch.cuda.Stream()
        self.control_stream = torch.cuda.Stream()
        
        # Setup logging
        self.logger = setup_logging(rank)
        
        # Initialize communication components
        comm_config = CommunicationConfig(
            max_outstanding=config.get('max_outstanding', 1),
            buffer_pool_size=config.get('buffer_pool_size', 10),
            enable_buffer_reuse=config.get('enable_buffer_reuse', True)
        )
        
        self.communicator = DistributedCommunicator(rank, world_size, device, comm_config)
        self.buffer_manager = BufferManager(device, comm_config)
        
        # Initialize pipeline
        self.pipeline = CausalStreamInferencePipeline(config, device=str(device))
        self.pipeline.to(device=str(device), dtype=torch.bfloat16)
        
        # Initialize KV cache manager
        self.kv_cache_manager = KVCacheManager(self.pipeline, device)
        
        # Initialize model data transfer
        self.data_transfer = ModelDataTransfer(
            self.communicator, 
            self.buffer_manager, 
            self.kv_cache_manager, 
            comm_config
        )
        
        # Performance tracking
        self.t_dit = 100.0
        self.t_total = 100.0
        self.processed = 0
        self.schedule_step = (self.world_size + len(config.denoising_step_list)) * 2
        self.processed_offset = 3
        self.base_chunk_size = 4
        self.t_refresh = 50
        
        self.logger.info(f"Initialized InferencePipelineManager for rank {rank}")
    
    def load_model(self, checkpoint_folder: str):
        """Load the model from checkpoint."""
        state_dict = torch.load(os.path.join(checkpoint_folder, "model.pt"), map_location="cpu")["generator"]
        self.pipeline.generator.load_state_dict(state_dict, strict=True)
        self.logger.info("Model loaded successfully")
    
    def prepare_pipeline(self, text_prompts: list, noise: torch.Tensor, 
                        block_mode: str, current_start: int, current_end: int, block_num: torch.Tensor):
        """Prepare the pipeline for inference."""
        denoised_pred = self.pipeline.prepare(
            text_prompts=text_prompts, 
            device=self.device, 
            dtype=torch.bfloat16, 
            noise=noise, 
            block_mode=block_mode, 
            current_start=current_start, 
            current_end=current_end,
            block_num=block_num
        )
        
        # Broadcast the prepared result from rank 0
        self.data_transfer.broadcast_tensor(denoised_pred, src=0)
        return denoised_pred
    
    def run_rank_0_loop(self, input_video_original: torch.Tensor, prompts: list, 
                       num_chunks: int, num_steps: int, chunk_size: int,
                       block_num: torch.Tensor, noise_scale: float, 
                       schedule_block: bool, total_blocks: int):
        """
        Run the main loop for rank 0 (encoder + async send).
        
        This method encapsulates the rank 0 logic using the communication abstractions.
        """
        self.logger.info("Starting rank 0 inference loop")
        
        # Initialize variables
        start_idx = 0
        end_idx = 1 + chunk_size
        current_start = 0
        current_end = self.pipeline.frame_seq_length * (1+chunk_size//self.base_chunk_size)
        init_noise_scale = noise_scale
        
        outstanding = []
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        while True:
            # Process new chunk if available
            start_idx = end_idx
            end_idx = end_idx + chunk_size
            current_start = current_end
            current_end = current_end + (chunk_size // self.base_chunk_size) * self.pipeline.frame_seq_length

            if schedule_block:
                torch.cuda.synchronize()
                start_vae = time.time()
                
            if end_idx <= input_video_original.shape[2]:
                inp = input_video_original[:, :, start_idx:end_idx]
                
                noise_scale, current_step = compute_noise_scale_and_step(
                    input_video_original, end_idx, chunk_size, noise_scale, init_noise_scale
                )
                
                latents = self.pipeline.vae.stream_encode(inp)
                latents = latents.transpose(2, 1).contiguous().to(dtype=torch.bfloat16)
                
                noise = torch.randn_like(latents)
                noisy_latents = noise * noise_scale + latents * (1 - noise_scale)

            # if current_start//self.pipeline.frame_seq_length >= self.t_refresh:
            #     current_start = self.pipeline.kv_cache_length - self.pipeline.frame_seq_length
            #     current_end = current_start + (chunk_size // self.base_chunk_size) * self.pipeline.frame_seq_length
            
            # Measure DiT time if scheduling is enabled
            if schedule_block:
                torch.cuda.synchronize()
                start_dit = time.time()
                t_vae = start_dit - start_vae
            
            # Run inference
            denoised_pred, patched_x_shape = self.pipeline.inference(
                noise=noisy_latents,
                current_start=current_start,
                current_end=current_end,
                current_step=current_step,
                block_mode='input',
                block_num=block_num[self.rank],
            )
            
            # Update DiT timing
            if schedule_block:
                torch.cuda.synchronize()
                temp = time.time() - start_dit
                if temp < self.t_dit:
                    self.t_dit = temp
            
            self.processed += 1
            
            with torch.cuda.stream(self.com_stream):
                if self.processed >= self.world_size:
                    if 'latent_data' in locals():
                        self.buffer_manager.return_buffer(latent_data.latents, "latent")
                        self.buffer_manager.return_buffer(latent_data.original_latents, "origin")

                        if hasattr(latent_data, 'patched_x_shape') and latent_data.patched_x_shape is not None:
                            self.buffer_manager.return_buffer(latent_data.patched_x_shape, "misc")
                        if hasattr(latent_data, 'current_start') and latent_data.current_start is not None:
                            self.buffer_manager.return_buffer(latent_data.current_start, "misc")
                        if hasattr(latent_data, 'current_end') and latent_data.current_end is not None:
                            self.buffer_manager.return_buffer(latent_data.current_end, "misc")

                    # Receive data from previous rank
                    latent_data = self.data_transfer.receive_latent_data_async(num_steps)
            
            torch.cuda.current_stream().wait_stream(self.com_stream)
            
            # Wait for outstanding operations
            while len(outstanding) >= self.config.get('max_outstanding', 1):
                oldest = outstanding.pop(0)
                for work in oldest:
                    work.wait()
            
            # Send data to next rank
            with torch.cuda.stream(self.com_stream):
                work_objects = self.data_transfer.send_latent_data_async(
                    chunk_idx=start_idx,
                    latents=denoised_pred,
                    original_latents=self.pipeline.hidden_states,
                    patched_x_shape=patched_x_shape,
                    current_start=self.pipeline.kv_cache_starts,
                    current_end=self.pipeline.kv_cache_ends,
                    current_step=current_step
                )
                outstanding.append(work_objects)
                # Handle block scheduling
                if schedule_block and self.processed >= self.schedule_step:
                    self._handle_block_scheduling(block_num, total_blocks)
                    schedule_block = False

            # Update timing and check completion
            torch.cuda.synchronize()
            end_time = time.time()
            t = end_time - start_time
            self.logger.info(f"Encode {self.processed}, time: {t:.4f} s, fps: {inp.shape[2]/t:.4f}")
            
            if schedule_block:
                t_total = self.t_dit + t_vae
                if t_total < self.t_total:
                    self.t_total = t_total

            if self.processed >= self.world_size:
                self.pipeline.hidden_states.copy_(latent_data.original_latents)
                self.pipeline.kv_cache_starts.copy_(latent_data.current_start)
                self.pipeline.kv_cache_ends.copy_(latent_data.current_end)
            
            start_time = end_time

            if self.processed + self.processed_offset >= num_chunks + num_steps * self.world_size + self.world_size - self.rank - 1:
                break
        
        self.logger.info("Rank 0 inference loop completed")
    
    def run_final_rank_loop(self, num_chunks: int, num_steps: int, chunk_size: int,
                           block_num: torch.Tensor, output_folder: str, fps: int,
                           schedule_block: bool, total_blocks: int, results: dict):
        """
        Run the main loop for the final rank (async receiver + decode).
        
        This method encapsulates the final rank logic using the communication abstractions.
        """
        self.logger.info("Starting final rank inference loop")
        
        os.makedirs(output_folder, exist_ok=True)
        save_results = 1
        
        outstanding = []

        fps_list = []
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        while save_results < num_chunks:
            # Receive data from previous rank
            with torch.cuda.stream(self.com_stream):
                if 'latent_data' in locals():
                    self.buffer_manager.return_buffer(latent_data.latents, "latent")
                    self.buffer_manager.return_buffer(latent_data.original_latents, "origin")

                    if hasattr(latent_data, 'patched_x_shape') and latent_data.patched_x_shape is not None:
                        self.buffer_manager.return_buffer(latent_data.patched_x_shape, "misc")
                    if hasattr(latent_data, 'current_start') and latent_data.current_start is not None:
                        self.buffer_manager.return_buffer(latent_data.current_start, "misc")
                    if hasattr(latent_data, 'current_end') and latent_data.current_end is not None:
                        self.buffer_manager.return_buffer(latent_data.current_end, "misc")

                latent_data = self.data_transfer.receive_latent_data_async(num_steps)
                # Handle block scheduling
                if schedule_block and self.processed >= self.schedule_step - self.rank:
                    self._handle_block_scheduling(block_num, total_blocks)
                    schedule_block = False
            torch.cuda.current_stream().wait_stream(self.com_stream)
            
            # Measure DiT time if scheduling is enabled
            if schedule_block:
                torch.cuda.synchronize()
                start_dit = time.time()
            
            # Run inference
            denoised_pred, _ = self.pipeline.inference(
                noise=latent_data.original_latents,
                current_start=latent_data.current_start,
                current_end=latent_data.current_end,
                current_step=latent_data.current_step,
                block_mode='output',
                block_num=block_num[self.rank],
                patched_x_shape=latent_data.patched_x_shape,
                block_x=latent_data.latents,
            )
            
            # Update DiT timing
            if schedule_block:
                torch.cuda.synchronize()
                temp = time.time() - start_dit
                if temp < self.t_dit:
                    self.t_dit = temp
            
            self.processed += 1
            
            # Wait for outstanding operations
            while len(outstanding) >= self.config.get('max_outstanding', 1):
                oldest = outstanding.pop(0)
                for work in oldest:
                    work.wait()
            
            # Send data to next rank (if not the last rank)
            with torch.cuda.stream(self.com_stream):
                work_objects = self.data_transfer.send_latent_data_async(
                    chunk_idx=latent_data.chunk_idx,
                    latents=latent_data.latents,
                    original_latents=denoised_pred,
                    patched_x_shape=latent_data.patched_x_shape,
                    current_start=latent_data.current_start,
                    current_end=latent_data.current_end,
                    current_step=latent_data.current_step
                )
                outstanding.append(work_objects)

            # Decode and save video
            if self.processed >= num_steps * self.world_size - 1:
                if schedule_block:
                    torch.cuda.synchronize()
                    start_vae = time.time()

                video = self.pipeline.vae.stream_decode_to_pixel(denoised_pred[[-1]])
                video = (video * 0.5 + 0.5).clamp(0, 1)
                video = video[0].permute(0, 2, 3, 1).contiguous()
                
                results[save_results] = video.cpu().float().numpy()
                
                torch.cuda.synchronize()
                end_time = time.time()
                t = end_time - start_time
                fps_test = video.shape[0]/t
                if self.processed > self.schedule_step:
                    fps_list.append(fps_test)
                self.logger.info(f"Decode {self.processed}, time: {t:.4f} s, FPS: {fps_test:.4f}")
                
                if schedule_block:
                    t_vae = end_time - start_vae
                    t_total = t_vae + self.t_dit
                    if t_total < self.t_total:
                        self.t_total = t_total
                
                save_results += 1
                start_time = end_time
                
            if save_results >= num_chunks:
                break
        
        # Save final video
        video_list = [results[i] for i in range(num_chunks)]
        video = np.concatenate(video_list, axis=0)

        fps_list = np.array(fps_list)
        fps_avg = np.mean(fps_list)
        self.logger.info(f"Video shape: {video.shape}, Average FPS: {fps_avg:.4f}")
        
        output_path = os.path.join(output_folder, f"output_{0:03d}.mp4")
        export_to_video(video, output_path, fps=fps)
        self.logger.info(f"Video saved to: {output_path} (Press Ctrl+C to force exit)")
    
    def run_middle_rank_loop(self, num_chunks: int, num_steps: int, chunk_size: int,
                            block_num: torch.Tensor, schedule_block: bool, total_blocks: int):
        """
        Run the main loop for middle ranks (async receiver + dit blocks + sender).
        
        This method encapsulates the middle rank logic using the communication abstractions.
        """
        self.logger.info("Starting middle rank inference loop")
        
        outstanding = []
        
        torch.cuda.synchronize()
        start_time = time.time()

        fps_list = []
        
        while True:
            # Receive data from previous rank
            with torch.cuda.stream(self.com_stream):
                if 'latent_data' in locals():
                    self.buffer_manager.return_buffer(latent_data.latents, "latent")
                    self.buffer_manager.return_buffer(latent_data.original_latents, "origin")
                    if hasattr(latent_data, 'patched_x_shape') and latent_data.patched_x_shape is not None:
                        self.buffer_manager.return_buffer(latent_data.patched_x_shape, "misc")
                    if hasattr(latent_data, 'current_start') and latent_data.current_start is not None:
                        self.buffer_manager.return_buffer(latent_data.current_start, "misc")
                    if hasattr(latent_data, 'current_end') and latent_data.current_end is not None:
                        self.buffer_manager.return_buffer(latent_data.current_end, "misc")
                latent_data = self.data_transfer.receive_latent_data_async(num_steps)

                # Handle block scheduling
                if schedule_block and self.processed >= self.schedule_step - self.rank:
                    self._handle_block_scheduling(block_num, total_blocks)
                    schedule_block = False

            torch.cuda.current_stream().wait_stream(self.com_stream)

            if schedule_block:
                torch.cuda.synchronize()
                start_dit = time.time()
            
            # Run inference
            denoised_pred, _ = self.pipeline.inference(
                noise=latent_data.original_latents,
                current_start=latent_data.current_start,
                current_end=latent_data.current_end,
                current_step=latent_data.current_step,
                block_mode='middle',
                block_num=block_num[self.rank],
                patched_x_shape=latent_data.patched_x_shape,
                block_x=latent_data.latents,
            )
            
            if schedule_block:
                torch.cuda.synchronize()
                temp = time.time() - start_dit
                if temp < self.t_dit:
                    self.t_dit = temp
            
            self.processed += 1

            # Wait for outstanding operations
            while len(outstanding) >= self.config.get('max_outstanding', 1):
                oldest = outstanding.pop(0)
                for work in oldest:
                    work.wait()
            
            # Send data to next rank
            with torch.cuda.stream(self.com_stream):
                work_objects = self.data_transfer.send_latent_data_async(
                    chunk_idx=latent_data.chunk_idx,
                    latents=denoised_pred,
                    original_latents=latent_data.original_latents,
                    patched_x_shape=latent_data.patched_x_shape,
                    current_start=latent_data.current_start,
                    current_end=latent_data.current_end,
                    current_step=latent_data.current_step
                )
                outstanding.append(work_objects)
            
            # Update timing
            torch.cuda.synchronize()
            end_time = time.time()
            t = end_time - start_time

            if self.processed > self.schedule_step:
                fps_list.append(chunk_size/t)

            if schedule_block:
                t_total = self.t_dit
                if t_total < self.t_total:
                    self.t_total = t_total
            
            self.logger.info(f"Middle {self.processed}, time: {t:.4f} s, fps: {chunk_size/t:.4f}")

            start_time = end_time

            if self.processed + self.processed_offset >= num_chunks + num_steps * self.world_size + self.world_size - self.rank - 1:
                break
        
        self.logger.info(f"DiT Average FPS: {np.mean(fps_list):.4f}")
        self.logger.info(f"Rank {self.rank} inference loop completed")
    
    def _handle_block_scheduling(self, block_num: torch.Tensor, total_blocks: int):
        """Handle block scheduling and rebalancing."""
        self.logger.info(f"Scheduling block in {self.processed}")
        
        # Gather timing information from all ranks
        t_total_tensor = torch.tensor(self.t_total, dtype=torch.float32, device=self.device)
        t_dit_tensor = torch.tensor(self.t_dit, dtype=torch.float32, device=self.device)
        
        gather_blocks = [torch.zeros_like(t_dit_tensor, dtype=torch.float32, device=self.device) 
                        for _ in range(self.world_size)]
        
        dist.all_gather(gather_blocks, t_dit_tensor)
        t_dit_list = [t_dit_i.item() for t_dit_i in gather_blocks]
        
        dist.all_gather(gather_blocks, t_total_tensor)
        t_list = [t_i.item() for t_i in gather_blocks]
        
        # Compute new block distribution
        new_block_num = torch.tensor(
            compute_balanced_split(total_blocks, t_list, t_dit_list, block_num.tolist()),
            dtype=torch.int64, device=self.device
        )

        self.logger.info(f"New block distribution: {new_block_num[self.rank].tolist()}")
        
        # Broadcast new block distribution
        dist.broadcast(new_block_num, src=self.world_size - 1)
        
        # Rebalance KV cache
        self.data_transfer.rebalance_kv_cache(block_num, new_block_num, total_blocks)
        
        # Update block_num
        block_num.copy_(new_block_num)

        start_block, end_block = block_num[self.rank][0].item(), block_num[self.rank][1].item()
        blocks_to_keep = list(range(start_block, end_block))
        for i in range(self.pipeline.num_transformer_blocks):
            if i not in blocks_to_keep:
                self.pipeline.kv_cache1[i]['k'] = self.pipeline.kv_cache1[i]['k'].cpu()
                self.pipeline.kv_cache1[i]['v'] = self.pipeline.kv_cache1[i]['v'].cpu()

        self.logger.info("Block scheduling completed")
    
    def cleanup(self):
        """Clean up resources."""
        self.data_transfer.cleanup()
        self.logger.info("InferencePipelineManager cleanup completed")


def main():
    """Main function for the refactored inference pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--checkpoint_folder", type=str)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--prompt_file_path", type=str)
    parser.add_argument("--video_path", type=str)
    parser.add_argument("--noise_scale", type=float, default=0.8)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max_outstanding", type=int, default=1, help="max number of outstanding sends/recv to keep")
    parser.add_argument("--dit_fsdp", action="store_true", default=False)
    parser.add_argument("--t5_fsdp", action="store_true", default=False)
    parser.add_argument("--ulysses_size", type=int, default=1)
    parser.add_argument("--ring_size", type=int, default=1)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--schedule_block", action="store_true", default=False)
    parser.add_argument("--model_type", type=str, default="T2V-1.3B", help="Model type (e.g., T2V-1.3B)")
    
    args = parser.parse_args()
    
    torch.set_grad_enabled(False)
    init_distributed()
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    
    assert world_size >= 2, "world_size must be at least 2"
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # Load configuration
    config = OmegaConf.load(args.config_path)
    for k, v in vars(args).items():
        config[k] = v
    # Always base on the canonical full list to ensure --step overrides YAML
    full_denoising_list = [700, 600, 500, 400, 0]
    step_value = args.step
    if step_value <= 1:
        config.denoising_step_list = [700, 0]
    elif step_value == 2:
        config.denoising_step_list = [700, 500, 0]
    elif step_value == 3:
        config.denoising_step_list = [700, 600, 400, 0]
    else:
        config.denoising_step_list = full_denoising_list
    
    # Load input video
    input_video_original = load_mp4_as_tensor(args.video_path, resize_hw=(args.height, args.width)).unsqueeze(0)
    if input_video_original.dtype != torch.bfloat16:
        input_video_original = input_video_original.to(dtype=torch.bfloat16).to(device)
    
    print(f"Input video tensor shape: {input_video_original.shape}")
    b, c, t, h, w = input_video_original.shape
    
    # Calculate number of chunks
    chunk_size = 4 * config.num_frame_per_block
    if rank == 0:
        num_chunks = (t - 1) // chunk_size
    else:
        num_chunks = 0
    num_chunks_tensor = torch.tensor([num_chunks], dtype=torch.int64, device=device)
    dist.broadcast(num_chunks_tensor, src=0)
    num_chunks = int(num_chunks_tensor.item())
    
    # Initialize pipeline manager
    pipeline_manager = InferencePipelineManager(config, device, rank, world_size)
    pipeline_manager.load_model(args.checkpoint_folder)

    # Load prompts
    dataset = TextDataset(args.prompt_file_path)
    prompts = [dataset[0]]
    num_steps = len(pipeline_manager.pipeline.denoising_step_list)
    
    # Determine block mode and setup block distribution
    if rank == 0:
        block_mode = 'input'
    elif rank == world_size - 1:
        block_mode = 'output'
    else:
        block_mode = 'middle'
    
    # Setup block distribution
    total_blocks = pipeline_manager.pipeline.num_transformer_blocks
    if world_size == 2:
        total_block_num = [[0, 15], [15, total_blocks]]
    else:
        base = total_blocks // world_size
        rem = total_blocks % world_size
        start = 0
        total_block_num = []
        for r in range(world_size):
            size = base + (1 if r < rem else 0)
            end = start + size if r < world_size - 1 else total_blocks
            total_block_num.append([start, end])
            start = end
    
    block_num = torch.tensor(total_block_num, dtype=torch.int64, device=device)
    
    # Prepare pipeline
    start_idx = 0
    end_idx = 5
    current_start = 0
    current_end = pipeline_manager.pipeline.frame_seq_length * 2
    
    inp = input_video_original[:, :, start_idx:end_idx]
    
    # Only rank 0 performs VAE encoding operation
    if rank == 0:
        latents = pipeline_manager.pipeline.vae.stream_encode(inp)
        latents = latents.transpose(2, 1).contiguous().to(dtype=torch.bfloat16)
        noise = torch.randn_like(latents)
        noisy_latents = noise * args.noise_scale + latents * (1 - args.noise_scale)
        
        # First broadcast the shape information
        latents_shape = torch.tensor(latents.shape, dtype=torch.int64, device=device)
        pipeline_manager.communicator.broadcast_tensor(latents_shape, src=0)
        # Then broadcast noisy_latents
        pipeline_manager.communicator.broadcast_tensor(noisy_latents, src=0)
    else:
        # Other ranks receive shape info first
        latents_shape = torch.zeros(5, dtype=torch.int64, device=device)
        pipeline_manager.communicator.broadcast_tensor(latents_shape, src=0)
        # Create tensor with same shape for receiving broadcast data
        noisy_latents = torch.zeros(tuple(latents_shape.tolist()), dtype=torch.bfloat16, device=device)
        # Receive the broadcasted noisy_latents
        pipeline_manager.communicator.broadcast_tensor(noisy_latents, src=0)
    
    denoised_pred = pipeline_manager.prepare_pipeline(
        text_prompts=prompts,
        noise=noisy_latents,
        block_mode=block_mode,
        current_start=current_start,
        current_end=current_end,
        block_num=block_num[rank],
    )

    # Clear unused GPU memory
    torch.cuda.empty_cache()

    # Save initial result for final rank
    if rank == world_size - 1:
        results = {}
        video = pipeline_manager.pipeline.vae.stream_decode_to_pixel(denoised_pred)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        video = video[0].permute(0, 2, 3, 1).contiguous()
        results[0] = video.cpu().float().numpy()
    
    dist.barrier()
    pipeline_manager.logger.info(f"Prepared, Block num: {block_num[rank].tolist()}")

    used_mem = torch.cuda.memory_allocated(device) / 1024 / 1024 / 1024
    total_mem = torch.cuda.get_device_properties(device).total_memory / 1024 / 1024 / 1024
    pipeline_manager.logger.info(f"Current GPU memory usage: {used_mem:.2f} GB / {total_mem:.2f} GB")
    
    # Run appropriate loop based on rank
    try:
        if rank == 0:
            pipeline_manager.run_rank_0_loop(
                input_video_original, prompts, num_chunks, num_steps, chunk_size,
                block_num, args.noise_scale, args.schedule_block, total_blocks
            )
        elif rank == world_size - 1:
            pipeline_manager.run_final_rank_loop(
                num_chunks, num_steps, chunk_size, block_num, args.output_folder,
                args.fps, args.schedule_block, total_blocks, results
            )
        else:
            pipeline_manager.run_middle_rank_loop(
                num_chunks, num_steps, chunk_size, block_num, args.schedule_block, total_blocks
            )
    finally:
        # Cleanup
        pipeline_manager.cleanup()
    
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
````

## File: streamv2v/inference_wo_batch.py

````python
"""
Single GPU Inference Pipeline - Refactored from inference_pipe.py

This file extracts core logic from multi-GPU inference code to implement a complete 
inference pipeline on a single GPU:
1. VAE encode input video
2. DiT inference (using input mode, processing all 30 blocks)
3. VAE decode output video
"""

from causvid.models.wan.causal_stream_inference import CausalStreamInferencePipeline
from diffusers.utils import export_to_video
from causvid.data import TextDataset
from omegaconf import OmegaConf
import argparse
import torch
import os
import time
import numpy as np
import logging

import torchvision
import torchvision.transforms.functional as TF
from einops import rearrange
from streamv2v.inference import compute_noise_scale_and_step


def load_mp4_as_tensor(
    video_path: str,
    max_frames: int = None,
    resize_hw: tuple[int, int] = None,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Load an .mp4 video and return it as a PyTorch tensor with shape [C, T, H, W].

    Args:
        video_path (str): Path to the input .mp4 video file
        max_frames (int, optional): Maximum number of frames to load. If None, loads all frames
        resize_hw (tuple, optional): Target (height, width) to resize each frame. If None, no resizing
        normalize (bool, optional): Whether to normalize pixel values to [-1, 1]

    Returns:
        torch.Tensor: Tensor of shape [C, T, H, W], dtype=torch.float32
    """
    assert os.path.exists(video_path), f"Video file not found: {video_path}"

    video, _, _ = torchvision.io.read_video(video_path, output_format="TCHW")
    if max_frames is not None:
        video = video[:max_frames]

    video = rearrange(video, "t c h w -> c t h w")
    if resize_hw is not None:
        c, t, h0, w0 = video.shape
        video = torch.stack([
            TF.resize(video[:, i], resize_hw, antialias=True)
            for i in range(t)
        ], dim=1)
    if video.dtype != torch.float32:
        video = video.float()
    if normalize:
        video = video / 127.5 - 1.0

    return video  # [C, T, H, W]


class SingleGPUInferencePipeline:
    """
    Single GPU Inference Pipeline Manager
    
    This class encapsulates the complete inference logic on a single GPU, 
    including encoding, inference, and decoding.
    """
    
    def __init__(self, config, device: torch.device):
        """
        Initialize the single GPU inference pipeline manager.
        
        Args:
            config: Configuration object
            device: GPU device
        """
        self.config = config
        self.device = device
        
        # Setup logging
        self.logger = logging.getLogger("SingleGPUInference")
        self.logger.setLevel(logging.INFO)
        # Prevent messages from propagating to the root logger (avoid double prints)
        self.logger.propagate = False
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Initialize pipeline
        self.pipeline = CausalStreamInferencePipeline(config, device=str(device))
        self.pipeline.to(device=str(device), dtype=torch.bfloat16)
        
        # Performance tracking
        self.t_dit = 100.0
        self.t_total = 100.0
        self.processed = 0
        self.processed_offset = 3
        self.base_chunk_size = 4
        self.t_refresh = 50
        
        self.logger.info("Single GPU inference pipeline manager initialized")
    
    def load_model(self, checkpoint_folder: str):
        """Load the model from checkpoint."""
        state_dict = torch.load(os.path.join(checkpoint_folder, "model.pt"), map_location="cpu")["generator"]
        self.pipeline.generator.load_state_dict(state_dict, strict=True)
        self.logger.info("Model loaded successfully")
    
    def prepare_pipeline(self, text_prompts: list, noise: torch.Tensor, 
                        current_start: int, current_end: int, batch_denoise: bool=True):
        """Prepare the pipeline for inference."""
        # Use the original prepare method which now handles distributed environment gracefully
        denoised_pred = self.pipeline.prepare(
            text_prompts=text_prompts,
            device=self.device,
            dtype=torch.bfloat16,
            block_mode='input',
            noise=noise,
            current_start=current_start,
            current_end=current_end,
            batch_denoise=batch_denoise,
        )
        return denoised_pred
    
    def run_inference(self, input_video_original: torch.Tensor, prompts: list, 
                     num_chunks: int, chunk_size: int, noise_scale: float, 
                     output_folder: str, fps: int, num_steps: int):
        """
        Run the complete single GPU inference pipeline.
        
        This method integrates the complete encoding, inference, and decoding pipeline.
        """
        self.logger.info("Starting single GPU inference pipeline")
        
        os.makedirs(output_folder, exist_ok=True)
        results = {}
        save_results = 0

        fps_list = []
        dit_fps_list = []
        
        # Initialize variables
        start_idx = 0
        end_idx = 1 + chunk_size
        current_start = 0
        current_end = self.pipeline.frame_seq_length * (1+chunk_size//4)
        init_noise_scale = noise_scale
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        # Process first chunk (initialization)
        if end_idx <= input_video_original.shape[2]:
            inp = input_video_original[:, :, start_idx:end_idx]
            
            # VAE encoding
            latents = self.pipeline.vae.stream_encode(inp)
            latents = latents.transpose(2, 1).contiguous().to(dtype=torch.bfloat16)
            
            noise = torch.randn_like(latents)
            noisy_latents = noise * noise_scale + latents * (1 - noise_scale)
            
            # Prepare pipeline
            denoised_pred = self.prepare_pipeline(
                text_prompts=prompts,
                noise=noisy_latents,
                current_start=current_start,
                current_end=current_end,
                batch_denoise=False,
            )
            
            # Save first result - only start decoding after num_steps
            video = self.pipeline.vae.stream_decode_to_pixel(denoised_pred)
            video = (video * 0.5 + 0.5).clamp(0, 1)
            video = video[0].permute(0, 2, 3, 1).contiguous()
            results[save_results] = video.cpu().float().numpy()
            save_results += 1
        
        # Process remaining chunks
        while self.processed < num_chunks + num_steps - 1:
            # Update indices
            start_idx = end_idx
            end_idx = end_idx + chunk_size
            current_start = current_end
            current_end = current_end + (chunk_size // 4) * self.pipeline.frame_seq_length
            
            if end_idx <= input_video_original.shape[2]:
                inp = input_video_original[:, :, start_idx:end_idx]
                
                noise_scale, current_step = compute_noise_scale_and_step(
                    input_video_original, end_idx, chunk_size, noise_scale, init_noise_scale
                )
                
                # VAE encoding
                latents = self.pipeline.vae.stream_encode(inp)
                latents = latents.transpose(2, 1).contiguous().to(dtype=torch.bfloat16)
                
                noise = torch.randn_like(latents)
                noisy_latents = noise * noise_scale + latents * (1 - noise_scale)

            if current_start//self.pipeline.frame_seq_length >= 50:
                current_start = self.pipeline.kv_cache_length - self.pipeline.frame_seq_length
                current_end = current_start + (chunk_size // 4) * self.pipeline.frame_seq_length
                
            torch.cuda.synchronize()
            dit_start_time = time.time()
            # DiT inference - using input mode to process all 30 blocks
            denoised_pred = self.pipeline.inference_wo_batch(
                noise=noisy_latents,
                current_start=current_start,
                current_end=current_end,
                current_step=current_step,
            )
            
            if self.processed >self.processed_offset:
                torch.cuda.synchronize()
                dit_fps_list.append(chunk_size/(time.time()-dit_start_time))
            
            self.processed += 1
            
            # VAE decoding - only start decoding after num_steps
            video = self.pipeline.vae.stream_decode_to_pixel(denoised_pred[[-1]])
            video = (video * 0.5 + 0.5).clamp(0, 1)
            video = video[0].permute(0, 2, 3, 1).contiguous()
            
            results[save_results] = video.cpu().float().numpy()
            save_results += 1
        
            # Update timing
            torch.cuda.synchronize()
            end_time = time.time()
            t = end_time - start_time
            fps_test = inp.shape[2]/t
            fps_list.append(fps_test)
            self.logger.info(f"Processed {self.processed}, time: {t:.4f} s, FPS: {fps_test:.4f}")
            start_time = end_time
        
        # Save final video
        video_list = [results[i] for i in range(num_chunks)]
        video = np.concatenate(video_list, axis=0)
        fps_avg = np.mean(np.array(fps_list))
        self.logger.info(f"DiT Average FPS: {np.mean(np.array(dit_fps_list)):.4f}")
        self.logger.info(f"Video shape: {video.shape}, Average FPS: {fps_avg:.4f}")
        
        output_path = os.path.join(output_folder, f"output_{0:03d}.mp4")
        export_to_video(video, output_path, fps=fps)
        self.logger.info(f"Video saved to: {output_path}")
        
        self.logger.info("Single GPU inference pipeline completed")


def main():
    """Main function for the single GPU inference pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Configuration file path")
    parser.add_argument("--checkpoint_folder", type=str, required=True, help="Checkpoint folder path")
    parser.add_argument("--output_folder", type=str, required=True, help="Output folder path")
    parser.add_argument("--prompt_file_path", type=str, required=True, help="Prompt file path")
    parser.add_argument("--video_path", type=str, required=True, help="Input video path")
    parser.add_argument("--noise_scale", type=float, default=0.700, help="Noise scale")
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--width", type=int, default=832, help="Video width")
    parser.add_argument("--fps", type=int, default=16, help="Output video fps")
    parser.add_argument("--step", type=int, default=2, help="Step")
    parser.add_argument("--model_type", type=str, default="T2V-1.3B", help="Model type (e.g., T2V-1.3B)")
    args = parser.parse_args()
    
    torch.set_grad_enabled(False)
    
    # Auto-detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load configuration
    config = OmegaConf.load(args.config_path)
    for k, v in vars(args).items():
        config[k] = v
    # Derive denoising_step_list from step if provided
    # Always base on the canonical full list to ensure --step overrides YAML
    full_denoising_list = [700, 600, 500, 400, 0]
    step_value = args.step
    if step_value <= 1:
        config.denoising_step_list = [700, 0]
    elif step_value == 2:
        config.denoising_step_list = [700, 500, 0]
    elif step_value == 3:
        config.denoising_step_list = [700, 600, 400, 0]
    else:
        config.denoising_step_list = full_denoising_list
    
    # Load input video
    input_video_original = load_mp4_as_tensor(args.video_path, resize_hw=(args.height, args.width)).unsqueeze(0)
    if input_video_original.dtype != torch.bfloat16:
        input_video_original = input_video_original.to(dtype=torch.bfloat16).to(device)
    
    print(f"Input video tensor shape: {input_video_original.shape}")
    b, c, t, h, w = input_video_original.shape
    
    # Calculate number of chunks
    chunk_size = 4 * config.num_frame_per_block
    num_chunks = (t - 1) // chunk_size
    
    # Initialize pipeline manager
    pipeline_manager = SingleGPUInferencePipeline(config, device)
    pipeline_manager.load_model(args.checkpoint_folder)
    
    # Load prompts
    dataset = TextDataset(args.prompt_file_path)
    prompts = [dataset[0]]
    num_steps = len(pipeline_manager.pipeline.denoising_step_list)
    
    # Run inference
    try:
        pipeline_manager.run_inference(
            input_video_original, prompts, num_chunks, chunk_size, 
            args.noise_scale, args.output_folder, args.fps, num_steps
        )
    except Exception as e:
        print(f"Error occurred during inference: {e}")
        raise


if __name__ == "__main__":
    main()
````

## File: causvid/data.py

````python
from causvid.ode_data.create_lmdb_iterative import get_array_shape_from_lmdb, retrieve_row_from_lmdb
from torch.utils.data import Dataset
import numpy as np
import torch
import lmdb

import torchvision
import torchvision.transforms.functional as TF
from einops import rearrange
import os

def load_mp4_as_tensor(
    video_path: str,
    max_frames: int = None,
    resize_hw: tuple[int, int] = None,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Loads an .mp4 video and returns it as a PyTorch tensor with shape [C, T, H, W].

    Args:
        video_path (str): Path to the input .mp4 video file.
        max_frames (int, optional): Maximum number of frames to load. If None, loads all.
        resize_hw (tuple, optional): Target (height, width) to resize each frame. If None, no resizing.
        normalize (bool, optional): Whether to normalize pixel values to [-1, 1].

    Returns:
        torch.Tensor: Tensor of shape [C, T, H, W], dtype=torch.float32
    """
    assert os.path.exists(video_path), f"Video file not found: {video_path}"

    video, _, _ = torchvision.io.read_video(video_path, output_format="TCHW")
    if max_frames is not None:
        video = video[:max_frames]

    video = rearrange(video, "t c h w -> c t h w")
    h, w = video.shape[-2:]
    aspect_ratio = h / w
    # assert 8 / 16 <= aspect_ratio <= 17 / 16, (
    #     f"Unsupported aspect ratio: {aspect_ratio:.2f} for shape {video.shape}"
    # )
    if resize_hw is not None:
        c, t, h0, w0 = video.shape
        video = torch.stack([
            TF.resize(video[:, i], resize_hw, antialias=True)
            for i in range(t)
        ], dim=1)
    if video.dtype != torch.float32:
        video = video.float()
    if normalize:
        video = video / 127.5 - 1.0

    return video  # Final shape: [C, T, H, W]

class TextDataset(Dataset):
    def __init__(self, data_path):
        self.texts = []
        with open(data_path, "r") as f:
            for line in f:
                self.texts.append(line.strip())

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


class ODERegressionDataset(Dataset):
    def __init__(self, data_path, max_pair=int(1e8)):
        self.data_dict = torch.load(data_path, weights_only=False)
        self.max_pair = max_pair

    def __len__(self):
        return min(len(self.data_dict['prompts']), self.max_pair)

    def __getitem__(self, idx):
        """
        Outputs:
            - prompts: List of Strings
            - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        return {
            "prompts": self.data_dict['prompts'][idx],
            "ode_latent": self.data_dict['latents'][idx].squeeze(0),
        }


class ODERegressionLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8), load_video: bool = False):
        self.env = lmdb.open(data_path, readonly=True,
                             lock=False, readahead=False, meminit=False)

        self.latents_shape = get_array_shape_from_lmdb(self.env, 'latents')
        self.max_pair = max_pair
        self.load_video = load_video
        

    def __len__(self):
        return min(self.latents_shape[0], self.max_pair)

    def __getitem__(self, idx):
        """
        Outputs:
            - prompts: List of Strings
            - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        latents = retrieve_row_from_lmdb(
            self.env,
            "latents", np.float16, idx, shape=self.latents_shape[1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.env,
            "prompts", str, idx
        )
        if self.load_video:
            video_tensor = load_mp4_as_tensor(f"mixkit_videos/{prompts[:100]}.mp4")
            return {
                "prompts": prompts,
                "ode_latent": torch.tensor(latents, dtype=torch.float32),
                "video_tensor": video_tensor
            }
        else:
            return {
                "prompts": prompts,
                "ode_latent": torch.tensor(latents, dtype=torch.float32),
            }
````

## File: causvid/dmd.py

````python
from causvid.models.model_interface import InferencePipelineInterface
from causvid.models import (
    get_diffusion_wrapper,
    get_text_encoder_wrapper,
    get_vae_wrapper,
    get_inference_pipeline_wrapper
)
from causvid.loss import get_denoising_loss
import torch.nn.functional as F
from typing import Tuple, Optional
from torch import nn
import torch
from transformers import AutoModel

class DMD(nn.Module):
    def __init__(self, args, device):
        """
        Initialize the DMD (Distribution Matching Distillation) module.
        This class is self-contained and compute generator and fake score losses
        in the forward pass.
        """
        super().__init__()

        # Step 1: Initialize all models

        self.generator_model_name = getattr(
            args, "generator_name", args.model_name)
        self.real_model_name = getattr(args, "real_name", args.model_name)
        self.fake_model_name = getattr(args, "fake_name", args.model_name)

        self.generator_task_type = getattr(
            args, "generator_task_type", args.generator_task)
        self.real_task_type = getattr(
            args, "real_task_type", args.generator_task)
        self.fake_task_type = getattr(
            args, "fake_task_type", args.generator_task)

        self.generator = get_diffusion_wrapper(
            model_name=self.generator_model_name)()
        self.generator.set_module_grad(
            module_grad=args.generator_grad
        )

        if getattr(args, "generator_ckpt", False):
            print(f"Loading pretrained generator from {args.generator_ckpt}")
            state_dict = torch.load(args.generator_ckpt, map_location="cpu")[
                'generator']
            self.generator.load_state_dict(
                state_dict, strict=True
            )
        
        self.is_repa = args.is_repa
        if self.is_repa:
            self.repa_loss_weight = args.repa_loss_weight
            self.generator.model.repa_layer = args.repa_layer-1 

            model_name = "facebook/dinov2-large"
            self.encoder = AutoModel.from_pretrained(model_name).to(device)
            self.encoder.eval() 

            self.repa_mlp = nn.Sequential(
                nn.LayerNorm(1536),
                nn.Linear(1536, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
            ).requires_grad_(True).to(device=device, dtype=torch.bfloat16)

        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.real_score = get_diffusion_wrapper(
            model_name=self.real_model_name)()
        self.real_score.set_module_grad(
            module_grad=args.real_score_grad
        )

        self.fake_score = get_diffusion_wrapper(
            model_name=self.fake_model_name)()
        self.fake_score.set_module_grad(
            module_grad=args.fake_score_grad
        )

        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()
            self.fake_score.enable_gradient_checkpointing()

        self.text_encoder = get_text_encoder_wrapper(
            model_name=args.model_name)()
        self.text_encoder.requires_grad_(False)

        self.vae = get_vae_wrapper(model_name=args.model_name)()
        self.vae.requires_grad_(False)

        # this will be init later with fsdp-wrapped modules
        self.inference_pipeline: InferencePipelineInterface = None

        # Step 2: Initialize all dmd hyperparameters

        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long, device=device)
        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        self.real_guidance_scale = args.real_guidance_scale
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)

        self.args = args
        self.device = device
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32
        self.scheduler = self.generator.get_scheduler()
        self.denoising_loss_func = get_denoising_loss(
            args.denoising_loss_type)()

        if args.warp_denoising_step:  # Warp the denoising step according to the scheduler time
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))).cuda().cuda()
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        if getattr(self.scheduler, "alphas_cumprod", None) is not None:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(
                device)
        else:
            self.scheduler.alphas_cumprod = None

    def _process_timestep(self, timestep: torch.Tensor, type: str) -> torch.Tensor:
        """
        Pre-process the randomly generated timestep based on the generator's task type.
        Input:
            - timestep: [batch_size, num_frame] tensor containing the randomly generated timestep.
            - type: a string indicating the type of the current model (image, bidirectional_video, or causal_video).
        Output Behavior:
            - image: check that the second dimension (num_frame) is 1.
            - bidirectional_video: broadcast the timestep to be the same for all frames.
            - causal_video: broadcast the timestep to be the same for all frames **in a block**.
        """
        if type == "image":
            assert timestep.shape[1] == 1
            return timestep
        elif type == "bidirectional_video":
            for index in range(timestep.shape[0]):
                timestep[index] = timestep[index, 0]
            return timestep
        elif type == "causal_video":
            # make the noise level the same within every motion block
            timestep = timestep.reshape(
                timestep.shape[0], -1, self.num_frame_per_block)
            timestep[:, :, 1:] = timestep[:, :, 0:1]
            timestep = timestep.reshape(timestep.shape[0], -1)
            return timestep
        else:
            raise NotImplementedError("Unsupported model type {}".format(type))

    def _compute_kl_grad(
        self, noisy_image_or_video: torch.Tensor,
        estimated_clean_image_or_video: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict, unconditional_dict: dict,
        normalization: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the KL grad (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - noisy_image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - estimated_clean_image_or_video: a tensor with shape [B, F, C, H, W] representing the estimated clean image or video.
            - timestep: a tensor with shape [B, F] containing the randomly generated timestep.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - normalization: a boolean indicating whether to normalize the gradient.
        Output:
            - kl_grad: a tensor representing the KL grad.
            - kl_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        # Step 1: Compute the fake score
        pred_fake_image = self.fake_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep
        )

        # Step 2: Compute the real score
        # We compute the conditional and unconditional prediction
        # and add them together to achieve cfg (https://arxiv.org/abs/2207.12598)
        pred_real_image_cond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep
        )

        pred_real_image_uncond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=unconditional_dict,
            timestep=timestep
        )

        pred_real_image = pred_real_image_cond + (
            pred_real_image_cond - pred_real_image_uncond
        ) * self.real_guidance_scale

        # Step 3: Compute the DMD gradient (DMD paper eq. 7).
        grad = (pred_fake_image - pred_real_image)

        # TODO: Change the normalizer for causal teacher
        if normalization:
            # Step 4: Gradient normalization (DMD paper eq. 8).
            p_real = (estimated_clean_image_or_video - pred_real_image)
            normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
            grad = grad / normalizer
        grad = torch.nan_to_num(grad)

        return grad, {
            "dmdtrain_clean_latent": estimated_clean_image_or_video.detach(),
            "dmdtrain_noisy_latent": noisy_image_or_video.detach(),
            "dmdtrain_pred_real_image": pred_real_image.detach(),
            "dmdtrain_pred_fake_image": pred_fake_image.detach(),
            "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "timestep": timestep.detach()
        }

    def compute_distribution_matching_loss(
        self, image_or_video: torch.Tensor, conditional_dict: dict,
        unconditional_dict: dict, gradient_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the DMD loss (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - gradient_mask: a boolean tensor with the same shape as image_or_video indicating which pixels to compute loss .
        Output:
            - dmd_loss: a scalar tensor representing the DMD loss.
            - dmd_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        original_latent = image_or_video

        batch_size, num_frame = image_or_video.shape[:2]

        with torch.no_grad():
            # Step 1: Randomly sample timestep based on the given schedule and corresponding noise
            timestep = torch.randint(
                0,
                self.num_train_timestep,
                [batch_size, num_frame],
                device=self.device,
                dtype=torch.long
            )

            timestep = self._process_timestep(
                timestep, type=self.real_task_type)

            # TODO: Add timestep warping
            if self.timestep_shift > 1:
                timestep = self.timestep_shift * \
                    (timestep / 1000) / \
                    (1 + (self.timestep_shift - 1) * (timestep / 1000)) * 1000
            timestep = timestep.clamp(self.min_step, self.max_step)

            noise = torch.randn_like(image_or_video)
            noisy_latent = self.scheduler.add_noise(
                image_or_video.flatten(0, 1),
                noise.flatten(0, 1),
                timestep.flatten(0, 1)
            ).detach().unflatten(0, (batch_size, num_frame))

            # Step 2: Compute the KL grad
            grad, dmd_log_dict = self._compute_kl_grad(
                noisy_image_or_video=noisy_latent,
                estimated_clean_image_or_video=original_latent,
                timestep=timestep,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict
            )

        if gradient_mask is not None:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            )[gradient_mask], (original_latent.double() - grad.double()).detach()[gradient_mask], reduction="mean")
        else:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            ), (original_latent.double() - grad.double()).detach(), reduction="mean")
        return dmd_loss, dmd_log_dict

    def _initialize_inference_pipeline(self):
        """
        Lazy initialize the inference pipeline during the first backward simulation run.
        Here we encapsulate the inference code with a model-dependent outside function.
        We pass our FSDP-wrapped modules into the pipeline to save memory.
        """
        self.inference_pipeline = get_inference_pipeline_wrapper(
            self.generator_model_name,
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block
        )

    @torch.no_grad()
    def _consistency_backward_simulation(self, noise: torch.Tensor, conditional_dict: dict) -> torch.Tensor:
        """
        Simulate the generator's input from noise to avoid training/inference mismatch.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Here we use the consistency sampler (https://arxiv.org/abs/2303.01469)
        Input:
            - noise: a tensor sampled from N(0, 1) with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - output: a tensor with shape [B, T, F, C, H, W].
            T is the total number of timesteps. output[0] is a pure noise and output[i] and i>0
            represents the x0 prediction at each timestep.
        """
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        return self.inference_pipeline.inference_with_trajectory(noise=noise, conditional_dict=conditional_dict)

    def _run_generator(self, image_or_video_shape, conditional_dict: dict, unconditional_dict: dict, clean_latent: torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optionally simulate the generator's input from noise using backward simulation
        and then run the generator for one-step.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - pred_image: a tensor with shape [B, F, C, H, W].
        """
        # Step 1: Sample noise and backward simulate the generator's input
        if getattr(self.args, "backward_simulation", True):
            simulated_noisy_input = self._consistency_backward_simulation(
                noise=torch.randn(image_or_video_shape,
                                  device=self.device, dtype=self.dtype),
                conditional_dict=conditional_dict
            )
        else:
            simulated_noisy_input = []
            for timestep in self.denoising_step_list:
                noise = torch.randn(
                    image_or_video_shape, device=self.device, dtype=self.dtype)

                noisy_timestep = timestep * torch.ones(
                    image_or_video_shape[:2], device=self.device, dtype=torch.long)

                if timestep != 0:
                    noisy_image = self.scheduler.add_noise(
                        clean_latent.flatten(0, 1),
                        noise.flatten(0, 1),
                        noisy_timestep.flatten(0, 1)
                    ).unflatten(0, image_or_video_shape[:2])
                else:
                    noisy_image = clean_latent

                simulated_noisy_input.append(noisy_image)

            simulated_noisy_input = torch.stack(simulated_noisy_input, dim=1)

        # Step 2: Randomly sample a timestep and pick the corresponding input
        index = torch.randint(0, len(self.denoising_step_list), [
                              image_or_video_shape[0], image_or_video_shape[1]], device=self.device, dtype=torch.long)
        index = self._process_timestep(index, type=self.generator_task_type)

        # select the corresponding timestep's noisy input from the stacked tensor [B, T, F, C, H, W]
        noisy_input = torch.gather(
            simulated_noisy_input, dim=1,
            index=index.reshape(index.shape[0], 1, index.shape[1], 1, 1, 1).expand(
                -1, -1, -1, *image_or_video_shape[2:])
        ).squeeze(1)

        timestep = self.denoising_step_list[index]

        pred_image_or_video = self.generator(
            noisy_image_or_video=noisy_input,
            conditional_dict=conditional_dict,
            timestep=timestep
        )

        gradient_mask = None  # timestep != 0

        # pred_image_or_video = noisy_input * \
        #     (1-gradient_mask.float()).reshape(*gradient_mask.shape, 1, 1, 1) + \
        #     pred_image_or_video * gradient_mask.float().reshape(*gradient_mask.shape, 1, 1, 1)

        pred_image_or_video = pred_image_or_video.type_as(noisy_input)

        return pred_image_or_video, gradient_mask
    
    def _compute_repa_loss(self, pred_image_or_video: torch.Tensor, video_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute the REPA loss using DINOv2 features.
        """
        assert self.is_repa, "REPA is not enabled"
        assert self.generator.model.repa_layer != -1, "REPA layer is not set"
        assert self.generator.model.repa_hidden_states is not None, "REPA hidden states are not available"
        
        # Get video parameters
        assert video_tensor.shape[0] == 1, "Video tensor must have batch size 1"
        video_tensor = video_tensor.squeeze(0)
        C, T, H, W = video_tensor.shape
        
        # Calculate patch grids
        original_patches_h, original_patches_w = H // 14, W // 14
        target_patches_h, target_patches_w = H // 16, W // 16
        
        # Extract features using DINOv2
        features = []
        batch_size = 32
        
        for i in range(0, T, batch_size):
            batch_frames = video_tensor[:, i:i + batch_size].permute(1, 0, 2, 3)  # [B, C, H, W]
            
            with torch.no_grad():
                outputs = self.encoder(batch_frames)
                batch_features = outputs.last_hidden_state[:, 1:, :]  # All patch tokens
                features.append(batch_features)
        
        # Concatenate all features
        features = torch.cat(features, dim=0)  # [T, num_patches, feature_dim]
        
        # Process features: first frame + average pooling every 4 frames
        processed_features = [features[0:1]]  # Keep first frame
        
        remaining_frames = features[1:]
        num_groups = (remaining_frames.shape[0] + 3) // 4
        
        for i in range(num_groups):
            start_idx = i * 4
            end_idx = min(start_idx + 4, remaining_frames.shape[0])
            group_frames = remaining_frames[start_idx:end_idx]
            avg_frame = group_frames.mean(dim=0, keepdim=True)
            processed_features.append(avg_frame)
        
        processed_features = torch.cat(processed_features, dim=0)  # [T_processed, num_patches, feature_dim]
        
        # Reshape and downsample to target patch grid
        T_processed, num_patches_raw, feature_dim = processed_features.shape
        
        # Reshape to spatial grid and downsample
        features_spatial = processed_features.view(T_processed, original_patches_h, original_patches_w, feature_dim)
        features_spatial = features_spatial.permute(0, 3, 1, 2)  # [T_processed, feature_dim, H//14, W//14]
        
        # Downsample to 16x16 patch grid
        features_downsampled = torch.nn.functional.interpolate(
            features_spatial, 
            size=(target_patches_h, target_patches_w), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Final reshape
        feature_tensor = features_downsampled.permute(0, 2, 3, 1).reshape(1, -1, feature_dim)
        
        # Process pred_image_or_video through repa_mlp
        pred_features = self.repa_mlp(pred_image_or_video)  # [B, T_processed, num_patches, 1024]
        
        # Assert shapes match
        assert pred_features.shape == feature_tensor.shape, f"Shape mismatch: pred_features {pred_features.shape} vs feature_tensor {feature_tensor.shape}"
        
        # Compute cosine similarity between corresponding tokens
        cosine_sim = torch.nn.functional.cosine_similarity(pred_features, feature_tensor, dim=-1)  # [B, T_processed, num_patches]
        
        # Take average across all tokens
        repa_loss = -cosine_sim.mean()  # Negative because we want to maximize similarity
        
        return repa_loss
        

    def generator_loss(self, image_or_video_shape, conditional_dict: dict, unconditional_dict: dict, clean_latent: torch.Tensor, video_tensor: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and compute the DMD loss.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:  
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - generator_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        # Step 1: Run generator on backward simulated noisy input
        pred_image, gradient_mask = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent
        )

        # Step 2: Compute the DMD loss
        dmd_loss, dmd_log_dict = self.compute_distribution_matching_loss(
            image_or_video=pred_image,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask
        )
        if self.is_repa:
            repa_loss = self._compute_repa_loss(self.generator.model.repa_hidden_states, video_tensor)
            dmd_loss += repa_loss * self.repa_loss_weight

        # Step 3: TODO: Implement the GAN loss

        return dmd_loss, dmd_log_dict

    def critic_loss(self, image_or_video_shape, conditional_dict: dict, unconditional_dict: dict, clean_latent: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and train the critic with generated samples.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - critic_log_dict: a dictionary containing the intermediate tensors for logging.
        """

        # Step 1: Run generator on backward simulated noisy input
        with torch.no_grad():
            generated_image, _ = self._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent
            )

        # Step 2: Compute the fake prediction
        critic_timestep = torch.randint(
            0,
            self.num_train_timestep,
            image_or_video_shape[:2],
            device=self.device,
            dtype=torch.long
        )
        critic_timestep = self._process_timestep(
            critic_timestep, type=self.fake_task_type)

        # TODO: Add timestep warping
        if self.timestep_shift > 1:
            critic_timestep = self.timestep_shift * \
                (critic_timestep / 1000) / (1 + (self.timestep_shift - 1) * (critic_timestep / 1000)) * 1000

        critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)

        critic_noise = torch.randn_like(generated_image)
        noisy_generated_image = self.scheduler.add_noise(
            generated_image.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1)
        ).unflatten(0, image_or_video_shape[:2])

        pred_fake_image = self.fake_score(
            noisy_image_or_video=noisy_generated_image,
            conditional_dict=conditional_dict,
            timestep=critic_timestep
        )

        # Step 3: Compute the denoising loss for the fake critic
        if self.args.denoising_loss_type == "flow":
            assert "wan" in self.args.model_name
            from causvid.models.wan.wan_wrapper import WanDiffusionWrapper
            flow_pred = WanDiffusionWrapper._convert_x0_to_flow_pred(
                scheduler=self.scheduler,
                x0_pred=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            )
            pred_fake_noise = None
        else:
            flow_pred = None
            pred_fake_noise = self.scheduler.convert_x0_to_noise(
                x0=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            ).unflatten(0, image_or_video_shape[:2])

        denoising_loss = self.denoising_loss_func(
            x=generated_image.flatten(0, 1),
            x_pred=pred_fake_image.flatten(0, 1),
            noise=critic_noise.flatten(0, 1),
            noise_pred=pred_fake_noise,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=critic_timestep.flatten(0, 1),
            flow_pred=flow_pred
        )

        # Step 4: TODO: Compute the GAN loss

        # Step 5: Debugging Log
        critic_log_dict = {
            "critictrain_latent": generated_image.detach(),
            "critictrain_noisy_latent": noisy_generated_image.detach(),
            "critictrain_pred_image": pred_fake_image.detach(),
            "critic_timestep": critic_timestep.detach()
        }

        return denoising_loss, critic_log_dict
````

## File: causvid/loss.py

````python
from abc import ABC, abstractmethod
import torch


class DenoisingLoss(ABC):
    @abstractmethod
    def __call__(
        self, x: torch.Tensor, x_pred: torch.Tensor,
        noise: torch.Tensor, noise_pred: torch.Tensor,
        alphas_cumprod: torch.Tensor,
        timestep: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Base class for denoising loss.
        Input:
            - x: the clean data with shape [B, F, C, H, W]
            - x_pred: the predicted clean data with shape [B, F, C, H, W]
            - noise: the noise with shape [B, F, C, H, W]
            - noise_pred: the predicted noise with shape [B, F, C, H, W]
            - alphas_cumprod: the cumulative product of alphas (defining the noise schedule) with shape [T]
            - timestep: the current timestep with shape [B, F]
        """
        pass


class X0PredLoss(DenoisingLoss):
    def __call__(
        self, x: torch.Tensor, x_pred: torch.Tensor,
        noise: torch.Tensor, noise_pred: torch.Tensor,
        alphas_cumprod: torch.Tensor,
        timestep: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        return torch.mean((x - x_pred) ** 2)


class VPredLoss(DenoisingLoss):
    def __call__(
        self, x: torch.Tensor, x_pred: torch.Tensor,
        noise: torch.Tensor, noise_pred: torch.Tensor,
        alphas_cumprod: torch.Tensor,
        timestep: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        weights = 1 / \
            (1 - alphas_cumprod[timestep].reshape(*timestep.shape, 1, 1, 1))
        return torch.mean(weights * (x - x_pred) ** 2)


class NoisePredLoss(DenoisingLoss):
    def __call__(
        self, x: torch.Tensor, x_pred: torch.Tensor,
        noise: torch.Tensor, noise_pred: torch.Tensor,
        alphas_cumprod: torch.Tensor,
        timestep: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        return torch.mean((noise - noise_pred) ** 2)


class FlowPredLoss(DenoisingLoss):
    def __call__(
        self, x: torch.Tensor, x_pred: torch.Tensor,
        noise: torch.Tensor, noise_pred: torch.Tensor,
        alphas_cumprod: torch.Tensor,
        timestep: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        return torch.mean((kwargs["flow_pred"] - (noise - x)) ** 2)


NAME_TO_CLASS = {
    "x0": X0PredLoss,
    "v": VPredLoss,
    "noise": NoisePredLoss,
    "flow": FlowPredLoss
}


def get_denoising_loss(loss_type: str) -> DenoisingLoss:
    return NAME_TO_CLASS[loss_type]
````

## File: causvid/models/__init__.py

````python
from .wan.wan_wrapper import WanTextEncoder, WanVAEWrapper, WanDiffusionWrapper, CausalWanDiffusionWrapper
from causvid.bidirectional_trajectory_pipeline import BidirectionalInferenceWrapper
from .sdxl.sdxl_wrapper import SDXLWrapper, SDXLTextEncoder, SDXLVAE
from transformers.models.t5.modeling_t5 import T5Block


DIFFUSION_NAME_TO_CLASS = {
    "sdxl": SDXLWrapper,
    "wan": WanDiffusionWrapper,
    "causal_wan": CausalWanDiffusionWrapper
}


def get_diffusion_wrapper(model_name):
    return DIFFUSION_NAME_TO_CLASS[model_name]


TEXTENCODER_NAME_TO_CLASS = {
    "sdxl": SDXLTextEncoder,
    "wan": WanTextEncoder,
    "causal_wan": WanTextEncoder
}


def get_text_encoder_wrapper(model_name):
    return TEXTENCODER_NAME_TO_CLASS[model_name]


VAE_NAME_TO_CLASS = {
    "sdxl": SDXLVAE,
    "wan": WanVAEWrapper,
    "causal_wan": WanVAEWrapper   # TODO: Change the VAE to the causal version
}


def get_vae_wrapper(model_name):
    return VAE_NAME_TO_CLASS[model_name]


PIPELINE_NAME_TO_CLASS = {
    "sdxl": BidirectionalInferenceWrapper,
    "wan": BidirectionalInferenceWrapper
}


def get_inference_pipeline_wrapper(model_name, **kwargs):
    return PIPELINE_NAME_TO_CLASS[model_name](**kwargs)


BLOCK_NAME_TO_BLOCK_CLASS = {
    "T5Block": T5Block
}


def get_block_class(model_name):
    return BLOCK_NAME_TO_BLOCK_CLASS[model_name]
````

## File: causvid/models/model_interface.py

````python
from causvid.scheduler import SchedulerInterface
from abc import abstractmethod, ABC
from typing import List, Optional
import torch
import types


class DiffusionModelInterface(ABC, torch.nn.Module):
    scheduler: SchedulerInterface

    @abstractmethod
    def forward(
        self, noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor, kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        current_end: Optional[int] = None
    ) -> torch.Tensor:
        """
        A method to run diffusion model.
        Input:
            - noisy_image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - timestep: a tensor with shape [B, F]  where the number of frame is 1 for images.
            all data should be on the same device as the model.
            - kv_cache: a list of dictionaries containing the key and value tensors for each attention layer.
            - current_start: the start index of the current frame in the sequence.
            - current_end: the end index of the current frame in the sequence.
        Output: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
        We always expect a X0 prediction form for the output.
        """
        pass

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()

    def set_module_grad(self, module_grad: dict) -> None:
        """
        Adjusts the state of each module in the object.

        Parameters:
        - module_grad (dict): A dictionary where each key is the name of a module (as an attribute of the object),
          and each value is a bool indicating whether the module's parameters require gradients.

        Functionality:
        For each module name in the dictionary:
        - Updates whether its parameters require gradients based on 'is_trainable'.
        """
        for k, is_trainable in module_grad.items():
            getattr(self, k).requires_grad_(is_trainable)

    @abstractmethod
    def enable_gradient_checkpointing(self) -> None:
        """
        Activates gradient checkpointing for the current model (may be referred to as *activation checkpointing* or
        *checkpoint activations* in other frameworks).
        """
        pass


class VAEInterface(ABC, torch.nn.Module):
    @abstractmethod
    def decode_to_pixel(self, latent: torch.Tensor) -> torch.Tensor:
        """
        A method to decode a latent representation to an image or video.
        Input: a tensor with shape [B, F // T, C, H // S, W // S] where T and S are temporal and spatial compression factors.
        Output: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
        """
        pass


class TextEncoderInterface(ABC, torch.nn.Module):
    @abstractmethod
    def forward(self, text_prompts: List[str]) -> dict:
        """
        A method to tokenize text prompts with a tokenizer and encode them into a latent representation.
        Input: a list of strings.
        Output: a dictionary containing the encoded representation of the text prompts.
        """
        pass


class InferencePipelineInterface(ABC):
    @abstractmethod
    def inference_with_trajectory(self, noise: torch.Tensor, conditional_dict: dict) -> torch.Tensor:
        """
        Run inference with the given diffusion / distilled generators.
        Input:
            - noise: a tensor sampled from N(0, 1) with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - output: a tensor with shape [B, T, F, C, H, W].
            T is the total number of timesteps. output[0] is a pure noise and output[i] and i>0
            represents the x0 prediction at each timestep.
        """
````

## File: causvid/models/wan/bidirectional_inference.py

````python
from causvid.models import (
    get_diffusion_wrapper,
    get_text_encoder_wrapper,
    get_vae_wrapper
)
from typing import List
import torch


class BidirectionalInferencePipeline(torch.nn.Module):
    def __init__(self, args, device):
        super().__init__()
        # Step 1: Initialize all models
        self.generator_model_name = getattr(
            args, "generator_name", args.model_name)
        self.generator = get_diffusion_wrapper(
            model_name=self.generator_model_name)()
        self.text_encoder = get_text_encoder_wrapper(
            model_name=args.model_name)()
        self.vae = get_vae_wrapper(model_name=args.model_name)()

        # Step 2: Initialize all bidirectional wan hyperparmeters
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long, device=device)

        self.scheduler = self.generator.get_scheduler()
        if args.warp_denoising_step:  # Warp the denoising step according to the scheduler time shift
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))).cuda()
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

    def inference(self, noise: torch.Tensor, text_prompts: List[str]) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_frames, num_channels, height, width). It is normalized to be in the range [0, 1].
        """
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        # initial point
        noisy_image_or_video = noise

        for index, current_timestep in enumerate(self.denoising_step_list):
            pred_image_or_video = self.generator(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=conditional_dict,
                timestep=torch.ones(
                    noise.shape[:2], dtype=torch.long, device=noise.device) * current_timestep
            )  # [B, F, C, H, W]

            if index < len(self.denoising_step_list) - 1:
                next_timestep = self.denoising_step_list[index + 1] * torch.ones(
                    noise.shape[:2], dtype=torch.long, device=noise.device)

                noisy_image_or_video = self.scheduler.add_noise(
                    pred_image_or_video.flatten(0, 1),
                    torch.randn_like(pred_image_or_video.flatten(0, 1)),
                    next_timestep.flatten(0, 1)
                ).unflatten(0, noise.shape[:2])

        video = self.vae.decode_to_pixel(pred_image_or_video)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        return video
````

## File: causvid/models/wan/causal_inference.py

````python
from causvid.models import (
    get_diffusion_wrapper,
    get_text_encoder_wrapper,
    get_vae_wrapper
)
from typing import List, Optional
import torch


class InferencePipeline(torch.nn.Module):
    def __init__(self, args, device):
        super().__init__()
        # Step 1: Initialize all models
        self.generator_model_name = getattr(
            args, "generator_name", args.model_name)
        self.generator = get_diffusion_wrapper(
            model_name=self.generator_model_name)()
        self.text_encoder = get_text_encoder_wrapper(
            model_name=args.model_name)()
        self.vae = get_vae_wrapper(model_name=args.model_name)()

        # Step 2: Initialize all causal hyperparmeters
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long, device=device)
        assert self.denoising_step_list[-1] == 0
        # remove the last timestep (which equals zero)
        self.denoising_step_list = self.denoising_step_list[:-1]

        self.scheduler = self.generator.get_scheduler()
        if args.warp_denoising_step:  # Warp the denoising step according to the scheduler time shift
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))).cuda()
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560

        self.kv_cache1 = None
        self.kv_cache2 = None
        self.args = args
        self.num_frame_per_block = getattr(
            args, "num_frame_per_block", 1)

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, 32760, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 32760, 12, 128], dtype=dtype, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })

        self.crossattn_cache = crossattn_cache  # always store the clean cache

    def inference(self, noise: torch.Tensor, text_prompts: List[str], start_latents: Optional[torch.Tensor] = None, return_latents: bool = False) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_frames, num_channels, height, width). It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        output = torch.zeros(
            [batch_size, num_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Step 1: Initialize KV cache
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )

            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False

        num_input_blocks = start_latents.shape[1] // self.num_frame_per_block if start_latents is not None else 0

        # Step 2: Temporal denoising loop
        num_blocks = num_frames // self.num_frame_per_block
        for block_index in range(num_blocks):
            noisy_input = noise[:, block_index *
                                self.num_frame_per_block:(block_index + 1) * self.num_frame_per_block]

            if start_latents is not None and block_index < num_input_blocks:
                timestep = torch.ones(
                    [batch_size, self.num_frame_per_block], device=noise.device, dtype=torch.int64) * 0

                current_ref_latents = start_latents[:, block_index * self.num_frame_per_block:(
                    block_index + 1) * self.num_frame_per_block]
                output[:, block_index * self.num_frame_per_block:(
                    block_index + 1) * self.num_frame_per_block] = current_ref_latents

                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=block_index * self.num_frame_per_block * self.frame_seq_length,
                    current_end=(block_index + 1) *
                    self.num_frame_per_block * self.frame_seq_length
                )
                continue

            # Step 2.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                # set current timestep
                timestep = torch.ones(
                    [batch_size, self.num_frame_per_block], device=noise.device, dtype=torch.int64) * current_timestep

                if index < len(self.denoising_step_list) - 1:
                    denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=block_index * self.num_frame_per_block * self.frame_seq_length,
                        current_end=(
                            block_index + 1) * self.num_frame_per_block * self.frame_seq_length
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep *
                        torch.ones([batch_size], device="cuda",
                                   dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output
                    denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=block_index * self.num_frame_per_block * self.frame_seq_length,
                        current_end=(
                            block_index + 1) * self.num_frame_per_block * self.frame_seq_length
                    )

            # Step 2.2: rerun with timestep zero to update the cache
            output[:, block_index * self.num_frame_per_block:(
                block_index + 1) * self.num_frame_per_block] = denoised_pred

            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=timestep * 0,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=block_index * self.num_frame_per_block * self.frame_seq_length,
                current_end=(block_index + 1) *
                self.num_frame_per_block * self.frame_seq_length
            )

        # Step 3: Decode the output
        video = self.vae.decode_to_pixel(output)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if return_latents:
            return video, output
        else:
            return video
````

## File: causvid/models/wan/causal_model.py

````python
from causvid.models.wan.wan_base.modules.attention import attention
from causvid.models.wan.wan_base.modules.model import (
    WanRMSNorm,
    rope_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    Head,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch
import math
from flash_attn import flash_attn_interface
import torch.distributed as dist

# wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
# see https://github.com/pytorch/pytorch/issues/133254
# change to default for other models
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune")


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        sf = start_frame[i].item() if isinstance(start_frame, torch.Tensor) else start_frame

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][sf:sf + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        
        self.sink_size = 3
        self.adapt_sink_thr = -1

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs, block_mask, kv_cache=None, current_start=0, current_end=0):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if kv_cache is None:
            roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
            roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

            padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
            padded_roped_query = torch.cat(
                [roped_query,
                 torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                             device=q.device, dtype=v.dtype)],
                dim=1
            )

            padded_roped_key = torch.cat(
                [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                        device=k.device, dtype=v.dtype)],
                dim=1
            )

            padded_v = torch.cat(
                [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                device=v.device, dtype=v.dtype)],
                dim=1
            )

            x = flex_attention(
                query=padded_roped_query.transpose(2, 1),
                key=padded_roped_key.transpose(2, 1),
                value=padded_v.transpose(2, 1),
                block_mask=block_mask
            )[:, :, :-padded_length].transpose(2, 1)
        else:
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            current_start_frame = current_start // frame_seqlen
            roped_query = causal_rope_apply(
                q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
            roped_key = causal_rope_apply(
                k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)

            seq_lens = []
            for i, c_start in enumerate(current_start):
                current_end = c_start + roped_query.shape[1]
                sink_tokens = self.sink_size * frame_seqlen
                
                if sink_tokens > 0 and self.adapt_sink_thr > -1 and v.shape[1] <= frame_seqlen:
                    # Caculate similarity between new keys/values and the oldest ones in the cache
                    k_sink_mean = kv_cache["k"][i:i+1, :sink_tokens].reshape(self.sink_size, frame_seqlen, -1).mean(1)
                    k_new_mean = roped_key[i:i+1].reshape(1, frame_seqlen, -1).mean(1)
                    k_cos_sim = torch.cosine_similarity(k_sink_mean, k_new_mean, dim=-1)

                    v_sink_mean = kv_cache["v"][i:i+1, :sink_tokens].reshape(self.sink_size, frame_seqlen, -1).mean(1)
                    v_new_mean = v[i:i+1].reshape(1, frame_seqlen, -1).mean(1)
                    v_cos_sim = torch.cosine_similarity(v_sink_mean, v_new_mean, dim=-1).mean()

                    avg_cos_sim = (k_cos_sim + v_cos_sim)/2
                    # When the similarity is low, refresh the sink
                    if avg_cos_sim.min() < self.adapt_sink_thr:
                        idx = torch.argmin(avg_cos_sim)
                        sink_tokens = idx * frame_seqlen

                # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
                kv_cache_size = kv_cache["k"].shape[1]
                num_new_tokens = roped_query.shape[1]
                if c_start + num_new_tokens >= kv_cache_size:
                    kv_cache["global_end_index"][i].fill_(c_start)
                    kv_cache["local_end_index"][i].fill_(kv_cache_size)
                if (current_end > kv_cache["global_end_index"][i].item()) and (
                        num_new_tokens + kv_cache["local_end_index"][i].item() > kv_cache_size):
                    # Calculate the number of new tokens added in this step
                    # Shift existing cache content left to discard oldest tokens
                    # Clone the source slice to avoid overlapping memory error
                    num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"][i].item() - kv_cache_size
                    num_rolled_tokens = kv_cache["local_end_index"][i].item() - num_evicted_tokens - sink_tokens
                    kv_cache["k"][i:i+1, sink_tokens:sink_tokens + num_rolled_tokens] = \
                        kv_cache["k"][i:i+1, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                    kv_cache["v"][i:i+1, sink_tokens:sink_tokens + num_rolled_tokens] = \
                        kv_cache["v"][i:i+1, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                    # Insert the new keys/values at the end
                    local_end_index = kv_cache["local_end_index"][i].item() + current_end - \
                        kv_cache["global_end_index"][i].item() - num_evicted_tokens
                else:
                    local_end_index = kv_cache["local_end_index"][i].item() + current_end - kv_cache["global_end_index"][i].item()

                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][i:i+1, local_start_index:local_end_index] = roped_key[i:i+1]
                kv_cache["v"][i:i+1, local_start_index:local_end_index] = v[i:i+1]

                seq_lens.append(local_end_index)

                kv_cache["global_end_index"][i].fill_(current_end)
                kv_cache["local_end_index"][i].fill_(local_end_index)
            
            seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=roped_query.device)

            x = flash_attn_interface.flash_attn_with_kvcache(
                q=roped_query,
                k_cache=kv_cache["k"][:, :seq_lens.max()],
                v_cache=kv_cache["v"][:, :seq_lens.max()],
                cache_seqlens=seq_lens,
            )

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, window_size, qk_norm,
                                                eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        current_end=0
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        # assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen))
             * (1 + e[1]) + e[0]).flatten(1, 2),
            seq_lens, grid_sizes,
            freqs, block_mask, kv_cache, current_start, current_end)

        # with amp.autocast(dtype=torch.float32):
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen))
                 * e[2]).flatten(1, 2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(self.norm3(x), context,
                                    context_lens, crossattn_cache=crossattn_cache)
            y = self.ffn(
                (self.norm2(x).unflatten(dim=1, sizes=(num_frames,
                 frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
            )
            # with amp.autocast(dtype=torch.float32):
            x = x + (y.unflatten(dim=1, sizes=(num_frames,
                     frame_seqlen)) * e[5]).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        return x


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = (self.head(
            self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) *
            (1 + e[1]) + e[0]))
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                    window_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

        self.block_mask = None

        self.num_frame_per_block = 1

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        return block_mask

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        current_end: int = 0,
        block_mode: str = 'input',
        block_num: int = [-1],
        patched_x_shape: torch.Tensor = None,
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if block_mode == 'input':
            if y is not None:
                x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

            # embeddings
            x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
            bsz, cch, tlen, hh, ww = x[0].shape
            patched_x_shape = torch.tensor([bsz, cch, tlen, hh, ww], dtype=torch.int64, device=device)
        else:
            bsz, cch, tlen, hh, ww = [int(i) for i in patched_x_shape.tolist()]
            x = [u.permute(1,0).reshape(bsz, cch, tlen, hh, ww) for u in x]
            
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)
        """
        torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        """

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward
        
        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                assert False
            else:
                if (block_mode == 'output' or block_mode == 'middle') and block_index < block_num[0]:
                    continue
                if (block_mode == 'input' or block_mode == 'middle') and block_index == block_num[-1]:
                    return x, patched_x_shape
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "current_end": current_end
                    }
                )
                x = block(x, **kwargs)
        if block_mode == 'input' and block_num[-1] == len(self.blocks):
            return x, patched_x_shape

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # Construct blockwise causal attn mask
        if self.block_mask is None:
            self.block_mask = self._prepare_blockwise_causal_attn_mask(
                device, num_frames=x.shape[2],
                frame_seqlen=x.shape[-2] *
                x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                num_frame_per_block=self.num_frame_per_block
            )

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
````

## File: causvid/models/wan/causal_stream_inference.py

````python
from causvid.models import (
    get_diffusion_wrapper,
    get_text_encoder_wrapper,
    get_vae_wrapper
)
from typing import List
import torch
import torch.distributed as dist

class CausalStreamInferencePipeline(torch.nn.Module):
    def __init__(self, args, device):
        super().__init__()
        model_type = args.model_type
        self.device = device
        # Step 1: Initialize all models
        self.generator_model_name = getattr(
            args, "generator_name", args.model_name)
        self.generator = get_diffusion_wrapper(
            model_name=self.generator_model_name)(model_type=model_type)
        self.text_encoder = get_text_encoder_wrapper(
            model_name=args.model_name)(model_type=model_type)
        self.vae = get_vae_wrapper(model_name=args.model_name)(model_type=model_type)

        # Step 2: Initialize all causal hyperparmeters
        self._init_denoising_step_list(args, device)

        if model_type == "T2V-1.3B":
            self.num_transformer_blocks = 30
            self.num_heads = 12
        elif model_type == "T2V-14B":
            self.num_transformer_blocks = 40
            self.num_heads = 40
        else:
            raise ValueError(f"Model type {model_type} not supported")
        scale_size = 16
        self.height = args.height//scale_size*2
        self.width = args.width//scale_size*2
        self.frame_seq_length = (args.height//scale_size) * (args.width//scale_size)
        self.num_kv_cache = args.num_kv_cache
        self.kv_cache_length = self.frame_seq_length*self.num_kv_cache
        self.num_sink_tokens = args.num_sink_tokens
        self.adapt_sink_threshold = args.adapt_sink_threshold

        self.conditional_dict = None
        self.kv_cache1 = None
        self.kv_cache2 = None
        self.hidden_states = None
        self.block_x = None
        self.args = args
        self.num_frame_per_block = getattr(
            args, "num_frame_per_block", 1)

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.generator.model.to(self.device)

    def _init_denoising_step_list(self, args, device):
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long, device=device)
        assert self.denoising_step_list[-1] == 0
        # remove the last timestep (which equals zero)
        self.denoising_step_list = self.denoising_step_list[:-1]

        self.scheduler = self.generator.get_scheduler()
        if args.warp_denoising_step:  # Warp the denoising step according to the scheduler time shift
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))).cuda()
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        
        for i in range(self.num_transformer_blocks):
            cache_length = self.kv_cache_length
            self.generator.model.blocks[i].self_attn.sink_size = self.num_sink_tokens
            self.generator.model.blocks[i].self_attn.adapt_sink_thr = self.adapt_sink_threshold

            kv_cache1.append({
                "k": torch.zeros([batch_size, cache_length, self.num_heads, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, cache_length, self.num_heads, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, self.num_heads, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, self.num_heads, 128], dtype=dtype, device=device),
                "is_init": False,
            })

        self.crossattn_cache = crossattn_cache  # always store the clean cache
    
    def prepare(
        self,
        text_prompts: List[str],
        device: torch.device,
        dtype: torch.dtype,
        block_mode: str='input',
        noise: torch.Tensor = None,
        current_start: int = 0,
        current_end: int = None,
        block_num: torch.Tensor = None,
        batch_denoise: bool=True,
    ):
        self.device = device
        batch_size = noise.shape[0]

        self.conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        # Step 1: Initialize KV cache
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=dtype,
                device=device
            )

            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=dtype,
                device=device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
        
        current_start = torch.tensor([current_start], dtype=torch.long, device=device)
        current_end = torch.tensor([current_end], dtype=torch.long, device=device)

        for index, current_timestep in enumerate(self.denoising_step_list):
            # set current timestep
            timestep = torch.ones(
                [batch_size, noise.shape[1]], device=noise.device, dtype=torch.int64) * current_timestep

            if index < len(self.denoising_step_list) - 1:
                denoised_pred = self.generator(
                    noisy_image_or_video=noise,
                    conditional_dict=self.conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start,
                    current_end=current_end
                )
                next_timestep = self.denoising_step_list[index + 1]
                noise = self.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    torch.randn_like(denoised_pred.flatten(0, 1)),
                    next_timestep *
                    torch.ones([batch_size], device="cuda",
                                dtype=torch.long)
                ).unflatten(0, denoised_pred.shape[:2])
            else:
                # for getting real output
                denoised_pred = self.generator(
                    noisy_image_or_video=noise,
                    conditional_dict=self.conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start,
                    current_end=current_end
                )

        if not batch_denoise:
            return denoised_pred

        # Pre-allocate hidden_states tensor to avoid memory allocation during inference
        self.batch_size = len(self.denoising_step_list)

        # Determine which blocks to keep based on block_num range
        blocks_to_keep = []
        if block_num is not None:
            start_block, end_block = block_num[0].item(), block_num[1].item()
            blocks_to_keep = list(range(start_block, end_block))
        else:
            blocks_to_keep = list(range(self.num_transformer_blocks))

        # Process only the blocks in the specified range
        for i in range(self.num_transformer_blocks):
            if dist.is_initialized():
                dist.broadcast(self.crossattn_cache[i]['k'], src=0)
                dist.broadcast(self.crossattn_cache[i]['v'], src=0)
                dist.broadcast(self.kv_cache1[i]['k'], src=0)
                dist.broadcast(self.kv_cache1[i]['v'], src=0)

            self.kv_cache1[i]['k'] = self.kv_cache1[i]['k'].repeat(self.batch_size, 1, 1, 1)
            self.kv_cache1[i]['v'] = self.kv_cache1[i]['v'].repeat(self.batch_size, 1, 1, 1)

            self.kv_cache1[i]['global_end_index'] = self.kv_cache1[i]['global_end_index'].repeat(self.batch_size)
            self.kv_cache1[i]['local_end_index'] = self.kv_cache1[i]['local_end_index'].repeat(self.batch_size)

            self.crossattn_cache[i]['k'] = self.crossattn_cache[i]['k'].repeat(self.batch_size, 1, 1, 1)
            self.crossattn_cache[i]['v'] = self.crossattn_cache[i]['v'].repeat(self.batch_size, 1, 1, 1)
        
        # Remove blocks outside the range
        if block_num is not None:
            for i in range(self.num_transformer_blocks):
                if i not in blocks_to_keep:
                    self.kv_cache1[i]['k'] = self.kv_cache1[i]['k'].cpu()
                    self.kv_cache1[i]['v'] = self.kv_cache1[i]['v'].cpu()

        self.hidden_states = torch.zeros(
            (self.batch_size, self.num_frame_per_block, *noise.shape[2:]), dtype=noise.dtype, device=device
        )

        if block_mode in ['output', 'middle']:
            self.block_x = torch.zeros(
                (self.batch_size, self.frame_seq_length, self.num_heads*128), dtype=noise.dtype, device=device
            )
        else:
            self.block_x = None

        self.kv_cache_starts = torch.ones(self.batch_size, dtype=torch.long, device=device) * current_end
        self.kv_cache_ends = torch.ones(self.batch_size, dtype=torch.long, device=device) * current_end + self.frame_seq_length

        self.timestep = self.denoising_step_list

        self.conditional_dict['prompt_embeds'] = self.conditional_dict['prompt_embeds'].repeat(self.batch_size, 1, 1)
    
        return denoised_pred
    
    def inference_stream(self, noise: torch.Tensor, current_start: int, current_end: int, current_step: int) -> torch.Tensor:

        self.hidden_states[1:] = self.hidden_states[:-1].clone()
        self.hidden_states[0] = noise[0]

        self.kv_cache_starts[1:] = self.kv_cache_starts[:-1].clone()
        self.kv_cache_starts[0] = current_start
        
        self.kv_cache_ends[1:] = self.kv_cache_ends[:-1].clone()
        self.kv_cache_ends[0] = current_end

        if current_step is not None:
            self.timestep[0] = current_step
        
        self.hidden_states = self.generator(
            noisy_image_or_video=self.hidden_states,
            conditional_dict=self.conditional_dict,
            timestep=self.timestep.unsqueeze(1).expand(-1, self.hidden_states.shape[1]),
            kv_cache=self.kv_cache1,
            crossattn_cache=self.crossattn_cache,
            current_start=self.kv_cache_starts,
            current_end=self.kv_cache_ends,
        )

        for i in range(len(self.denoising_step_list) - 1):
            self.hidden_states[[i]] = self.scheduler.add_noise(
                self.hidden_states[[i]],
                torch.randn_like(self.hidden_states[[i]]),
                self.denoising_step_list[i + 1] *
                torch.ones([1], device="cuda",
                            dtype=torch.long)
            )

        return self.hidden_states
    
    def inference_wo_batch(self, noise: torch.Tensor, current_start: int, current_end: int, current_step: int) -> torch.Tensor:
        batch_size = noise.shape[0]

        current_start = torch.ones(batch_size, dtype=torch.long, device=self.device) * current_start
        current_end = torch.ones(batch_size, dtype=torch.long, device=self.device) * current_end

        # Step 2.1: Spatial denoising loop
        self.denoising_step_list[0] = current_step
        for index, current_timestep in enumerate(self.denoising_step_list):
            # set current timestep
            timestep = torch.ones(
                [batch_size, noise.shape[1]], device=noise.device, dtype=torch.int64) * current_timestep

            if index < len(self.denoising_step_list) - 1:
                denoised_pred = self.generator(
                    noisy_image_or_video=noise,
                    conditional_dict=self.conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start,
                    current_end=current_end
                )
                next_timestep = self.denoising_step_list[index + 1]
                noise = self.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    torch.randn_like(denoised_pred.flatten(0, 1)),
                    next_timestep *
                    torch.ones([batch_size], device="cuda",
                                dtype=torch.long)
                ).unflatten(0, denoised_pred.shape[:2])
            else:
                # for getting real output
                denoised_pred = self.generator(
                    noisy_image_or_video=noise,
                    conditional_dict=self.conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start,
                    current_end=current_end
                )

        return denoised_pred

    def inference(self, noise: torch.Tensor, current_start: int, current_end: int, \
        current_step: int, block_mode: str='input', block_num=None,\
            patched_x_shape: torch.Tensor=None, block_x: torch.Tensor=None) -> torch.Tensor:

        if block_mode == 'input':
            self.hidden_states[1:] = self.hidden_states[:-1].clone()
            self.hidden_states[0] = noise[0]

            self.kv_cache_starts[1:] = self.kv_cache_starts[:-1].clone()
            self.kv_cache_starts[0] = current_start
            
            self.kv_cache_ends[1:] = self.kv_cache_ends[:-1].clone()
            self.kv_cache_ends[0] = current_end
        else:
            self.block_x.copy_(block_x)
            self.hidden_states.copy_(noise)
            self.kv_cache_starts.copy_(current_start)
            self.kv_cache_ends.copy_(current_end)

        if current_step is not None:
            self.timestep[0] = current_step
        
        if block_mode == 'output':
            denoised_pred = self.generator.forward_output(
                noisy_image_or_video=self.hidden_states,
                conditional_dict=self.conditional_dict,
                timestep=self.timestep.unsqueeze(1).expand(-1, self.hidden_states.shape[1]),
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=self.kv_cache_starts,
                current_end=self.kv_cache_ends,
                block_mode=block_mode,
                block_num=block_num,
                patched_x_shape=patched_x_shape,
                block_x=self.block_x
            )

            for i in range(len(self.denoising_step_list) - 1):
                denoised_pred[[i]] = self.scheduler.add_noise(
                    denoised_pred[[i]],
                    torch.randn_like(denoised_pred[[i]]),
                    self.denoising_step_list[i + 1] *
                    torch.ones([1], device="cuda",
                                dtype=torch.long)
                )
            patched_x_shape = None

        else:
            denoised_pred, patched_x_shape = self.generator.forward_input(
                noisy_image_or_video=self.hidden_states,
                conditional_dict=self.conditional_dict,
                timestep=self.timestep.unsqueeze(1).expand(-1, self.hidden_states.shape[1]),
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=self.kv_cache_starts,
                current_end=self.kv_cache_ends,
                block_mode=block_mode,
                block_num=block_num,
                patched_x_shape=patched_x_shape,
                block_x=self.block_x,
            ) 

        return denoised_pred, patched_x_shape
````

## File: causvid/models/wan/flow_match.py

````python
"""
The following code is copied from https://github.com/modelscope/DiffSynth-Studio/blob/main/diffsynth/schedulers/flow_match.py
"""
import torch


class FlowMatchScheduler():

    def __init__(self, num_inference_steps=100, num_train_timesteps=1000, shift=3.0, sigma_max=1.0, sigma_min=0.003 / 1.002, inverse_timesteps=False, extra_one_step=False, reverse_sigmas=False):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False):
        sigma_start = self.sigma_min + \
            (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(
                sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(
                sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        self.sigmas = self.shift * self.sigmas / \
            (1 + (self.shift - 1) * self.sigmas)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) /
                          num_inference_steps) ** 2)
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * \
                (num_inference_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing

    def step(self, model_output, timestep, sample, to_final=False):
        self.sigmas = self.sigmas.to(model_output.device)
        self.timesteps = self.timesteps.to(model_output.device)
        timestep_id = torch.argmin(
            (self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma = self.sigmas[timestep_id].reshape(-1, 1, 1, 1)
        if to_final or (timestep_id + 1 >= len(self.timesteps)).any():
            sigma_ = 1 if (
                self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1].reshape(-1, 1, 1, 1)
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample

    def add_noise(self, original_samples, noise, timestep):
        """
        Diffusion forward corruption process.
        Input:
            - clean_latent: the clean latent with shape [B, C, H, W]
            - noise: the noise with shape [B, C, H, W]
            - timestep: the timestep with shape [B]
        Output: the corrupted latent with shape [B, C, H, W]
        """
        self.sigmas = self.sigmas.to(noise.device)
        self.timesteps = self.timesteps.to(noise.device)
        timestep_id = torch.argmin(
            (self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma = self.sigmas[timestep_id].reshape(-1, 1, 1, 1)
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample.type_as(noise)

    def training_target(self, sample, noise, timestep):
        target = noise - sample
        return target

    def training_weight(self, timestep):
        timestep_id = torch.argmin(
            (self.timesteps - timestep.to(self.timesteps.device)).abs())
        weights = self.linear_timesteps_weights[timestep_id]
        return weights
````

## File: causvid/models/wan/wan_base/README.md

````markdown
Code in this folder is modified from https://github.com/Wan-Video/Wan2.1
Apache-2.0 License 
````

## File: causvid/models/wan/wan_base/distributed/__init__.py

````python

````

## File: causvid/models/wan/wan_base/distributed/fsdp.py

````python
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from functools import partial

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy


def shard_model(
    model,
    device_id,
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
    process_group=None,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    sync_module_states=True,
):
    model = FSDP(
        module=model,
        process_group=process_group,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=partial(
            lambda_auto_wrap_policy, lambda_fn=lambda m: m in model.blocks),
        mixed_precision=MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype),
        device_id=device_id,
        sync_module_states=sync_module_states)
    return model
````

## File: causvid/models/wan/wan_base/distributed/xdit_context_parallel.py

````python
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.cuda.amp as amp
from xfuser.core.distributed import (get_sequence_parallel_rank,
                                     get_sequence_parallel_world_size,
                                     get_sp_group)
from xfuser.core.long_ctx_attention import xFuserLongContextAttention

from ..modules.model import sinusoidal_embedding_1d
from ..modules.attention import attention

import torch.distributed as dist
import math
import time


def pad_freqs(original_tensor, target_len):
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    padding_tensor = torch.ones(
        pad_size,
        s1,
        s2,
        dtype=original_tensor.dtype,
        device=original_tensor.device)
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor

def rope_apply(x, grid_sizes, freqs):
    """
    x:          [B, L, N, C].
    grid_sizes: [B, 3].
    freqs:      [M, C // 2].
    """
    s, n, c = x.size(1), x.size(2), x.size(3) // 2
    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(
            s, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        sp_size = get_sequence_parallel_world_size()
        sp_rank = get_sequence_parallel_rank()
        freqs_i = pad_freqs(freqs_i, s * sp_size)
        s_per_rank = s
        freqs_i_rank = freqs_i[(sp_rank * s_per_rank):((sp_rank + 1) *
                                                       s_per_rank), :, :]
        x_i = torch.view_as_real(x_i * freqs_i_rank).flatten(2)
        x_i = torch.cat([x_i, x[i, s:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()

def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    s, n, c = x.size(1), x.size(2), x.size(3) // 2
    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        sf = start_frame[i].item() if isinstance(start_frame, torch.Tensor) else start_frame

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            s, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][sf:sf + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        sp_size = get_sequence_parallel_world_size()
        sp_rank = get_sequence_parallel_rank()
        freqs_i = pad_freqs(freqs_i, s * sp_size)
        s_per_rank = s
        freqs_i_rank = freqs_i[(sp_rank * s_per_rank):((sp_rank + 1) *
                                                       s_per_rank), :, :]
        x_i = torch.view_as_real(x_i * freqs_i_rank).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)

def usp_dit_forward(
    self,
    x,
    t,
    context,
    seq_len,
    clip_fea=None,
    y=None,
    kv_cache: dict = None,
    crossattn_cache: dict = None,
    current_start: int = 0,
    current_end: int = 0
):
    """
    x:              A list of videos each with shape [C, T, H, W].
    t:              [B].
    context:        A list of text embeddings each with shape [L, C].
    """
    if self.model_type == 'i2v':
        assert clip_fea is not None and y is not None
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat(x)
    """
    torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                    dim=1) for u in x
    ])
    """

    # time embeddings
    # with amp.autocast(dtype=torch.float32):
    e = self.time_embedding(
        sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
    e0 = self.time_projection(e).unflatten(
        1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
    # assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat(
                [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    if clip_fea is not None:
        context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
        context = torch.concat([context_clip, context], dim=1)

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens,
        block_mask=self.block_mask
    )

    # Context Parallel
    x = torch.chunk(
        x, get_sequence_parallel_world_size(),
        dim=1)[get_sequence_parallel_rank()]

    for block_index, block in enumerate(self.blocks):
        kwargs.update(
            {
                "kv_cache": kv_cache[block_index],
                "crossattn_cache": crossattn_cache[block_index],
                "current_start": current_start,
                "current_end": current_end
            }
        )
        x = block(x, **kwargs)
    # head
    x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

    # Context Parallel
    x = get_sp_group().all_gather(x, dim=1)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return torch.stack(x)


def usp_attn_forward(self,
                     x,
                     seq_lens,
                     grid_sizes,
                     freqs,
                     block_mask,
                     kv_cache=None,
                     current_start=0,
                     current_end=0,
                     ):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)

    frame_seqlen = math.prod(grid_sizes[0][1:]).item()
    frame_seqlen = frame_seqlen // dist.get_world_size()
    current_start_frame = current_start // frame_seqlen
    roped_query = causal_rope_apply(
        q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
    roped_key = causal_rope_apply(
        k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)

    current_end = current_start + roped_query.shape[1]
    sink_tokens = self.sink_size * frame_seqlen
    # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
    kv_cache_size = kv_cache["k"].shape[1]
    num_new_tokens = roped_query.shape[1]
    if (current_end > kv_cache["global_end_index"].item()) and (
            num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
        # Calculate the number of new tokens added in this step
        # Shift existing cache content left to discard oldest tokens
        # Clone the source slice to avoid overlapping memory error
        num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
        num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
        kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
            kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
        kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
            kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
        # Insert the new keys/values at the end
        local_end_index = kv_cache["local_end_index"].item() + current_end - \
            kv_cache["global_end_index"].item() - num_evicted_tokens
        local_start_index = local_end_index - num_new_tokens
        kv_cache["k"][:, local_start_index:local_end_index] = roped_key
        kv_cache["v"][:, local_start_index:local_end_index] = v

    else:
        # Assign new keys/values directly up to current_end
        local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
        local_start_index = local_end_index - num_new_tokens
        kv_cache["k"][:, local_start_index:local_end_index] = roped_key
        kv_cache["v"][:, local_start_index:local_end_index] = v

    x = xFuserLongContextAttention()(
        None,
        roped_query,
        kv_cache["k"][:, :local_end_index],
        kv_cache["v"][:, :local_end_index],
        window_size=self.window_size
    )

    # x = attention(
    #     roped_query,
    #     kv_cache["k"][:, :local_end_index],
    #     kv_cache["v"][:, :local_end_index]
    # )
    kv_cache["global_end_index"].fill_(current_end)
    kv_cache["local_end_index"].fill_(local_end_index)

    # output
    x = x.flatten(2)
    x = self.o(x)
    return x
````

## File: causvid/models/wan/wan_base/image2video.py

````python
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .modules.clip import CLIPModel
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class WanI2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        init_on_cpu=True,
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.use_usp = use_usp
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        self.clip = CLIPModel(
            dtype=config.clip_dtype,
            device=self.device,
            checkpoint_path=os.path.join(checkpoint_dir,
                                         config.clip_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.clip_tokenizer))

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)

        if t5_fsdp or dit_fsdp or use_usp:
            init_on_cpu = False

        if use_usp:
            from xfuser.core.distributed import \
                get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (usp_attn_forward,
                                                            usp_dit_forward)
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            if not init_on_cpu:
                self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

    def generate(self,
                 input_prompt,
                 img,
                 max_area=720 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=40,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        F = frame_num
        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]

        max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            16,
            21,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        msk = torch.ones(1, 81, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ],
            dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # preprocess
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        self.clip.model.to(self.device)
        clip_context = self.clip.visual([img[:, None, :, :]])
        if offload_model:
            self.clip.model.cpu()

        y = self.vae.encode([
            torch.concat([
                torch.nn.functional.interpolate(
                    img[None].cpu(), size=(h, w), mode='bicubic').transpose(
                        0, 1),
                torch.zeros(3, 80, h, w)
            ],
                dim=1).to(self.device)
        ])[0]
        y = torch.concat([msk, y])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latent = noise

            arg_c = {
                'context': [context[0]],
                'clip_fea': clip_context,
                'seq_len': max_seq_len,
                'y': [y],
            }

            arg_null = {
                'context': context_null,
                'clip_fea': clip_context,
                'seq_len': max_seq_len,
                'y': [y],
            }

            if offload_model:
                torch.cuda.empty_cache()

            self.model.to(self.device)
            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = [latent.to(self.device)]
                timestep = [t]

                timestep = torch.stack(timestep).to(self.device)

                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0].to(
                        torch.device('cpu') if offload_model else self.device)
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0].to(
                        torch.device('cpu') if offload_model else self.device)
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                latent = latent.to(
                    torch.device('cpu') if offload_model else self.device)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latent = temp_x0.squeeze(0)

                x0 = [latent.to(self.device)]
                del latent_model_input, timestep

            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()

            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latent
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
````

## File: causvid/models/wan/wan_base/modules/attention.py

````python
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

import warnings

__all__ = [
    'flash_attention',
    'attention',
]


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
    attn_mask=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out
````

## File: causvid/models/wan/wan_base/modules/model.py

````python
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import flash_attention

__all__ = ['WanModel']


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


# @amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


# @amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens, crossattn_cache=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
            crossattn_cache (List[dict], *optional*): Contains the cached key and value tensors for context embedding.
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)

        if crossattn_cache is not None:
            if not crossattn_cache["is_init"]:
                crossattn_cache["is_init"] = True
                k = self.norm_k(self.k(context)).view(b, -1, n, d)
                v = self.v(context).view(b, -1, n, d)
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        else:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(
            dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.modulation + e).chunk(6, dim=1)
        # assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x) * (1 + e[1]) + e[0], seq_lens, grid_sizes,
            freqs)
        # with amp.autocast(dtype=torch.float32):
        x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
            # with amp.autocast(dtype=torch.float32):
            x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).type_as(x))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
````

## File: causvid/models/wan/wan_base/modules/vae.py

````python
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging

import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

__all__ = [
    'WanVAE',
]

CACHE_T = 2


class CausalConv3d(nn.Conv3d):
    """
    Causal 3d convolusion.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1],
                         self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)

        return super().forward(x)


class RMS_norm(nn.Module):

    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.

    def forward(self, x):
        return F.normalize(
            x, dim=(1 if self.channel_first else
                    -1)) * self.scale * self.gamma + self.bias


class Upsample(nn.Upsample):

    def forward(self, x):
        """
        Fix bfloat16 support for nearest neighbor interpolation.
        """
        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):

    def __init__(self, dim, mode):
        assert mode in ('none', 'upsample2d', 'upsample3d', 'downsample2d',
                        'downsample3d')
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == 'upsample2d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
        elif mode == 'upsample3d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
            self.time_conv = CausalConv3d(
                dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

        elif mode == 'downsample2d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == 'downsample3d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == 'upsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = 'Rep'
                    feat_idx[0] += 1
                else:

                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < CACHE_T and feat_cache[
                            idx] is not None and feat_cache[idx] != 'Rep':
                        # cache last frame of last two chunk
                        cache_x = torch.cat([
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                                cache_x.device), cache_x
                        ],
                            dim=2)
                    if cache_x.shape[2] < CACHE_T and feat_cache[
                            idx] is not None and feat_cache[idx] == 'Rep':
                        cache_x = torch.cat([
                            torch.zeros_like(cache_x).to(cache_x.device),
                            cache_x
                        ],
                            dim=2)
                    if feat_cache[idx] == 'Rep':
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]),
                                    3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.resample(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)

        if self.mode == 'downsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:

                    cache_x = x[:, :, -1:, :, :].clone()
                    # if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None and feat_cache[idx]!='Rep':
                    #     # cache last frame of last two chunk
                    #     cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)

                    x = self.time_conv(
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x

    def init_weight(self, conv):
        conv_weight = conv.weight
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        one_matrix = torch.eye(c1, c2)
        init_matrix = one_matrix
        nn.init.zeros_(conv_weight)
        # conv_weight.data[:,:,-1,1,1] = init_matrix * 0.5
        conv_weight.data[:, :, 1, 0, 0] = init_matrix  # * 0.5
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def init_weight2(self, conv):
        conv_weight = conv.weight.data
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        init_matrix = torch.eye(c1 // 2, c2)
        # init_matrix = repeat(init_matrix, 'o ... -> (o 2) ...').permute(1,0,2).contiguous().reshape(c1,c2)
        conv_weight[:c1 // 2, :, -1, 0, 0] = init_matrix
        conv_weight[c1 // 2:, :, -1, 0, 0] = init_matrix
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)


class ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # layers
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False), nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False), nn.SiLU(), nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1))
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) \
            if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h


class AttentionBlock(nn.Module):
    """
    Causal self-attention with a single head.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.norm(x)
        # compute query, key, value
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3,
                                         -1).permute(0, 1, 3,
                                                     2).contiguous().chunk(
                                                         3, dim=-1)

        # apply attention
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
        )
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)

        # output
        x = self.proj(x)
        x = rearrange(x, '(b t) c h w-> b c t h w', t=t)
        return x + identity


class Encoder3d(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # downsample block
            if i != len(dim_mult) - 1:
                mode = 'downsample3d' if temperal_downsample[
                    i] else 'downsample2d'
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout), AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout))

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                        cache_x.device), cache_x
                ],
                    dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        # downsamples
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        # middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        # head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


class Decoder3d(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_upsample=[False, True, True],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2**(len(dim_mult) - 2)

        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout), AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout))

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # upsample block
            if i != len(dim_mult) - 1:
                mode = 'upsample3d' if temperal_upsample[i] else 'upsample2d'
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, 3, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        # conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                        cache_x.device), cache_x
                ],
                    dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        # middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        # upsamples
        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        # head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


def count_conv3d(model):
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1
    return count


class WanVAE_(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        # modules
        self.encoder = Encoder3d(dim, z_dim * 2, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_downsample, dropout)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dim, z_dim, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_upsample, dropout)
        self.first_encode = True
        self.first_decode = True

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def encode(self, x, scale):
        self.clear_cache()
        # cache
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        # 对encode输入的x，按时间拆分为1、4、4、4....
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(
                    x[:, :, :1, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx)
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx)
                out = torch.cat([out, out_], 2)
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if isinstance(scale[0], torch.Tensor):
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                1, self.z_dim, 1, 1, 1)
        else:
            mu = (mu - scale[0]) * scale[1]
        self.clear_cache()
        return mu

    def stream_encode(self, x, scale):
        # cache
        t = x.shape[2]
        if self.first_encode:
            self.first_encode = False
            self.clear_cache_encode()
            self._enc_conv_idx = [0]
            out = self.encoder(
                x[:, :, :1, :, :],
                feat_cache=self._enc_feat_map,
                feat_idx=self._enc_conv_idx,
                )
            self._enc_conv_idx = [0]
            out_ = self.encoder(
                x[:, :, 1:, :, :],
                feat_cache=self._enc_feat_map,
                feat_idx=self._enc_conv_idx,
                )
            out = torch.cat([out, out_], 2)
        else:
            out=[]
            for i in range(t//4):
                self._enc_conv_idx = [0]
                out.append(self.encoder(
                    x[:, :, i*4:(i+1)*4, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                    ))
            out = torch.cat(out, 2)
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if scale is not None:
            if isinstance(scale[0], torch.Tensor):
                mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                    1, self.z_dim, 1, 1, 1)
            else:
                mu = (mu - scale[0]) * scale[1]
        # self.clear_cache()
        return mu

    def decode(self, z, scale):
        self.clear_cache()
        # z: [b,c,t,h,w]
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1)
        else:
            z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        x = self.conv2(z)
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx)
            else:
                out_ = self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx)
                out = torch.cat([out, out_], 2)
        self.clear_cache()
        return out

    def stream_decode(self, z, scale):
        # z: [b,c,t,h,w]
        t=z.shape[2]
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1)
        else:
            z = z / scale[1] + scale[0]
        x = self.conv2(z)
        if self.first_decode:
            self.first_decode = False
            self.clear_cache_decode()
            self.first_batch = False
            self._conv_idx = [0]
            out = self.decoder(
                x[:, :, :1, :, :],
                feat_cache=self._feat_map,
                feat_idx=self._conv_idx,
                )
            self._conv_idx = [0]
            out_ = self.decoder(
                x[:, :, 1:, :, :],
                feat_cache=self._feat_map,
                feat_idx=self._conv_idx,
                )
            out = torch.cat([out, out_], 2)
        else:
            out = []
            for i in range(t):
                self._conv_idx = [0]
                out.append(self.decoder(
                    x[:, :, i:(i+1), :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                    ))
            out = torch.cat(out, 2)
        # self.clear_cache()
        return out

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self, imgs, deterministic=False):
        mu, log_var = self.encode(imgs)
        if deterministic:
            return mu
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))
        return mu + std * torch.randn_like(std)

    def clear_cache(self):
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        # cache encode
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num

    def clear_cache_decode(self):
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
    
    def clear_cache_encode(self):
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num
    


def _video_vae(pretrained_path=None, z_dim=None, device='cpu', **kwargs):
    """
    Autoencoder3d adapted from Stable Diffusion 1.x, 2.x and XL.
    """
    # params
    cfg = dict(
        dim=96,
        z_dim=z_dim,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0)
    cfg.update(**kwargs)

    # init model
    with torch.device('meta'):
        model = WanVAE_(**cfg)

    # load checkpoint
    logging.info(f'loading {pretrained_path}')
    model.load_state_dict(
        torch.load(pretrained_path, map_location=device), assign=True)

    return model


class WanVAE:

    def __init__(self,
                 z_dim=16,
                 vae_pth='cache/vae_step_411000.pth',
                 dtype=torch.float,
                 device="cuda"):
        self.dtype = dtype
        self.device = device

        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=dtype, device=device)
        self.std = torch.tensor(std, dtype=dtype, device=device)
        self.scale = [self.mean, 1.0 / self.std]

        # init model
        self.model = _video_vae(
            pretrained_path=vae_pth,
            z_dim=z_dim,
        ).eval().requires_grad_(False).to(device)

    def encode(self, videos):
        """
        videos: A list of videos each with shape [C, T, H, W].
        """
        with amp.autocast(dtype=self.dtype):
            return [
                self.model.encode(u.unsqueeze(0), self.scale).float().squeeze(0)
                for u in videos
            ]

    def decode(self, zs):
        with amp.autocast(dtype=self.dtype):
            return [
                self.model.decode(u.unsqueeze(0),
                                  self.scale).float().clamp_(-1, 1).squeeze(0)
                for u in zs
            ]
````

## File: causvid/models/wan/wan_base/text2video.py

````python
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class WanT2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)

        if use_usp:
            from xfuser.core.distributed import \
                get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (usp_attn_forward,
                                                            usp_dit_forward)
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

    def generate(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                self.model.to(self.device)
                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0]

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

            x0 = latents
            if offload_model:
                self.model.cpu()
            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
````

## File: causvid/models/wan/wan_wrapper.py

````python
from causvid.models.model_interface import (
    DiffusionModelInterface,
    TextEncoderInterface,
    VAEInterface
)
from causvid.models.wan.wan_base.modules.tokenizers import HuggingfaceTokenizer
from causvid.models.wan.wan_base.modules.model import WanModel
from causvid.models.wan.wan_base.modules.vae import _video_vae
from causvid.models.wan.wan_base.modules.t5 import umt5_xxl
from causvid.models.wan.flow_match import FlowMatchScheduler
from causvid.models.wan.causal_model import CausalWanModel
from typing import List, Tuple, Dict, Optional
import torch
import os
import torch.distributed as dist
import time

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


class WanTextEncoder(TextEncoderInterface):
    def __init__(self, model_type="T2V-1.3B") -> None:
        super().__init__()

        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device=torch.device('cpu')
        ).eval().requires_grad_(False)
        self.text_encoder.load_state_dict(
            torch.load(
                os.path.join(repo_root, f"wan_models/Wan2.1-{model_type}/models_t5_umt5-xxl-enc-bf16.pth"),
                map_location='cpu', weights_only=False
            )
        )

        self.tokenizer = HuggingfaceTokenizer(
            name=os.path.join(repo_root, f"wan_models/Wan2.1-{model_type}/google/umt5-xxl/"), seq_len=512, clean='whitespace')

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)

        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0

        return {
            "prompt_embeds": context
        }


class WanVAEWrapper(VAEInterface):
    def __init__(self, model_type="T2V-1.3B"):
        super().__init__()
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        # init model
        self.model = _video_vae(
            pretrained_path=os.path.join(repo_root, f"wan_models/Wan2.1-{model_type}/Wan2.1_VAE.pth"),
            z_dim=16,
        ).eval().requires_grad_(False)

    def decode_to_pixel(self, latent: torch.Tensor) -> torch.Tensor:
        # from [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        zs = latent.permute(0, 2, 1, 3, 4)

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        output = [
            self.model.decode(u.unsqueeze(0),
                              scale).float().clamp_(-1, 1).squeeze(0)
            for u in zs
        ]
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        # from [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        zs = latent.permute(0, 2, 1, 3, 4)

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        output = self.model.decode(zs, scale).clamp_(-1, 1)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        # output = output.permute(0, 2, 1, 3, 4)
        return output
    
    def stream_encode(self, video: torch.Tensor, is_scale=False) -> torch.Tensor:
        if is_scale:
            device, dtype = video.device, video.dtype
            scale = [self.mean.to(device=device, dtype=dtype),
                    1.0 / self.std.to(device=device, dtype=dtype)]
        else:
            scale = None
        return self.model.stream_encode(video, scale)
    
    def stream_decode_to_pixel(self, latent: torch.Tensor) -> torch.Tensor:
        zs = latent.permute(0, 2, 1, 3, 4)
        zs = zs.to(torch.bfloat16).to('cuda')
        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]
        output = self.model.stream_decode(zs, scale).float().clamp_(-1, 1)
        output = output.permute(0, 2, 1, 3, 4)
        return output


class WanDiffusionWrapper(DiffusionModelInterface):
    def __init__(self, model_type="T2V-1.3B"):
        super().__init__()

        self.model = WanModel.from_pretrained(os.path.join(repo_root, f"wan_models/Wan2.1-{model_type}/"))
        self.model.eval()

        self.uniform_timestep = True

        self.scheduler = FlowMatchScheduler(
            shift=8.0, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        self.seq_len = 32760  # [1, 21, 16, 60, 104]
        super().post_init()

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt,
                                                        self.scheduler.sigmas,
                                                        self.scheduler.timesteps]
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device), [x0_pred, xt,
                                                      scheduler.sigmas,
                                                      scheduler.timesteps]
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def forward(
        self, noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor, kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        current_end: Optional[int] = None
    ) -> torch.Tensor:
        prompt_embeds = conditional_dict["prompt_embeds"]

        # [B, F] -> [B]
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        if kv_cache is not None:
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                current_end=current_end
            ).permute(0, 2, 1, 3, 4)
        else:
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len
            ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, flow_pred.shape[:2])

        return pred_x0

    def forward_input(
        self, noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor,block_mode: str='input', block_num = None, kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        current_end: Optional[int] = None,
        patched_x_shape: torch.Tensor = None,
        block_x: torch.Tensor = None,
    ) -> torch.Tensor:
        assert kv_cache is not None, "kv_cache must be provided"

        prompt_embeds = conditional_dict["prompt_embeds"]

        # [B, F] -> [B]
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep
        
        if block_x is not None and block_mode == 'middle':
            noisy_image_or_video = block_x
        else:
            noisy_image_or_video = noisy_image_or_video.permute(0, 2, 1, 3, 4)

        output, patched_x_shape = self.model(
            noisy_image_or_video,
            t=input_timestep, context=prompt_embeds,
            seq_len=self.seq_len,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=current_start,
            current_end=current_end,
            block_mode=block_mode,
            block_num=block_num,
            patched_x_shape=patched_x_shape,
        )

        return output, patched_x_shape

    def forward_output(
        self, noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor, block_mode: str='output', block_num = None, kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        current_end: Optional[int] = None,
        patched_x_shape: torch.Tensor = None,
        block_x: torch.Tensor = None,
    ) -> torch.Tensor:
        assert kv_cache is not None, "kv_cache must be provided"

        prompt_embeds = conditional_dict["prompt_embeds"]

        # [B, F] -> [B]
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        flow_pred = self.model(
            block_x,
            t=input_timestep, context=prompt_embeds,
            seq_len=self.seq_len,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=current_start,
            current_end=current_end,
            block_mode=block_mode,
            block_num=block_num,
            patched_x_shape=patched_x_shape,
        ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, flow_pred.shape[:2])

        return pred_x0


class CausalWanDiffusionWrapper(WanDiffusionWrapper):
    def __init__(self, model_type="T2V-1.3B"):
        super().__init__()

        self.model = CausalWanModel.from_pretrained(
            os.path.join(repo_root, f"wan_models/Wan2.1-{model_type}/"))
        self.model.eval()

        self.uniform_timestep = False
````

## File: causvid/scheduler.py

````python
from abc import abstractmethod, ABC
import torch


class SchedulerInterface(ABC):
    """
    Base class for diffusion noise schedule.
    """
    alphas_cumprod: torch.Tensor  # [T], alphas for defining the noise schedule

    @abstractmethod
    def add_noise(
        self, clean_latent: torch.Tensor,
        noise: torch.Tensor, timestep: torch.Tensor
    ):
        """
        Diffusion forward corruption process.
        Input:
            - clean_latent: the clean latent with shape [B, C, H, W]
            - noise: the noise with shape [B, C, H, W]
            - timestep: the timestep with shape [B]
        Output: the corrupted latent with shape [B, C, H, W]
        """
        pass

    def convert_x0_to_noise(
        self, x0: torch.Tensor, xt: torch.Tensor,
        timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert the diffusion network's x0 prediction to noise predidction.
        x0: the predicted clean data with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        noise = (xt-sqrt(alpha_t)*x0) / sqrt(beta_t) (eq 11 in https://arxiv.org/abs/2311.18828)
        """
        # use higher precision for calculations
        original_dtype = x0.dtype
        x0, xt, alphas_cumprod = map(
            lambda x: x.double().to(x0.device), [x0, xt,
                                                 self.alphas_cumprod]
        )

        alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t

        noise_pred = (xt - alpha_prod_t **
                      (0.5) * x0) / beta_prod_t ** (0.5)
        return noise_pred.to(original_dtype)

    def convert_noise_to_x0(
        self, noise: torch.Tensor, xt: torch.Tensor,
        timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert the diffusion network's noise prediction to x0 predidction.
        noise: the predicted noise with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        x0 = (x_t - sqrt(beta_t) * noise) / sqrt(alpha_t) (eq 11 in https://arxiv.org/abs/2311.18828)
        """
        # use higher precision for calculations
        original_dtype = noise.dtype
        noise, xt, alphas_cumprod = map(
            lambda x: x.double().to(noise.device), [noise, xt,
                                                    self.alphas_cumprod]
        )
        alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t

        x0_pred = (xt - beta_prod_t **
                   (0.5) * noise) / alpha_prod_t ** (0.5)
        return x0_pred.to(original_dtype)

    def convert_velocity_to_x0(
        self, velocity: torch.Tensor, xt: torch.Tensor,
        timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert the diffusion network's velocity prediction to x0 predidction.
        velocity: the predicted noise with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        v = sqrt(alpha_t) * noise - sqrt(beta_t) x0
        noise = (xt-sqrt(alpha_t)*x0) / sqrt(beta_t)
        given v, x_t, we have
        x0 = sqrt(alpha_t) * x_t - sqrt(beta_t) * v
        see derivations https://chatgpt.com/share/679fb6c8-3a30-8008-9b0e-d1ae892dac56
        """
        # use higher precision for calculations
        original_dtype = velocity.dtype
        velocity, xt, alphas_cumprod = map(
            lambda x: x.double().to(velocity.device), [velocity, xt,
                                                       self.alphas_cumprod]
        )
        alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t

        x0_pred = (alpha_prod_t ** 0.5) * xt - (beta_prod_t ** 0.5) * velocity
        return x0_pred.to(original_dtype)
````

## File: causvid/util.py

````python
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.fsdp import (
    FullStateDictConfig,
    StateDictType,
    MixedPrecision,
    ShardingStrategy,
    FullyShardedDataParallel as FSDP
)
from torchvision.utils import make_grid
from datetime import timedelta, datetime
import torch.distributed as dist
from omegaconf import OmegaConf
from functools import partial
import numpy as np
import random
import torch
import wandb
import os


def launch_distributed_job(backend: str = "nccl"):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    host = os.environ["MASTER_ADDR"]
    port = int(os.environ["MASTER_PORT"])

    if ":" in host:  # IPv6
        init_method = f"tcp://[{host}]:{port}"
    else:  # IPv4
        init_method = f"tcp://{host}:{port}"
    dist.init_process_group(rank=rank, world_size=world_size, backend=backend,
                            init_method=init_method, timeout=timedelta(minutes=30))
    torch.cuda.set_device(local_rank)


def set_seed(seed: int, deterministic: bool = False):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.

    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)


def init_logging_folder(args):
    date = str(datetime.now()).replace(" ", "-").replace(":", "-")
    output_path = os.path.join(
        args.output_path,
        f"{date}_seed{args.seed}"
    )
    os.makedirs(output_path, exist_ok=False)

    os.makedirs(args.output_path, exist_ok=True)
    wandb.login(host=args.wandb_host, key=args.wandb_key)
    # Use wandb_mode from config, default to "online" if not specified
    wandb_mode = getattr(args, 'wandb_mode', 'online')
    run = wandb.init(config=OmegaConf.to_container(args, resolve=True), dir=args.output_path, **
                     {"mode": wandb_mode, "entity": args.wandb_entity, "project": args.wandb_project})
    wandb.run.log_code(".")
    wandb.run.name = args.wandb_name
    print(f"run dir: {run.dir}")
    wandb_folder = run.dir
    os.makedirs(wandb_folder, exist_ok=True)

    return output_path, wandb_folder


def fsdp_wrap(module, sharding_strategy="full", mixed_precision=False, wrap_strategy="size", min_num_params=int(5e7), transformer_module=None):
    if mixed_precision:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
            cast_forward_inputs=False
        )
    else:
        mixed_precision_policy = None

    if wrap_strategy == "transformer":
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_module
        )
    elif wrap_strategy == "size":
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=min_num_params
        )
    else:
        raise ValueError(f"Invalid wrap strategy: {wrap_strategy}")

    os.environ["NCCL_CROSS_NIC"] = "1"

    sharding_strategy = {
        "full": ShardingStrategy.FULL_SHARD,
        "hybrid_full": ShardingStrategy.HYBRID_SHARD,
        "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
        "no_shard": ShardingStrategy.NO_SHARD,
    }[sharding_strategy]

    module = FSDP(
        module,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision_policy,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        sync_module_states=False  # Load ckpt on rank 0 and sync to other ranks
    )
    return module


def cycle(dl):
    while True:
        for data in dl:
            yield data


def fsdp_state_dict(model):
    fsdp_fullstate_save_policy = FullStateDictConfig(
        offload_to_cpu=True, rank0_only=True
    )
    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, fsdp_fullstate_save_policy
    ):
        checkpoint = model.state_dict()

    return checkpoint


def barrier():
    if dist.is_initialized():
        dist.barrier()


def prepare_for_saving(tensor, fps=16, caption=None):
    # Convert range [-1, 1] to [0, 1]
    tensor = (tensor * 0.5 + 0.5).clamp(0, 1).detach()

    if tensor.ndim == 4:
        # Assuming it's an image and has shape [batch_size, 3, height, width]
        tensor = make_grid(tensor, 4, padding=0, normalize=False)
        return wandb.Image((tensor * 255).cpu().numpy().astype(np.uint8), caption=caption)
    elif tensor.ndim == 5:
        # Assuming it's a video and has shape [batch_size, num_frames, 3, height, width]
        return wandb.Video((tensor * 255).cpu().numpy().astype(np.uint8), fps=fps, format="webm", caption=caption)
    else:
        raise ValueError("Unsupported tensor shape for saving. Expected 4D (image) or 5D (video) tensor.")
````

## File: configs/sdxl_8node_dmd_config.yaml

````yaml
model_name: sdxl
generator_grad:
  model: true
real_score_grad:
  model: false
fake_score_grad:
  model: true
denoising_step_list:
- 999
- 749
- 499
- 249
num_train_timestep: 1000
real_guidance_scale: 8.0
generator_task: image
denoising_loss_type: noise
mixed_precision: true
seed: 0
wandb_host: WANDB_HOST
wandb_key: WANDB_KEY
wandb_entity: tyin
wandb_project: causvid
wandb_name: causvid_sdxl_test_run_1node
sharding_strategy: full
lr: 5.0e-07
beta1: 0.9
beta2: 0.999
data_path: captions_laion6.25.txt
batch_size: 2
log_iters: 1000
negative_prompt: ''
dfake_gen_update_ratio: 5
image_or_video_shape:
- 2
- 1
- 4
- 128
- 128
output_path: /mnt/localssd/sdxl_logs
distillation_loss: dmd
gradient_checkpointing: false
warp_denoising_step: false
````

## File: configs/wan_bidirectional_dmd.yaml

````yaml
model_name: wan
generator_fsdp_wrap_strategy: size
real_score_fsdp_wrap_strategy: size
fake_score_fsdp_wrap_strategy: size
text_encoder_fsdp_wrap_strategy: size
generator_grad:
  model: true
real_score_grad:
  model: false
fake_score_grad:
  model: true
denoising_step_list:
- 1000
- 757
- 522
num_train_timestep: 1000
timestep_shift: 8.0
real_guidance_scale: 3.5
generator_task: bidirectional_video
real_task_type: bidirectional_video
fake_task_type: bidirectional_video
denoising_loss_type: flow
mixed_precision: true
seed: 0
wandb_host: WANDB_HOST
wandb_key: WANDB_KEY
wandb_entity: tyin
wandb_project: causvid
wandb_name: wan_bidirectional_dmd
sharding_strategy: hybrid_full
lr: 2.0e-06
beta1: 0.9
beta2: 0.999
data_path: sample_dataset/mixkit_prompts.txt
batch_size: 1
log_iters: 200
negative_prompt: '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'
dfake_gen_update_ratio: 5
image_or_video_shape:
- 1
- 21
- 16
- 60
- 104
output_path: /mnt/localssd/wan_bidirectional_dmd
distillation_loss: dmd
gradient_checkpointing: true
warp_denoising_step: false
````

## File: configs/wan_bidirectional_dmd_from_scratch.yaml

````yaml
model_name: wan
generator_fsdp_wrap_strategy: size
real_score_fsdp_wrap_strategy: size
fake_score_fsdp_wrap_strategy: size
text_encoder_fsdp_wrap_strategy: size
generator_grad:
  model: true
real_score_grad:
  model: false
fake_score_grad:
  model: true
denoising_step_list:
- 1000
- 757
- 522
num_train_timestep: 1000
timestep_shift: 8.0
real_guidance_scale: 3.5
generator_task: bidirectional_video
real_task_type: bidirectional_video
fake_task_type: bidirectional_video
denoising_loss_type: flow
mixed_precision: true
seed: 0
wandb_host: WANDB_HOST
wandb_key: WANDB_KEY
wandb_entity: tyin
wandb_project: causvid
wandb_name: wan_bidirectional_dmd_from_scratch
sharding_strategy: hybrid_full
lr: 2.0e-06
beta1: 0.9
beta2: 0.999
data_path: mixkit_latents_lmdb
batch_size: 1
log_iters: 200
negative_prompt: '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'
dfake_gen_update_ratio: 5
image_or_video_shape:
- 1
- 21
- 16
- 60
- 104
output_path: /mnt/localssd/wan_bidirectional_dmd_from_scratch
distillation_loss: dmd
gradient_checkpointing: true
backward_simulation: false
warp_denoising_step: false
````

## File: configs/wan_causal_dmd_v2v.yaml

````yaml
model_name: wan
generator_name: causal_wan
generator_ckpt: "ckpts/autoregressive_checkpoint/model.pt"
generator_fsdp_wrap_strategy: size
real_score_fsdp_wrap_strategy: size
fake_score_fsdp_wrap_strategy: size
text_encoder_fsdp_wrap_strategy: size
generator_grad:
  model: true
real_score_grad:
  model: false
fake_score_grad:
  model: true
denoising_step_list:
- 700
- 500
- 0
num_train_timestep: 1000
timestep_shift: 8.0
real_guidance_scale: 3.5
generator_task: causal_video
real_task_type: bidirectional_video
fake_task_type: bidirectional_video
denoising_loss_type: flow
mixed_precision: true
seed: 0
wandb_host: WANDB_HOST
wandb_key: WANDB_KEY
wandb_entity: tyin
wandb_project: causvid
wandb_name: wan_causal_dmd
sharding_strategy: hybrid_full
lr: 2.0e-06
beta1: 0.9
beta2: 0.999
data_path: mixkit_latents_lmdb
batch_size: 1
log_iters: 200
negative_prompt: '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'
dfake_gen_update_ratio: 5
image_or_video_shape:
- 1
- 21
- 16
- 60
- 104
output_path: /mnt/localssd/wan_causal_dmd
distillation_loss: dmd
gradient_checkpointing: true
backward_simulation: false
num_frame_per_block: 1
num_kv_cache: 6
num_sink_tokens: 3
adapt_sink_threshold: 0.2
warp_denoising_step: false
````

## File: configs/wan_causal_dmd_v2v_14b.yaml

````yaml
model_name: wan
generator_name: causal_wan
generator_ckpt: "ckpts/autoregressive_checkpoint/model.pt"
generator_fsdp_wrap_strategy: size
real_score_fsdp_wrap_strategy: size
fake_score_fsdp_wrap_strategy: size
text_encoder_fsdp_wrap_strategy: size
generator_grad:
  model: true
real_score_grad:
  model: false
fake_score_grad:
  model: true
denoising_step_list:
- 700
- 500
- 0
num_train_timestep: 1000
timestep_shift: 8.0
real_guidance_scale: 3.5
generator_task: causal_video
real_task_type: bidirectional_video
fake_task_type: bidirectional_video
denoising_loss_type: flow
mixed_precision: true
seed: 0
wandb_host: WANDB_HOST
wandb_key: WANDB_KEY
wandb_entity: tyin
wandb_project: causvid
wandb_name: wan_causal_dmd
sharding_strategy: full
lr: 2.0e-06
beta1: 0.9
beta2: 0.999
data_path: mixkit_latents_lmdb
batch_size: 1
log_iters: 200
negative_prompt: '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'
dfake_gen_update_ratio: 5
image_or_video_shape:
- 1
- 21
- 16
- 60
- 104
output_path: /mnt/localssd/wan_causal_dmd
distillation_loss: dmd
gradient_checkpointing: true
backward_simulation: false
num_frame_per_block: 1
num_kv_cache: 6
num_sink_tokens: 3
adapt_sink_threshold: 0.2
warp_denoising_step: false
````

## File: configs/wan_causal_dmd_warp_4step_cfg2.yaml

````yaml
model_name: wan
generator_name: causal_wan
generator_ckpt: "wan_causal_ode_checkpoint_model_003000/model.pt"
generator_fsdp_wrap_strategy: size
real_score_fsdp_wrap_strategy: size
fake_score_fsdp_wrap_strategy: size
text_encoder_fsdp_wrap_strategy: size
generator_grad:
  model: true
real_score_grad:
  model: false
fake_score_grad:
  model: true
denoising_step_list:
- 1000
- 750
- 500
- 250
- 0
num_train_timestep: 1000
timestep_shift: 8.0
real_guidance_scale: 2.0
generator_task: causal_video
real_task_type: bidirectional_video
fake_task_type: bidirectional_video
denoising_loss_type: flow
mixed_precision: true
seed: 0
wandb_host: WANDB_HOST
wandb_key: WANDB_KEY
wandb_entity: tyin
wandb_project: causvid
wandb_name: wan_causal_dmd
sharding_strategy: hybrid_full
lr: 2.0e-06
beta1: 0.9
beta2: 0.999
data_path: mixkit_latents_lmdb
batch_size: 1
log_iters: 200
negative_prompt: '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'
dfake_gen_update_ratio: 5
image_or_video_shape:
- 1
- 21
- 16
- 60
- 104
output_path: /mnt/localssd/wan_causal_dmd_warp_4step
distillation_loss: dmd
gradient_checkpointing: true
backward_simulation: false
num_frame_per_block: 3
warp_denoising_step: true
````

## File: configs/wan_causal_ode.yaml

````yaml
model_name: causal_wan
generator_grad:
  model: true
denoising_step_list:
- 1000
- 757
- 522
- 0
generator_task: causal_video
generator_fsdp_wrap_strategy: size
text_encoder_fsdp_wrap_strategy: size
mixed_precision: true
seed: 0
wandb_host: WANDB_HOST
wandb_key: WANDB_KEY
wandb_entity: tyin
wandb_project: causvid
wandb_name: wan_causal_ode
sharding_strategy: hybrid_full
lr: 2.0e-06
beta1: 0.9
beta2: 0.999
data_path: mixkit_ode_lmdb
batch_size: 2
log_iters: 200
output_path: /mnt/localssd/wan_causal_ode
distillation_loss: ode
gradient_checkpointing: true
num_frame_per_block: 3
warp_denoising_step: false
````

## File: demo/README.md

````markdown
# StreamDiffusionV2 Demo (Web UI)

This demo provides a simple web interface for live video-to-video inference using the backend in this repository. It supports webcam or screen capture input in the browser.

## Prerequisites
- Python 3.10 (follow the root README for environment setup)
- Node.js 18
- NVIDIA GPU recommended (single or multi-GPU)

## Setup
1) Complete the Python environment and model checkpoint setup as described in the root `README.md` (Installation and Download Checkpoints).
2) Build the frontend and start the backend via the script:
```
# Install
cd demo
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
\. "$HOME/.nvm/nvm.sh"
nvm install 18

cd frontend
npm install
npm run build
cd ../

# Start
chmod +x start.sh
./start.sh
```
The script will:
- Install and build the frontend (`npm install && npm run build` in `demo/frontend`)
- Launch the backend with `torchrun` on port `7860` and host `0.0.0.0`

## Access
- Local: `http://0.0.0.0:7860` or `http://localhost:7860`
- Remote server: `http://<server-ip>:7860` (ensure the port is open)

## Multi-GPU and Denoising Timesteps
- Edit `start.sh`: By default, `start.sh` uses `num_gpus=1`. To enable multi-GPU inference on a single node or adjust the denoising steps, edit the corresponding variables in `start.sh`.  For example:
```
python main.py --port 7860 --host 0.0.0.0 --num_gpus 2 --gpu_ids 0,1 --step 2
```
Our model supports denoising steps from 1 to 4 — feel free to set this value as needed.  
For real-time live-streaming applications, we found that using **2 steps** provides a good balance between speed and quality.


## Troubleshooting
- Camera not available:
  - Allow camera/microphone access for the site in your browser.
  - Error example: `Cannot read properties of undefined (reading 'enumerateDevices')`.
- Frontend not reachable:
  - Ensure the build succeeded (look for `frontend build success`).
  - Check that port 7860 is free, or adjust the port in the script and visit the new port.
  - For remote servers, open the port in firewall/security group.
- Model errors:
  - Verify that all checkpoints were downloaded and placed in the expected directories.

For advanced usage and CLI-based inference, see the root `README.md` (single-GPU and multi-GPU inference scripts).
````

## File: demo/frontend/README.md

````markdown
# create-svelte

Everything you need to build a Svelte project, powered by [`create-svelte`](https://github.com/sveltejs/kit/tree/master/packages/create-svelte).

## Creating a project

If you're seeing this, you've probably already done this step. Congrats!

```bash
# create a new project in the current directory
npm create svelte@latest

# create a new project in my-app
npm create svelte@latest my-app
```

## Developing

Once you've created a project and installed dependencies with `npm install` (or `pnpm install` or `yarn`), start a development server:

```bash
npm run dev

# or start the server and open the app in a new browser tab
npm run dev -- --open
```

## Building

To create a production version of your app:

```bash
npm run build
```

You can preview the production build with `npm run preview`.

> To deploy your app, you may need to install an [adapter](https://kit.svelte.dev/docs/adapters) for your target environment.
````

