# XDIT Project Suspension Notice

## Project Status

It is with regret that I announce the suspension of the XDIT project due to critical technical issues encountered during the sampling process. After numerous attempts and adjustments, I've decided to temporarily shelve the project until conditions become more favorable for progress.

## Integration Challenges with Windows ComfyUI Ecosystem

The primary obstacle preventing the integration of XDIT technology into the Windows ComfyUI ecosystem is the persistent sampler issue with dimensions (1,1,512). This specific tensor shape problem has proven to be insurmountable in the Windows environment despite extensive troubleshooting.

### The Sampler Dimension Problem

Our testing revealed a fundamental incompatibility when processing latent tensors with the shape [1,1,512,x] during the sampling process on Windows. When attempting to distribute these tensors across multiple GPUs, the following critical issues emerged:

- **Tensor Fragmentation Failure**: The distributed computing framework consistently fails to properly fragment these tensors across GPUs, resulting in memory access violations.
- **Context Window Corruption**: The system is unable to maintain proper context window boundaries when processing these tensor dimensions, leading to visual artifacts and generation failures.
- **Memory Synchronization Issues**: After approximately 20-30% completion of the sampling process, memory synchronization between worker processes breaks down, resulting in complete generation failure.

We attempted multiple approaches to address this issue:
1. Modifying the tensor reshaping logic
2. Implementing custom padding strategies
3. Restructuring the sampling pipeline to handle alternative dimensions
4. Creating Windows-specific patch code for the tensor handling

Unfortunately, all these approaches either failed outright or introduced unacceptable performance degradation, making them non-viable solutions.

## Project Architecture Overview

XDIT is a high-performance image generation framework based on distributed GPU computing, designed to fully leverage multi-GPU environments to improve inference efficiency for large diffusion models. The project consists of these core components:

- **Distributed Manager (XDiTDistributedManager)**: Implemented as a singleton pattern, responsible for initializing the distributed environment, creating and managing worker processes, and coordinating the distribution and collection of generation tasks and results.
- **Worker Processes**: Each GPU device corresponds to a worker process responsible for actual model loading and inference computation.
- **Sampler**: Provides advanced APIs supporting various sampling algorithms and parallel strategies.
- **Compatibility Layer**: Implements seamless integration with third-party frameworks such as ComfyUI.

The project supports multiple parallel strategies, including:

- Ulysses Parallel (USP): Enables splitting and parallel processing of diffusion model inference across multiple GPUs
- PipeFusion Parallel: Divides model layers across different GPUs for pipeline execution
- CFG Parallel: Simultaneously calculates conditional and unconditional branches in multi-GPU environments

## Current Progress

The framework's basic architecture has been largely completed, including:

- Core distributed computing architecture design and implementation
- Task coordination and management in multi-GPU environments
- Integration with existing diffusion models (Flux, PixArt, SD3, etc.)
- Example workflows (included in the project)
- Implementation of various optimization strategies (teacache, fbcache, torch compile, etc.)
- Optimization of CUDA communication layer
- Integration interface with ComfyUI

## Core Technical Challenges

### 1. Critical Sampling Process Issues

During implementation, a series of difficult-to-overcome technical barriers emerged in the sampling phase:

- **Sampling Instability**: In multi-GPU environments, especially when Ulysses parallel degree (`ulysses_degree`) exceeds 2, the sampling process frequently becomes unstable, resulting in deteriorating generation quality or complete failure.
- **Cross-GPU Communication Bottleneck**: During large image generation, particularly when processing context overlap (`context_overlap`) regions, inter-GPU communication becomes a severe bottleneck, causing performance degradation rather than improvement.
- **Memory Allocation Issues**: Memory leaks and continuous growth problems were observed during long-step sampling processes, especially when using TeaCache optimization.
- **Gradient Synchronization Problems**: CFG calculation in multi-GPU environments faces consistency issues with gradient synchronization, potentially causing excessive differences in inference results between different batches.

### 2. Windows Platform Compatibility Issues

XDIT has very limited support on Windows platforms, mainly manifested in:

- **NCCL Library Compatibility**: NCCL (NVIDIA Collective Communications Library) has incomplete support on Windows, resulting in inefficient or directly failing multi-GPU communication.
- **Process Management Challenges**: Python's `multiprocessing` module on Windows uses spawn instead of fork mode, causing memory and performance issues when copying large models to child processes.
- **Concurrency Control Issues**: Signal handling and inter-process synchronization mechanisms on Windows differ significantly from Linux, affecting task scheduling stability.
- **CUDA Context Management**: Sharing and isolation of CUDA contexts in multi-process environments present special challenges on Windows.

### 3. Environmental Dependency Issues

- **Ray Version Adaptation**: The project has strict version requirements for the Ray distributed computing framework, needing specific versions (such as 2.0.x) to work properly with XDIT, while these versions often have other compatibility issues in newer environments.
- **Python Version Rollback Requirement**: Latest tests show that the project runs more stably on Python 3.8-3.9, while various unexpected errors occur on Python 3.10+, especially with C++ extensions and CUDA-related components.
- **CUDA Version Sensitivity**: The project shows high sensitivity to CUDA versions, with significant performance differences between CUDA 11.7 and 12.1, requiring precise matching.

## For Developers Interested in Trying

If you have the relevant technical background and are interested in solving these problems, you're welcome to continue development. Here are some suggestions:

### Environment Configuration Recommendations

- Recommended to use Linux systems (Ubuntu 20.04/22.04) for development
- Python version 3.8 or 3.9 recommended
- Ray 2.0.0 confirmed experimentally to have good compatibility with the project
- CUDA 11.7 or 11.8 recommended
- Ensure correct PyTorch version (1.13.x or 2.0.x recommended)

### Key Improvement Directions

- Improve memory management in the sampler, especially the processing logic for context overlap regions
- Optimize multi-process communication mechanisms on Windows platforms, potentially considering alternative solutions
- Implement more robust error handling and recovery mechanisms, especially for long-running stability
- Refactor the implementation of parallel strategies to reduce dependencies and complexity

## Personal Note

I deeply regret that the project cannot move forward at this time. Despite some progress in distributed diffusion model acceleration, adaptation issues on Windows platforms and stability challenges in the sampling process ultimately became insurmountable obstacles.

I hope that developers with the technical capability can take over this project, solve these technical challenges, and advance this promising framework to maturity. If you achieve breakthroughs in your attempts or have any questions, please feel free to contact me.

Good luck with development!
