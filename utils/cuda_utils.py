import torch
import gc
import os

def reset_cuda_context():
    """
    重置CUDA上下文，避免与父进程冲突
    通常在子进程开始时调用
    """
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, '_lazy_init'):
        torch.cuda._lazy_init = True
    if hasattr(torch.cuda, '_initialized'):
        torch.cuda._initialized = False

def optimize_cuda_cache():
    """
    优化CUDA缓存，减少内存碎片
    通常在大型操作之前调用
    """
    gc.collect()
    torch.cuda.empty_cache()

def set_cuda_visible_devices(device_indices):
    """
    设置可见的CUDA设备
    
    Args:
        device_indices: 设备索引列表，如[0,1]
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_indices))

def get_cuda_memory_info(device=None):
    """
    获取CUDA内存使用情况
    
    Args:
        device: 设备索引或设备对象
        
    Returns:
        (已用内存, 总内存) 单位为字节
    """
    if device is None:
        device = torch.cuda.current_device()
    
    if isinstance(device, int):
        device = torch.device(f"cuda:{device}")
    
    # 获取内存信息
    allocated = torch.cuda.memory_allocated(device)
    max_allocated = torch.cuda.max_memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    max_reserved = torch.cuda.max_memory_reserved(device)
    
    return {
        "allocated": allocated,
        "max_allocated": max_allocated,
        "reserved": reserved,
        "max_reserved": max_reserved,
        "allocated_gb": allocated / (1024**3),
        "max_allocated_gb": max_allocated / (1024**3),
        "reserved_gb": reserved / (1024**3),
        "max_reserved_gb": max_reserved / (1024**3),
    }

def print_cuda_memory_summary():
    """打印所有CUDA设备的内存摘要"""
    print(torch.cuda.memory_summary())

def get_available_cuda_devices():
    """获取可用的CUDA设备数量"""
    return list(range(torch.cuda.device_count()))

def set_cuda_optimization_flags():
    """设置CUDA优化标志"""
    # 启用cudnn自动调优
    torch.backends.cudnn.benchmark = True
    # 如果输入大小一致，这可以提高性能
    torch.backends.cudnn.deterministic = False 