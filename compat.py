"""
xDiT与ComfyUI的兼容层

提供额外的兼容函数和工具，简化xDiT在ComfyUI环境中的使用
"""

import os
import sys
import torch
import json
from pathlib import Path
from typing import Dict, Any, Union, List, Tuple, Optional

# 导入ComfyUI相关函数（如果需要）
try:
    import folder_paths
    COMFY_PATH = os.path.dirname(folder_paths.__file__)
    IN_COMFYUI = True
except ImportError:
    COMFY_PATH = ""
    IN_COMFYUI = False

# 获取配置
def get_config() -> Dict[str, Any]:
    """加载xDiT节点配置"""
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {
        "xdit_path": "C:/Users/Administrator/Desktop/xdit/xDiT-main/xDiT-main",
        "parallel": {
            "default_world_size": 2,
            "ulysses_degree": 2,
            "pipefusion_parallel_degree": 1,
            "use_cfg_parallel": True,
            "use_parallel_vae": False
        },
        "optimization": {
            "use_torch_compile": False,
            "use_onediff": False,
            "warmup_steps": 1
        }
    }

# 设置xDiT路径
def setup_xdit_path():
    """设置xDiT路径到Python路径"""
    config = get_config()
    xdit_path = config.get("xdit_path", "")
    if xdit_path and os.path.exists(xdit_path):
        if xdit_path not in sys.path:
            sys.path.append(xdit_path)
            return True
    return False

# 设置环境变量
def setup_env_vars():
    """设置xDiT所需的环境变量"""
    # NCCL环境变量，优化多GPU通信
    os.environ["NCCL_P2P_DISABLE"] = "0"  # 启用P2P通信
    os.environ["NCCL_IB_DISABLE"] = "1"   # 禁用InfiniBand
    os.environ["NCCL_SOCKET_IFNAME"] = "lo"  # 使用本地网络接口
    # PyTorch环境变量
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # 减少内存碎片
    return True

# ComfyUI相关工具函数
def find_model_in_folder_paths(model_name: str) -> str:
    """
    在ComfyUI的模型路径中查找模型
    
    Args:
        model_name: 模型名称
        
    Returns:
        模型完整路径或原始名称（如果找不到）
    """
    if not IN_COMFYUI or not model_name:
        return model_name
        
    # 检查是否已经是完整路径
    if os.path.exists(model_name):
        return model_name
        
    # 尝试在ComfyUI的模型目录中查找
    try:
        # 检查checkpoints目录
        checkpoints_dir = folder_paths.get_folder_paths("checkpoints")
        for dir_path in checkpoints_dir:
            full_path = os.path.join(dir_path, model_name)
            if os.path.exists(full_path):
                return full_path
                
        # 检查diffusers目录
        diffusers_dir = folder_paths.get_folder_paths("diffusers")
        for dir_path in diffusers_dir:
            full_path = os.path.join(dir_path, model_name)
            if os.path.exists(full_path):
                return full_path
    except:
        pass
        
    # 如果找不到，返回原始名称
    return model_name

# 转换ComfyUI节点参数到xDiT参数
def convert_comfy_params_to_xdit(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    将ComfyUI节点参数转换为xDiT参数
    
    Args:
        params: ComfyUI参数字典
        
    Returns:
        转换后的xDiT参数字典
    """
    xdit_params = {}
    
    # 常见参数映射
    param_mapping = {
        "prompt": "prompt",
        "negative_prompt": "negative_prompt",
        "steps": "num_inference_steps",
        "cfg_scale": "guidance_scale",
        "width": "width",
        "height": "height",
        "seed": "seed",
    }
    
    for comfy_key, xdit_key in param_mapping.items():
        if comfy_key in params:
            xdit_params[xdit_key] = params[comfy_key]
    
    return xdit_params

# 初始化
setup_xdit_path()
setup_env_vars()

# 导出的函数
__all__ = [
    "get_config",
    "setup_xdit_path",
    "setup_env_vars",
    "find_model_in_folder_paths",
    "convert_comfy_params_to_xdit",
] 