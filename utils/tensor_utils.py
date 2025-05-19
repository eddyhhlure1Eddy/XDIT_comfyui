import torch
import io
import base64
import numpy as np
from PIL import Image

def tensor_to_base64(tensor, format="PNG"):
    """
    将张量转换为base64编码的图像
    
    Args:
        tensor: 形状为 [C, H, W] 的张量
        format: 图像格式，如 "PNG", "JPEG"
        
    Returns:
        base64编码的字符串
    """
    if tensor.dim() == 4 and tensor.size(0) == 1:
        # 移除批次维度
        tensor = tensor.squeeze(0)
    
    # 确保张量在CPU上
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # 将值范围从 [-1, 1] 调整到 [0, 1]
    if tensor.min() < 0:
        tensor = (tensor + 1) / 2
    
    # 将值范围从 [0, 1] 调整到 [0, 255]
    if tensor.max() <= 1:
        tensor = tensor * 255
    
    # 转换为 uint8
    tensor = tensor.clamp(0, 255).to(torch.uint8)
    
    # 如果通道是RGB顺序 [C, H, W]，转换为 [H, W, C]
    if tensor.size(0) == 3:
        tensor = tensor.permute(1, 2, 0)
    
    # 转换为NumPy数组
    array = tensor.numpy()
    
    # 转换为PIL图像
    image = Image.fromarray(array)
    
    # 转换为Base64
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return img_str

def base64_to_tensor(base64_str, device="cpu"):
    """
    将base64编码的图像转换为张量
    
    Args:
        base64_str: base64编码的图像字符串
        device: 设备
        
    Returns:
        torch.Tensor: 图像张量
    """
    # 解码base64
    img_data = base64.b64decode(base64_str)
    
    # 转换为PIL图像
    image = Image.open(io.BytesIO(img_data))
    
    # 转换为numpy数组
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # 转换为torch张量，调整通道顺序 (H, W, C) -> (C, H, W)
    img_array = img_array.transpose(2, 0, 1)
    img_tensor = torch.from_numpy(img_array).to(device)
    
    # 添加批次维度
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor

def latent_to_pil(latent_tensor, vae=None):
    """
    将潜在表示转换为PIL图像
    
    Args:
        latent_tensor: 潜在表示张量
        vae: VAE解码器
        
    Returns:
        PIL图像
    """
    if vae is None:
        raise ValueError("需要提供VAE解码器")
    
    # 确保latent_tensor在正确的设备上
    if vae.device != latent_tensor.device:
        latent_tensor = latent_tensor.to(vae.device)
    
    # 使用VAE解码
    with torch.no_grad():
        image = vae.decode(latent_tensor)
    
    # 处理图像
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    
    # 转换为PIL
    images = []
    for i in range(image.shape[0]):
        img = Image.fromarray((image[i] * 255).astype(np.uint8))
        images.append(img)
    
    return images[0] if len(images) == 1 else images

def comfyui_latent_format_to_tensor(latent_format):
    """
    将ComfyUI的latent格式转换为标准的torch张量
    
    Args:
        latent_format: ComfyUI的latent格式 {"samples": tensor}
        
    Returns:
        torch.Tensor: 标准的torch张量
    """
    if isinstance(latent_format, dict) and "samples" in latent_format:
        return latent_format["samples"]
    else:
        # 如果输入已经是张量，直接返回
        if isinstance(latent_format, torch.Tensor):
            return latent_format
        raise ValueError("无效的latent格式")
        
def tensor_to_comfyui_latent_format(tensor):
    """
    将标准的torch张量转换为ComfyUI的latent格式
    
    Args:
        tensor: 标准的torch张量
        
    Returns:
        dict: ComfyUI的latent格式 {"samples": tensor}
    """
    if isinstance(tensor, torch.Tensor):
        return {"samples": tensor}
    elif isinstance(tensor, dict) and "samples" in tensor:
        # 如果输入已经是ComfyUI格式，直接返回
        return tensor
    else:
        raise ValueError("无效的张量") 