import os
import torch
import folder_paths
from ..xdit_manager import manager

class XDiTUNETLoader:
    """
    xDiT Model Loader Node
    
    For loading xDiT models with multi-GPU parallel support
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {
                    "default": "black-forest-labs/FLUX.1-dev"
                }),
                "model_type": (["flux", "pixart", "pixart-sigma", "sd3", "hunyuan"], {
                    "default": "flux"
                }),
                "weight_precision": (["default", "fp16", "bf16"], {
                    "default": "fp16"
                }),
                "device_count": ("INT", {
                    "default": 2, 
                    "min": 1, 
                    "max": 8
                }),
                "usp_degree": ("INT", {
                    "default": 2, 
                    "min": 1, 
                    "max": 4
                }),
                "pipeline_degree": ("INT", {
                    "default": 1, 
                    "min": 1, 
                    "max": 4
                }),
            },
            "optional": {
                "enable_cfg_parallel": ("BOOLEAN", {
                    "default": True
                }),
                "enable_parallel_vae": ("BOOLEAN", {
                    "default": False
                }),
                "warmup_steps": ("INT", {
                    "default": 1, 
                    "min": 0, 
                    "max": 10
                }),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "xDiT"
    
    def load_model(self, model_path, model_type, weight_precision, device_count, 
                  usp_degree=2, pipeline_degree=1, 
                  enable_cfg_parallel=True, enable_parallel_vae=False, warmup_steps=1):
        """
        Load the xDiT model
        
        Args:
            model_path: Model ID or local path
            model_type: Type of model
            weight_precision: Weight data type
            device_count: Number of parallel processes
            usp_degree: Sequence parallelism degree
            pipeline_degree: Pipeline parallelism degree
            enable_cfg_parallel: Whether to use CFG parallelism
            enable_parallel_vae: Whether to parallelize VAE
            warmup_steps: Number of warmup steps
            
        Returns:
            Model configuration dictionary
        """
        print(f"Loading xDiT model: {model_path}, type: {model_type}, parallel devices: {device_count}")
        
        # Determine model path
        resolved_model_path = model_path
        
        # Check if it's a local path
        if os.path.exists(model_path):
            resolved_model_path = model_path
        else:
            # Try to find in ComfyUI model paths
            # Determine appropriate directory based on model type
            if model_type in ["flux", "pixart", "pixart-sigma", "sd3"]:
                # These model types are typically stored in unet or diffusion_models directory
                possible_paths = [
                    folder_paths.get_full_path("unet", model_path),
                    folder_paths.get_full_path("diffusion_models", model_path),
                    folder_paths.get_full_path("checkpoints", model_path)
                ]
                
                for path in possible_paths:
                    if path is not None and os.path.exists(path):
                        resolved_model_path = path
                        break
        
        # Initialize the distributed environment
        try:
            # Convert weight type
            dtype_map = {
                "default": None,
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
            }
            torch_dtype = dtype_map[weight_precision]
            
            # Verify parallel configuration
            # Ensure world_size = usp_degree * pipeline_degree * (2 if enable_cfg_parallel else 1)
            actual_world_size = usp_degree * pipeline_degree
            if enable_cfg_parallel:
                actual_world_size *= 2
                
            if actual_world_size != device_count:
                print(f"Warning: Parallel configuration may be inconsistent. Recommended device_count is {actual_world_size}, current setting is {device_count}")
            
            # Pre-initialize the model configuration
            # Actual model loading will be deferred to the sampling step for better memory management
            model_config = {
                "model_path": resolved_model_path,
                "model_type": model_type,
                "world_size": device_count,
                "torch_dtype": torch_dtype,
                "ulysses_degree": usp_degree,
                "pipefusion_degree": pipeline_degree,
                "use_cfg_parallel": enable_cfg_parallel,
                "use_parallel_vae": enable_parallel_vae,
                "warmup_steps": warmup_steps
            }
            
            # Attempt early initialization to detect errors
            # This provides early feedback without full model loading
            # manager.validate_config(model_config)
            
            return (model_config,)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"xDiT model loading failed: {str(e)}")
            raise Exception(f"Failed to initialize xDiT model: {str(e)}")



class XfuserPipelineLoader:
    """
    Xfuser Pipeline Loader Node
    
    Simplified version of the Xfuser model loader, compatible with the official version
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("checkpoints"), {
                    "default": "ponyRealism_V22MainVAE.safetensors"
                }),
                "width": ("INT", {
                    "default": 1024, 
                    "min": 512, 
                    "max": 2048
                }),
                "height": ("INT", {
                    "default": 1024, 
                    "min": 512, 
                    "max": 2048
                }),
                "device_count": ("INT", {
                    "default": 1, 
                    "min": 1, 
                    "max": 8
                }),
            },
            "optional": {
                "model_type": (["auto", "flux", "pixart", "pixart-sigma", "sd3", "hunyuan", "sdxl"], {
                    "default": "auto"
                }),
                "vae": (folder_paths.get_filename_list("vae") + ["None", "auto"], {
                    "default": "auto"
                }),
                "vae_precision": (["fp16", "fp32"], {
                    "default": "fp16"
                }),
                "parallel_mode": (["auto", "usp", "pipefusion", "hybrid"], {
                    "default": "auto"
                }),
                "enable_torch_compile": ("BOOLEAN", {
                    "default": False
                }),
                "enable_teacache": ("BOOLEAN", {
                    "default": False
                }),
                "enable_parallel_vae": ("BOOLEAN", {
                    "default": False
                }),
            }
        }
    
    RETURN_TYPES = ("XFUSER_PIPELINE", "VAE")
    RETURN_NAMES = ("pipeline", "vae")
    FUNCTION = "load_pipeline"
    CATEGORY = "xDiT"
    
    def load_pipeline(self, model_name, width, height, device_count, model_type="auto", 
                     vae="auto", vae_precision="fp16", parallel_mode="auto", 
                     enable_torch_compile=False, enable_teacache=False, enable_parallel_vae=False):
        """
        Load Xfuser Pipeline
        
        Args:
            model_name: Model filename
            width: Image width
            height: Image height
            device_count: Number of GPUs
            model_type: Model type
            parallel_mode: Parallelization mode
            enable_torch_compile: Whether to use torch.compile
            enable_teacache: Whether to use TeaCache acceleration
            enable_parallel_vae: Whether to parallelize VAE
            
        Returns:
            Pipeline configuration
        """
        print(f"Loading Xfuser Pipeline - Model: {model_name}, Size: {width}x{height}, GPUs: {device_count}")
        
        # Get model path
        model_path = folder_paths.get_full_path("checkpoints", model_name)
        if not model_path:
            error_msg = f"Error: Cannot find model {model_name}"
            print(error_msg)
            raise Exception(error_msg)
            
        # Configure parallelization parameters
        # Automatically determine optimal configuration based on device count and parallel mode
        if parallel_mode == "auto":
            if device_count >= 4:
                # Use hybrid mode for many GPUs
                parallel_mode = "hybrid"
                usp_degree = min(2, device_count // 2)
                pipefusion_degree = min(2, device_count // usp_degree // 2) 
                use_cfg_parallel = True
            elif device_count >= 2:
                # Use USP for fewer GPUs
                parallel_mode = "usp"
                usp_degree = min(2, device_count)
                pipefusion_degree = 1
                use_cfg_parallel = device_count >= 2
            else:
                # Single GPU, no parallelism
                usp_degree = 1
                pipefusion_degree = 1
                use_cfg_parallel = False
        elif parallel_mode == "usp":
            usp_degree = min(device_count, 4)
            pipefusion_degree = 1
            use_cfg_parallel = device_count > usp_degree
        elif parallel_mode == "pipefusion":
            usp_degree = 1
            pipefusion_degree = min(device_count, 4)
            use_cfg_parallel = device_count > pipefusion_degree
        elif parallel_mode == "hybrid":
            # Balanced allocation
            usp_degree = min(2, max(1, device_count // 2))
            pipefusion_degree = min(2, max(1, device_count // usp_degree // 2))
            use_cfg_parallel = device_count > (usp_degree * pipefusion_degree)
            
        # Try to detect model type (if auto)
        if model_type == "auto":
            model_name_lower = model_name.lower()
            # Guess based on model name
            if "flux" in model_name_lower:
                model_type = "flux"
            elif "pixart-sigma" in model_name_lower or "pixartsigma" in model_name_lower:
                model_type = "pixart-sigma"
            elif "pixart" in model_name_lower:
                model_type = "pixart"
            elif "sd3" in model_name_lower or "stable-diffusion-3" in model_name_lower:
                model_type = "sd3"
            elif "hunyuan" in model_name_lower:
                model_type = "hunyuan"
            elif any(x in model_name_lower for x in ["sdxl", "sd-xl", "sd_xl", "xl-base", "ponyrealism", "dreamshaper_xl", "juggernaut_xl"]):
                model_type = "sdxl"
                print(f"SDXL model detected: {model_name}")
            else:
                # Try to determine from file size or other heuristics
                try:
                    import os
                    file_size_gb = os.path.getsize(model_path) / (1024*1024*1024)
                    if file_size_gb > 5:
                        # Large models are likely SDXL
                        model_type = "sdxl"
                        print(f"SDXL model detected based on file size ({file_size_gb:.1f} GB): {model_name}")
                    else:
                        # Default to flux type for smaller models
                        model_type = "flux"
                except Exception:
                    # If error in size detection, default to flux
                    model_type = "flux"
                
            print(f"Automatically detected model type: {model_type}")
        
        # 确定VAE路径并加载VAE模型
        vae_path = None
        vae_model = None
        if vae != "None" and (vae != "auto" or model_type in ["sdxl", "sd3"]):  # 仅对特定模型类型或显式要求使用VAE
            if vae == "auto":
                # 尝试根据模型文件名匹配VAE
                for vae_option in folder_paths.get_filename_list("vae"):
                    if model_name.lower().replace(".safetensors", "").replace(".ckpt", "") in vae_option.lower():
                        vae_path = folder_paths.get_full_path("vae", vae_option)
                        print(f"自动选择匹配的VAE: {vae_option}")
                        break
                
                # 如果没找到匹配的VAE，对SDXL模型使用默认VAE
                if not vae_path and model_type == "sdxl":
                    for default_vae in ["sdxl_vae.safetensors", "madebyollin-sdxl-vae-fix.safetensors", "sdxl_vae.pt"]:
                        default_path = folder_paths.get_full_path("vae", default_vae)
                        if default_path:
                            vae_path = default_path
                            print(f"使用默认SDXL VAE: {default_vae}")
                            break
            else:  # 用户选择了特定VAE
                vae_path = folder_paths.get_full_path("vae", vae)
                if vae_path:
                    print(f"使用用户选择的VAE: {vae}")
        
        # 加载VAE模型(如果有路径)
        if vae_path:
            try:
                import comfy.sd
                from comfy.utils import load_torch_file
                
                print(f"正在加载VAE: {vae_path}")
                vae_sd = load_torch_file(vae_path)
                # 使用ComfyUI的VAE加载器
                vae_model = comfy.sd.VAE(sd=vae_sd)
                print(f"VAE加载成功，将作为输出提供")
            except Exception as e:
                print(f"VAE加载失败: {str(e)}")
        
        # Return configuration dictionary with VAE information and the VAE model
        return ({
            "model_path": model_path,
            "width": width,
            "height": height,
            "world_size": device_count,
            "model_type": model_type,
            "use_torch_compile": enable_torch_compile,
            "use_teacache": enable_teacache,
            "use_dit_parallel_vae": enable_parallel_vae,
            "usp_degree": usp_degree,
            "pipefusion_degree": pipefusion_degree,
            "use_cfg_parallel": use_cfg_parallel,
            "parallel_mode": parallel_mode,
            "scheduler": "euler_a",
            "vae_path": vae_path,
            "vae_precision": vae_precision,
        }, vae_model)