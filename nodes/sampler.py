import os
import torch
import time
import numpy as np
from PIL import Image
import io
import base64
import random

from ..xdit_manager import manager
from ..utils.tensor_utils import comfyui_latent_format_to_tensor, tensor_to_comfyui_latent_format, base64_to_tensor

class XfuserSampler:
    """
    Xfuser Sampler Node
    
    Generate images using the Xfuser Pipeline
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("XFUSER_PIPELINE", {}),
                "positive": ("XFUSER_POSITIVE", {}),
                "negative": ("XFUSER_NEGATIVE", {}),
                "seed": ("INT", {
                    "default": 14461861, 
                    "min": 0, 
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True
                }),
                "steps": ("INT", {
                    "default": 30, 
                    "min": 1, 
                    "max": 100
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 7.0, 
                    "min": 0.0, 
                    "max": 20.0, 
                    "step": 0.1
                }),
            },
            "optional": {
                "vae": ("VAE", {}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "sample"
    CATEGORY = "xDiT"
    
    def sample(self, pipeline, positive, negative, seed, steps=30, cfg_scale=7.0, vae=None):
        """
        Execute the sampling process with optional external VAE for decoding
        
        Args:
            pipeline: Pipeline configuration from XfuserPipelineLoader
            positive: Positive prompt
            negative: Negative prompt
            seed: Random seed
            steps: Number of sampling steps
            cfg_scale: CFG scale factor
            
        Returns:
            Generated image
        """
        print("\n=== XDIT SAMPLER DIAGNOSTIC LOG ===\n")
        print(f"[SAMPLER] Starting sampling process - seed: {seed}, steps: {steps}, cfg: {cfg_scale}")
        
        try:
            # Get pipeline configuration
            print(f"[SAMPLER] Processing pipeline object type: {type(pipeline)}")
            model_path = pipeline.get("model_path")
            print(f"[SAMPLER] Model path: {model_path}")
            model_type = pipeline.get("model_type", "auto")
            print(f"[SAMPLER] Model type: {model_type}")
            width = pipeline.get("width", 1024)
            height = pipeline.get("height", 1024)
            print(f"[SAMPLER] Output dimensions: {width}x{height}")
            world_size = pipeline.get("world_size", 1)
            print(f"[SAMPLER] World size: {world_size}")
            
            # Get optional parameters if available
            ulysses_degree = pipeline.get("ulysses_degree", 1)
            pipefusion_degree = pipeline.get("pipefusion_degree", 1)
            use_cfg_parallel = pipeline.get("use_cfg_parallel", world_size > 1)
            use_parallel_vae = pipeline.get("use_parallel_vae", False)
            warmup_steps = pipeline.get("warmup_steps", 1)
            scheduler = pipeline.get("scheduler", "euler_a")
            vae_name = pipeline.get("vae_name", "")
            start_at_step = pipeline.get("start_at_step", 0)
            
            print(f"[SAMPLER] Parallel config: USP={ulysses_degree}, PipeFusion={pipefusion_degree}, CFG Parallel={use_cfg_parallel}")
            print(f"[SAMPLER] Scheduler: {scheduler}, VAE: {vae_name or 'default'}")
            
            # Other optional parameters with defaults
            target_width = pipeline.get("target_width", 0)
            target_height = pipeline.get("target_height", 0)
            crop_w = pipeline.get("crop_w", 0)
            crop_h = pipeline.get("crop_h", 0)
            context_overlap = pipeline.get("context_overlap", 0)
            context_stride = pipeline.get("context_stride", 0)
            use_torch_compile = pipeline.get("use_torch_compile", False)
            use_teacache = pipeline.get("use_teacache", False)
            use_dit_fast_attn = pipeline.get("use_dit_fast_attn", False)
            
            print(f"[SAMPLER] Optimization flags: Compile={use_torch_compile}, TeaCache={use_teacache}, FastAttn={use_dit_fast_attn}")
        except Exception as e:
            print(f"[SAMPLER ERROR] Pipeline parameter extraction failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
            
        # Get prompts - handle both string type and object with get() method
        try:
            print(f"[SAMPLER] Processing positive prompt object type: {type(positive)}")
            if isinstance(positive, str):
                print(f"[SAMPLER] Positive prompt is string type")
                positive_prompt = positive
            else:
                print(f"[SAMPLER] Positive prompt is object type, converting")
                positive_prompt = str(positive)
            
            print(f"[SAMPLER] Processing negative prompt object type: {type(negative)}")
            if isinstance(negative, str):
                print(f"[SAMPLER] Negative prompt is string type")
                negative_prompt = negative
            else:
                print(f"[SAMPLER] Negative prompt is object type, converting")
                negative_prompt = str(negative)
            
            # Check if external VAE provided
            if vae is not None:
                print(f"[SAMPLER] External VAE provided, will use for decoding")
                use_external_vae = True
            else:
                print(f"[SAMPLER] No external VAE provided, will use pipeline's internal VAE")
                use_external_vae = False
            
            print(f"[SAMPLER] Final prompts - Positive: '{positive_prompt}', Negative: '{negative_prompt}'")
        except Exception as e:
            print(f"[SAMPLER ERROR] Prompt processing failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
        print(f"Using Xfuser Sampler - Model: {model_path}, Steps: {steps}, CFG: {cfg_scale}")
        print(f"Prompt: {positive_prompt}")
        if negative_prompt:
            print(f"Negative prompt: {negative_prompt}")
        
        try:
            # Initialize distributed environment
            manager.initialize(model_path, world_size)
            
            # Prepare task data
            task = {
                "action": "generate",
                "prompt": positive_prompt,
                "negative_prompt": negative_prompt,
                "steps": steps,
                "cfg_scale": cfg_scale,
                "height": height,
                "width": width,
                "seed": seed,
                "model_type": model_type,
                # Use default parameters for other options
                "scheduler": "euler_a",
                "ulysses_degree": min(4, world_size // 2) if world_size > 1 else 1,
                "pipefusion_degree": 1,
                "use_cfg_parallel": world_size >= 2,
                "context_overlap": 4,
                "context_stride": 2,
                "return_latent": use_external_vae,
                "output_type": "latent" if use_external_vae else "pil",
            }
            
            print(f"Starting generation, height: {height}, width: {width}, steps: {steps}")
            start_time = time.time()
            
            # Execute generation
            result = manager.generate(task)
            
            end_time = time.time()
            print(f"Generation completed, time taken: {end_time - start_time:.2f} seconds")
            
            # Process results
            if result["status"] == "success":
                print(f"Generation successful, model time: {result.get('elapsed_time', 0):.2f} seconds")
                
                if not use_external_vae or "latent" not in result:
                    # No external VAE or no latent available - use base64 image
                    print(f"[SAMPLER] Using pipeline's internal VAE decoded image")
                    if "image" in result:
                        # Convert base64 image to PIL image
                        img_data = base64.b64decode(result["image"])
                        image = Image.open(io.BytesIO(img_data))
                        print(f"Image dimensions: {image.size}")
                        
                        # Convert to ComfyUI format image tensor
                        image_np = np.array(image).astype(np.float32) / 255.0
                        image_tensor = torch.from_numpy(image_np).unsqueeze(0)
                        
                        return (image_tensor,)
                    else:
                        print("Failed to get image data")
                        # Return a blank image on failure
                        raise Exception("No image data returned from generation")
                else:
                    # We have external VAE and latent data - decode with external VAE
                    print(f"[SAMPLER] Decoding latent with external VAE")
                    try:
                        latent_data = result["latent"]
                        # Convert to proper format for VAE decoding
                        latent_tensor = torch.tensor(latent_data).to(device=comfy.model_management.get_torch_device())
                        # Create proper latent format for ComfyUI VAE
                        samples = {"samples": latent_tensor}
                        # Decode using external VAE
                        decoded = vae.decode(samples)["images"]
                        print(f"[SAMPLER] External VAE decoding successful")
                        return (decoded,)
                    except Exception as e:
                        print(f"[SAMPLER] Error during external VAE decoding: {e}, falling back to internal VAE")
                        # Fallback to base64 image
                        if "image" in result:
                            # Convert base64 image to PIL image
                            img_data = base64.b64decode(result["image"])
                            image = Image.open(io.BytesIO(img_data))
                            print(f"Image dimensions: {image.size}")
                            
                            # Convert to ComfyUI format image tensor
                            image_np = np.array(image).astype(np.float32) / 255.0
                            image_tensor = torch.from_numpy(image_np).unsqueeze(0)
                            
                            return (image_tensor,)
                        else:
                            print("Failed to get image data")
                            # Return a blank image on failure
                            raise Exception("No image data returned from generation")
            else:
                # Handle errors
                error_message = result.get('message', 'Unknown error')
                print(f"Generation failed: {error_message}")
                raise Exception(f"Image generation failed: {error_message}")
        
        except Exception as e:
            print(f"Error during sampling process: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return a blank image when there's an error
            return (torch.zeros((1, 3, height, width)),)
    
    def _update_seed(self, seed, control):
        """Handle seed updates"""
        if control == "randomize":
            return random.randint(0, 0xffffffffffffffff)
        else:  # fixed
            return seed

class XDiTSamplerCustomAdvanced:
    """
    xDiT Advanced Sampler Node
    
    Uses the xDiT engine to execute sampling in parallel on multiple GPUs
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {}),
                "latent": ("LATENT", {}),
                "prompt": ("STRING", {
                    "default": "A beautiful landscape"
                }),
                "negative_prompt": ("STRING", {
                    "default": "", 
                    "multiline": True
                }),
                "steps": ("INT", {
                    "default": 20, 
                    "min": 1, 
                    "max": 100
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 7.5, 
                    "min": 0, 
                    "max": 20
                }),
                "seed": ("INT", {
                    "default": 42, 
                    "min": 0, 
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True
                }),
                "scheduler": (["euler_a", "ddim_uniform", "dpmpp_2m", "dpmpp_3m_sde", "uni_pc"], {
                    "default": "euler_a"
                }),
            },
            "optional": {
                "context_overlap": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 8, 
                    "step": 1
                }),
                "context_stride": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 8, 
                    "step": 1
                }),
                "start_step": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 999
                }),
                "vae_model": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "width_override": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 8192
                }),
                "height_override": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 8192
                }),
                "target_width": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 8192
                }),
                "target_height": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 8192
                }),
                "crop_width": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 8192
                }),
                "crop_height": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 8192
                }),
                "enable_attention_acceleration": ("BOOLEAN", {
                    "default": False
                }),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "xDiT"
    
    def sample(self, model, latent, prompt, negative_prompt="", steps=20, cfg_scale=7.5, 
              seed=42, scheduler="euler_a", context_overlap=0, context_stride=0, start_step=0,
              vae_model=None, width_override=0, height_override=0, target_width=0, target_height=0, 
              crop_width=0, crop_height=0, enable_attention_acceleration=False):
        """
        Execute the sampling process
        
        Args:
            model: Model configuration from XDiTUNETLoader
            latent: Latent image
            prompt: Text prompt
            negative_prompt: Negative text prompt
            steps: Sampling steps
            cfg_scale: CFG scale strength
            seed: Random seed
            scheduler: Scheduler type
            context_overlap: Context overlap parameter
            context_stride: Context stride parameter
            start_step: Start step
            vae_model: VAE model name
            width_override, height_override: Input dimensions
            target_width, target_height: Target dimensions
            crop_width, crop_height: Cropping dimensions
            enable_attention_acceleration: Whether to use DiTFastAttn acceleration
            
        Returns:
            Sampling result
        """
        # Get model configuration parameters
        model_path = model.get("model_path")
        model_type = model.get("model_type", "flux")
        world_size = model.get("world_size", 2)
        torch_dtype = model.get("torch_dtype", None)
        ulysses_degree = model.get("ulysses_degree", 1)
        pipefusion_degree = model.get("pipefusion_degree", 1)
        use_cfg_parallel = model.get("use_cfg_parallel", world_size > 1)  
        use_parallel_vae = model.get("use_parallel_vae", False)
        warmup_steps = model.get("warmup_steps", 1)
        
        print(f"使用xDiT采样器 - 模型: {model_path}, 设备数量: {world_size}")
        print(f"提示词: {prompt}")
        
        try:
            # Initialize distributed environment
            print(f"[SAMPLER] Initializing distributed environment with model={model_path}, world_size={world_size}")
            manager_init_start = time.time()
            try:
                manager.initialize(model_path, world_size)
                print(f"[SAMPLER] Manager initialization completed in {time.time() - manager_init_start:.2f}s")
            except Exception as e:
                print(f"[SAMPLER ERROR] Manager initialization failed: {str(e)}")
                import traceback
                traceback.print_exc()
                raise
            
            # Auto-calculate context parameters if not specified
            if context_overlap == 0 or context_stride == 0:
                # Calculate based on image size and model type
                print(f"[SAMPLER] Calculating optimal context parameters for {width}x{height} {model_type} image")
                context_overlap, context_stride = self._calculate_context_params(
                    width, height, model_type
                )
                print(f"[SAMPLER] Auto-selected context parameters: overlap={context_overlap}, stride={context_stride}")
            
            # Prepare task data
            print(f"[SAMPLER] Preparing task data dictionary")
            task = {
                "action": "generate",
                "prompt": prompt,
                "negative_prompt": neg_prompt,
                "steps": steps,
                "cfg_scale": cfg_scale,
                "width": width,
                "height": height,
                "seed": seed,
                "scheduler": scheduler,
                "context_overlap": context_overlap,
                "context_stride": context_stride,
                "start_at_step": start_at_step,
                "model_type": model_type,
                "ulysses_degree": ulysses_degree,
                "pipefusion_degree": pipefusion_degree,
                "use_cfg_parallel": use_cfg_parallel,
                "use_parallel_vae": use_parallel_vae,
                "use_torch_compile": use_torch_compile,
                "use_teacache": use_teacache,
                "warmup_steps": warmup_steps,
                "target_width": target_width if target_width > 0 else width // 2,
                "target_height": target_height if target_height > 0 else height // 2,
                "crop_w": crop_w,
                "crop_h": crop_h,
                "vae_name": vae_name,
                "use_dit_fast_attn": use_dit_fast_attn,
                "return_latent": True,
                "output_type": "latent",
            }
            
            # Output configuration information
            print(f"[SAMPLER] Starting generation: {width}x{height}, steps={steps}, context=[{context_overlap},{context_stride}], seed={seed}")
            print(f"[SAMPLER] Using parallel config: USP={ulysses_degree}, PipeFusion={pipefusion_degree}, CFG={use_cfg_parallel}")
            if use_dit_fast_attn:
                print("[SAMPLER] Enabled DiTFastAttn acceleration")
            if use_torch_compile:
                print("[SAMPLER] Enabled Torch compile acceleration")
            if use_teacache:
                print("[SAMPLER] Enabled TeaCache acceleration")
            
            start_time = time.time()
            print(f"[SAMPLER] Calling manager.generate() at {time.strftime('%H:%M:%S')}")
            
            # Execute generation with error retry mechanism
            try:
                print("[SAMPLER] First attempt with full settings")
                result = manager.generate(task)
                print(f"[SAMPLER] Generation completed successfully in first attempt")
            except Exception as e:
                # First attempt failed, may be due to advanced parameters, try fallback to basic mode
                print(f"[SAMPLER ERROR] First generation attempt failed: {str(e)}")
                print(f"[SAMPLER] Attempting fallback to basic mode...")
                # Disable all advanced acceleration options
                task["use_torch_compile"] = False
                task["use_teacache"] = False
                task["use_dit_fast_attn"] = False
                # If multi-GPU, try simplest parallel config
                if world_size > 1:
                    task["ulysses_degree"] = 1
                    task["pipefusion_degree"] = 1
                    task["use_cfg_parallel"] = world_size >= 2
                    print("[SAMPLER] Falling back to basic parallel mode: USP=1, PipeFusion=1")
                    
                # Try again with basic settings
                print(f"[SAMPLER] Second attempt with basic settings at {time.strftime('%H:%M:%S')}")
                try:
                    result = manager.generate(task)
                    print(f"[SAMPLER] Generation completed successfully with basic settings")
                except Exception as e2:
                    print(f"[SAMPLER ERROR] Second attempt also failed: {str(e2)}")
                    import traceback
                    traceback.print_exc()
                    raise
            
            end_time = time.time()
            print(f"Generation completed, time taken: {end_time - start_time:.2f} seconds")
            
            if result["status"] == "success":
                # Process the resulting image
                print(f"[SAMPLER] Generation successful, model processing time: {result.get('elapsed_time', 0):.2f} seconds")
                print(f"Generation successful, model processing time: {result.get('elapsed_time', 0):.2f} seconds")
                
                # Convert base64 image to PIL image
                if "image" in result:
                    # Decode base64 image to PIL image
                    img_data = base64.b64decode(result["image"])
                    image = Image.open(io.BytesIO(img_data))
                    print(f"Image size: {image.size}")
                    
                    # Convert to ComfyUI format
                    # Return as IMAGE type directly
                    image_np = np.array(image).astype(np.float32) / 255.0
                    image_tensor = torch.from_numpy(image_np)[None,]
                    
                    # Return the image directly
                    return (image_tensor,)
                else:
                    print("Failed to get image data from result")
                    # Return empty image as fallback
                    empty_image = torch.zeros(1, 3, height, width)
                    return (empty_image,)
            else:
                # Handle errors
                error_message = result.get('message', 'Unknown error')
                print(f"Generation failed: {error_message}")
                
                # Return empty image as fallback
                empty_image = torch.zeros(1, 3, height, width)
                empty_image = empty_image.to(torch.float32)  # Ensure float32 format
                
                # Raise exception with the error message for better diagnostics
                raise Exception(f"Image generation failed: {error_message}")
        
        except Exception as e:
            error_message = str(e)
            print(f"Error during sampling process: {error_message}")
            import traceback
            traceback.print_exc()
            
            # Generate a proper empty image that ComfyUI can handle
            empty_image = torch.zeros(1, 3, 512, 512)
            # Ensure it's in the right format and range (0-1 float32)
            empty_image = empty_image.to(torch.float32)
            
            if "timed out" in error_message.lower():
                print("The xDiT worker process timed out. Possible solutions:")
                print("1. Try a smaller image size")
                print("2. Check if xfuser package is installed: pip install xfuser")
                print("3. Try using 'auto' parallel mode")
                print("4. Verify your GPU has enough memory")
            
            return (empty_image,)
    
    def _calculate_context_params(self, width, height, model_type):
        """
        Calculate optimal context parameters based on image dimensions and model type
        
        Args:
            width: Image width
            height: Image height
            model_type: Model type string
            
        Returns:
            (context_overlap, context_stride): Context overlap and stride parameters
        """
        # Default values based on model type
        overlap = 2
        stride = 1
        
        # Larger stride for big images
        if max(width, height) >= 1536:
            stride = 2
            
        # Adjust for specific model types
        model_type_lower = model_type.lower() if model_type else ""
        
        if "flux" in model_type_lower:
            # Flux models work better with larger overlap
            overlap = max(3, overlap)
        elif "pixart" in model_type_lower:
            # PixArt works better with larger stride and overlap
            overlap = max(3, overlap)
            stride = max(2, stride)
        elif "sdxl" in model_type_lower:
            # SDXL models typically need more overlap
            overlap = max(3, overlap)
            # For very large SDXL images, increase stride further
            if max(width, height) >= 2048:
                stride = max(3, stride)
        elif "sd3" in model_type_lower:
            # SD3 models need balanced parameters
            overlap = max(3, overlap)
            stride = max(2, stride)
        
        # Ensure parameters are within valid ranges
        overlap = min(8, max(1, overlap))
        stride = min(8, max(1, stride))
                
        return overlap, stride