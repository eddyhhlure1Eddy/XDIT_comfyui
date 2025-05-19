import os
import sys
import time
import torch
import torch.distributed
import io
from PIL import Image
import base64
from pathlib import Path

def reset_cuda_context():
    """重置CUDA上下文，避免与父进程冲突"""
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, '_lazy_init'):
        torch.cuda._lazy_init = True
    if hasattr(torch.cuda, '_initialized'):
        torch.cuda._initialized = False

def worker_fn(rank, world_size, model_path, comm_queue, result_queue):
    """
    xDiT工作进程函数
    
    处理分布式生成任务
    """
    try:
        # 设置异常处理
        def exception_handler(exc_type, exc_value, exc_traceback):
            print(f"工作进程 {rank} 异常: {exc_value}", file=sys.stderr)
            import traceback
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)
            # 确保主进程能收到错误通知
            if rank == 0:
                result_queue.put({
                    "status": "error",
                    "message": str(exc_value)
                })
        
        sys.excepthook = exception_handler
        
        # 重置CUDA上下文
        reset_cuda_context()
        
        # 设置分布式环境变量
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["LOCAL_RANK"] = str(rank)
        
        # 延迟导入xDiT相关模块，避免导入冲突
        try:
            from xfuser import (
                xFuserArgs,
                xFuserFluxPipeline,
                xFuserPixArtAlphaPipeline,
                xFuserPixArtSigmaPipeline,
                xFuserStableDiffusion3Pipeline,
                xFuserHunyuanDiTPipeline,
                xFuserWanDiTPipeline,
            )
            
            # 尝试导入SDXL支持
            try:
                from xfuser import xFuserSDXLPipeline
                has_sdxl_support = True
                print("SDXL模型支持已加载")
            except ImportError:
                # 如果官方xfuser包中没有SDXL支持，使用Flux作为备选
                has_sdxl_support = False
                print("警告: xfuser包中没有SDXL模型支持，将使用Flux pipeline作为SDXL的替代")
        except ImportError:
            if rank == 0:
                result_queue.put({
                    "status": "error",
                    "message": "Error: xfuser package not found. This node requires the official xDiT package. Please install it with: pip install xfuser or pip install \"xfuser[diffusers,flash-attn]\""
                })
            return
        
        # 打印工作进程状态
        print(f"工作进程 {rank}/{world_size} 已启动，使用设备 cuda:{rank}")
        
        # 设置设备
        torch.cuda.set_device(rank)
        
        # 初始化分布式环境
        if world_size > 1:
            try:
                torch.distributed.init_process_group(
                    backend="nccl", 
                    rank=rank, 
                    world_size=world_size
                )
                print(f"工作进程 {rank} 分布式环境初始化完成")
            except Exception as e:
                print(f"分布式环境初始化失败: {str(e)}，尝试使用单设备模式")
                # 如果分布式环境初始化失败，强制使用单设备模式
                world_size = 1
        
        # 模型初始化
        engine_args = None
        pipe = None
        input_config = None
        model_name = ""
        
        # 主循环
        while True:
            try:
                # 等待任务
                task = comm_queue.get()
                
                # 处理停止命令
                if task.get("action") == "stop":
                    print(f"工作进程 {rank} 收到停止命令")
                    break
                
                # 处理生成任务
                if task.get("action") == "generate":
                    # 检查模型是否已初始化
                    if pipe is None:
                        print(f"工作进程 {rank} 初始化模型 {model_path}")
                        
                        # 获取模型类型和名称
                        model_type = task.get("model_type", "flux")
                        model_name = model_path.split("/")[-1].lower()
                        
                        # 选择合适的Pipeline类
                        pipeline_class = None
                        if model_type == "sdxl" or any(x in model_name for x in ["sdxl", "sd-xl", "sd_xl", "xl-base", "ponyrealism", "dreamshaper_xl", "juggernaut_xl"]):
                            # 使用SDXL Pipeline，如果支持
                            if has_sdxl_support:
                                pipeline_class = xFuserSDXLPipeline
                                print(f"使用SDXL Pipeline处理模型: {model_name}")
                            else:
                                # 如果不支持SDXL，回退到Flux作为替代
                                pipeline_class = xFuserFluxPipeline
                                print(f"没有SDXL Pipeline支持，尝试使用Flux Pipeline作为替代处理: {model_name}")
                        elif "flux" in model_name or model_type == "flux":
                            pipeline_class = xFuserFluxPipeline
                        elif "pixart-sigma" in model_name or model_type == "pixart-sigma":
                            pipeline_class = xFuserPixArtSigmaPipeline
                        elif "pixart" in model_name or model_type == "pixart":
                            pipeline_class = xFuserPixArtAlphaPipeline
                        elif "stable-diffusion-3" in model_name or "sd3" in model_name or model_type == "sd3":
                            pipeline_class = xFuserStableDiffusion3Pipeline
                        elif "hunyuandit" in model_name or model_type == "hunyuan":
                            pipeline_class = xFuserHunyuanDiTPipeline
                        elif "wan" in model_name or model_type == "wan":
                            pipeline_class = xFuserWanDiTPipeline
                        else:
                            # 默认使用Flux
                            pipeline_class = xFuserFluxPipeline
                            print(f"未识别的模型类型 '{model_type}'，默认使用Flux Pipeline")
                        
                        # 获取并行参数
                        ulysses_degree = task.get("ulysses_degree", 2)
                        pipefusion_degree = task.get("pipefusion_degree", 1)
                        use_cfg_parallel = task.get("use_cfg_parallel", True)
                        use_parallel_vae = task.get("use_parallel_vae", False)
                        warmup_steps = task.get("warmup_steps", 1)
                        
                        # 验证并行参数有效性
                        if ulysses_degree * pipefusion_degree > world_size:
                            print(f"警告: 并行配置无效 (USP={ulysses_degree} * PF={pipefusion_degree} > {world_size})，自动调整")
                            # 调整并行参数
                            if ulysses_degree > pipefusion_degree:
                                ulysses_degree = max(1, world_size // pipefusion_degree)
                            else:
                                pipefusion_degree = max(1, world_size // ulysses_degree)
                            
                            print(f"调整后: USP={ulysses_degree}, PF={pipefusion_degree}")
                        
                        # 获取加速选项
                        use_torch_compile = task.get("use_torch_compile", False)
                        use_onediff = task.get("use_onediff", False)
                        use_teacache = task.get("use_teacache", False)
                        use_fbcache = task.get("use_fbcache", False)
                        use_dit_fast_attn = task.get("use_dit_fast_attn", False)
                        
                        # 获取VAE配置
                        vae_path = task.get("vae_path", None)
                        vae_precision = task.get("vae_precision", "fp16")
                        if vae_path and rank == 0:
                            print(f"Using VAE: {vae_path}")
                        
                        # TeaCache只能在USP模式下使用
                        if use_teacache and not (ulysses_degree > 1 and "flux" in model_type.lower()):
                            print("警告: TeaCache仅支持Flux模型和USP并行模式，已禁用")
                            use_teacache = False
                        
                        # 配置并行参数
                        try:
                            engine_args = xFuserArgs(
                                model=model_path,
                                trust_remote_code=True,
                                warmup_steps=warmup_steps,
                                use_parallel_vae=use_parallel_vae,
                                use_torch_compile=use_torch_compile,
                                use_onediff=use_onediff,
                                use_teacache=use_teacache,
                                use_fbcache=use_fbcache,
                                use_dit_fast_attn=use_dit_fast_attn,
                                ulysses_degree=ulysses_degree,  
                                pipefusion_parallel_degree=pipefusion_degree,
                                use_cfg_parallel=use_cfg_parallel,
                                dit_parallel_size=0,
                                # 可选优化参数
                                scheduler=task.get("scheduler", "euler_a"),
                                width=task.get("width", 1024),
                                height=task.get("height", 1024),
                                crop_w=task.get("crop_w", 0),
                                crop_h=task.get("crop_h", 0),
                                target_width=task.get("target_width", 512),
                                target_height=task.get("target_height", 512),
                                context_overlap=task.get("context_overlap", 3),
                                context_stride=task.get("context_stride", 3),
                            )
                            
                            # 创建配置
                            engine_config, input_config = engine_args.create_config()
                            
                            # 准备VAE（如果提供了路径）
                            vae = None
                            if vae_path and (model_type.lower() in ["sdxl", "sd3"]):
                                try:
                                    # 仅在需要时导入这些依赖
                                    from diffusers import AutoencoderKL
                                    
                                    print(f"工作进程 {rank} 加载VAE: {vae_path}")
                                    vae_dtype = torch.float16 if vae_precision == "fp16" else torch.float32
                                    vae = AutoencoderKL.from_pretrained(
                                        vae_path,
                                        torch_dtype=vae_dtype
                                    ).to(f"cuda:{rank}")
                                    print(f"工作进程 {rank} VAE加载成功")
                                except Exception as vae_err:
                                    print(f"工作进程 {rank} VAE加载失败: {str(vae_err)}")
                            
                            # 初始化Pipeline（带VAE）
                            pipe = pipeline_class.from_pretrained(
                                pretrained_model_name_or_path=model_path,
                                engine_config=engine_config,
                                torch_dtype=torch.float16,
                                vae=vae  # 传入VAE（如果有）
                            ).to(f"cuda:{rank}")
                            
                            # 准备运行环境
                            pipe.prepare_run(input_config)
                            print(f"工作进程 {rank} 模型初始化完成")
                            
                        except Exception as e:
                            # 初始化失败，尝试使用基础配置
                            if rank == 0:
                                print(f"高级模式初始化失败: {str(e)}，尝试回退到基础模式...")
                            
                            # 禁用所有高级特性
                            use_torch_compile = False
                            use_onediff = False
                            use_teacache = False
                            use_fbcache = False
                            use_dit_fast_attn = False
                            
                            # 使用最基础的并行配置
                            if world_size > 1:
                                ulysses_degree = 1
                                pipefusion_degree = 1
                                use_cfg_parallel = world_size >= 2
                            
                            # 重新创建配置
                            engine_args = xFuserArgs(
                                model=model_path,
                                trust_remote_code=True,
                                warmup_steps=1,
                                use_parallel_vae=False,
                                ulysses_degree=ulysses_degree,
                                pipefusion_parallel_degree=pipefusion_degree,
                                use_cfg_parallel=use_cfg_parallel,
                                scheduler=task.get("scheduler", "euler_a"),
                                width=task.get("width", 1024),
                                height=task.get("height", 1024),
                                context_overlap=task.get("context_overlap", 3),
                                context_stride=task.get("context_stride", 3),
                            )
                            
                            engine_config, input_config = engine_args.create_config()
                            
                            # 尝试使用基础配置初始化
                            try:
                                pipe = pipeline_class.from_pretrained(
                                    pretrained_model_name_or_path=model_path,
                                    engine_config=engine_config,
                                    torch_dtype=torch.float16,
                                ).to(f"cuda:{rank}")
                                
                                pipe.prepare_run(input_config)
                                print(f"工作进程 {rank} 使用基础模式初始化完成")
                            except Exception as e2:
                                # 如果基础模式也失败，报告错误
                                error_msg = f"模型初始化失败: {str(e2)}"
                                print(f"工作进程 {rank} {error_msg}")
                                if rank == 0:
                                    result_queue.put({
                                        "status": "error",
                                        "message": error_msg
                                    })
                                continue
                    
                    # Execute generation with detailed diagnostics
                    if rank == 0:  # Only log on main node
                        print(f"[WORKER-{rank}] Starting image generation at {time.strftime('%H:%M:%S')}")
                        print(f"[WORKER-{rank}] Prompt: {task.get('prompt', '')}")
                        print(f"[WORKER-{rank}] Device: cuda:{rank}, CUDA memory: {torch.cuda.memory_allocated(rank)/(1024**2):.1f}MB allocated")
                    
                    start_time = time.time()
                    
                    # Check if too much CUDA memory is already allocated
                    if torch.cuda.memory_allocated(rank) / torch.cuda.max_memory_allocated(rank) > 0.8:
                        print(f"[WORKER-{rank}] WARNING: High CUDA memory usage detected before starting generation!")
                    
                    # Set generation parameters with diagnostics
                    print(f"[WORKER-{rank}] Preparing generation parameters")
                    params = {
                        "height": task.get("height", 1024),
                        "width": task.get("width", 1024),
                        "prompt": task.get("prompt", ""),
                        "negative_prompt": task.get("negative_prompt", ""),
                        "num_inference_steps": task.get("steps", 20),
                        "output_type": "pil",
                        "guidance_scale": task.get("cfg_scale", 7.5),
                        "generator": torch.Generator(device="cuda").manual_seed(task.get("seed", 42)),
                    }
                    print(f"[WORKER-{rank}] Parameters prepared: {params['width']}x{params['height']}, steps={params['num_inference_steps']}, cfg={params['guidance_scale']}")
                    print(f"[WORKER-{rank}] Using seed: {task.get('seed', 42)}")
                    
                    # Report pipeline model details
                    if hasattr(pipe, "_name_or_path") and pipe._name_or_path:
                        print(f"[WORKER-{rank}] Pipeline path: {pipe._name_or_path}")
                    if hasattr(pipe, "_execution_device"):
                        print(f"[WORKER-{rank}] Pipeline execution device: {pipe._execution_device}")
                    
                    # Memory check before generation
                    torch.cuda.empty_cache()  # Try to clear any unused memory
                    print(f"[WORKER-{rank}] CUDA memory before generation: {torch.cuda.memory_allocated(rank)/(1024**2):.1f}MB allocated, {torch.cuda.max_memory_allocated(rank)/(1024**2):.1f}MB peak")
                    # Add start_at_step parameter
                    if task.get("start_at_step", 0) > 0:
                        params["start_at_step"] = task.get("start_at_step")
                        print(f"[WORKER-{rank}] Using start_at_step={task.get('start_at_step')}")
                    
                    # DETAILED PROGRESS LOGGING BEFORE GENERATION
                    print(f"[WORKER-{rank}] ===== STARTING MODEL INFERENCE AT {time.strftime('%H:%M:%S')} =====")
                    print(f"[WORKER-{rank}] Step 1/4: Preparing pipeline components...")
                    
                    # Execute generation with error handling and detailed progress reporting
                    try:
                        # Add step debugger hook to track progress
                        original_step_callback = None
                        current_step = 0
                        
                        # Define callback to track progress
                        def progress_callback(pipe, i, t, callback_kwargs):
                            nonlocal current_step
                            current_step = i
                            # Only log every 5 steps to avoid log overload
                            if i == 0 or i % 5 == 0 or i == params["num_inference_steps"]-1:
                                print(f"[WORKER-{rank}] Diffusion step {i+1}/{params['num_inference_steps']} at {time.strftime('%H:%M:%S')}")
                            # Call original callback if it exists
                            if original_step_callback is not None:
                                return original_step_callback(pipe, i, t, callback_kwargs)
                            return callback_kwargs
                        
                        # Save original callback and set our tracking callback
                        if hasattr(pipe, "callback_steps") and hasattr(pipe, "_callback"):
                            original_step_callback = pipe._callback
                            pipe._callback = progress_callback
                            pipe.callback_steps = 1  # Call on every step
                            print(f"[WORKER-{rank}] Progress tracking enabled")
                        
                        # Log memory before starting
                        torch.cuda.empty_cache()
                        print(f"[WORKER-{rank}] Step 2/4: Beginning text encoding...")
                        memory_before = torch.cuda.memory_allocated(rank)/(1024**2)
                        print(f"[WORKER-{rank}] GPU memory before inference: {memory_before:.1f}MB")
                        
                        # Check if VAE exists and set appropriate output type
                        if hasattr(pipe, "vae") and pipe.vae is not None:
                            print(f"[WORKER-{rank}] VAE found in pipeline, setting output_type='pil'")
                            params["output_type"] = "pil"  # Can use PIL output since VAE is present
                        else:
                            print(f"[WORKER-{rank}] No VAE found, setting output_type='latent'")
                            params["output_type"] = "latent"  # Without VAE, we need to get latents
                        
                        # Execute pipeline
                        inference_start = time.time()
                        print(f"[WORKER-{rank}] Step 3/4: Starting diffusion sampling process...")
                        output = pipe(**params)
                        
                        # Log completion
                        print(f"[WORKER-{rank}] Step 4/4: Diffusion completed, post-processing...")
                        
                        end_time = time.time()
                        elapsed = end_time - start_time
                        inference_time = end_time - inference_start
                        
                        # Report memory usage
                        memory_after = torch.cuda.memory_allocated(rank)/(1024**2)
                        print(f"[WORKER-{rank}] GPU memory after inference: {memory_after:.1f}MB (change: {memory_after-memory_before:.1f}MB)")
                        print(f"[WORKER-{rank}] ===== INFERENCE COMPLETE: {inference_time:.2f}s for sampling, {elapsed:.2f}s total =====")
                        
                        # 处理生成结果，根据输出类型决定处理方式
                        if hasattr(output, "images"):
                            if params["output_type"] == "pil":
                                # 已经是PIL图像
                                images = output.images
                                print(f"工作进程 {rank} 接收到 {len(images)} 张PIL格式图像")
                            elif params["output_type"] == "latent":
                                # 需要手动将潜空间转换为图像
                                try:
                                    from diffusers.image_processor import VaeImageProcessor
                                    import numpy as np
                                    from PIL import Image
                                    
                                    print(f"工作进程 {rank} 手动转换潜空间输出为图像")
                                    latents = output.images
                                    # 标准化潜空间张量
                                    latents = 1 / 0.18215 * latents
                                    
                                    # 创建简单处理器用于潜空间→图像转换
                                    processor = VaeImageProcessor()
                                    images = []
                                    for latent in latents:
                                        # 转为numpy
                                        image = processor.postprocess(latent)  # 返回numpy数组
                                        # 转为PIL
                                        pil_img = Image.fromarray((image * 255).astype(np.uint8))
                                        images.append(pil_img)
                                    
                                    print(f"工作进程 {rank} 成功转换 {len(images)} 个潜空间张量为图像")
                                except Exception as e:
                                    print(f"工作进程 {rank} 潜空间转换错误: {str(e)}")
                                    raise e
                        else:
                            raise ValueError(f"不支持的输出类型: {params['output_type']}")
                        
                        if len(images) > 0:
                            # 将PIL图像转换为base64字符串
                            import base64
                            import io
                            
                            # 确保有有效的图像
                            image_b64s = []
                            
                            for img in images:
                                buffered = io.BytesIO()
                                img.save(buffered, format="PNG")
                                img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                                image_b64s.append(img_b64)
                            
                            # 初始化latent变量
                            latent_data = None
                            if hasattr(output, "latents") and output.latents is not None:
                                latent_data = output.latents.cpu().numpy().tolist()
                            
                            # 将结果放入结果队列
                            result = {
                                "status": "success",
                                "elapsed_time": elapsed,
                                "image": image_b64s[0]
                            }
                            
                            if latent_data is not None:
                                result["latent"] = latent_data
                                
                            result_queue.put(result)
                        else:
                            print("生成失败，未能获得有效输出")
                            result_queue.put({
                                "status": "error",
                                "message": "生成失败，未能获得有效输出"
                            })
                    except Exception as e:
                        error_msg = f"生成过程错误: {str(e)}"
                        print(f"工作进程 {rank} {error_msg}")
                        import traceback
                        traceback.print_exc()
                        # 确保错误被报告回主进程
                        if rank == 0:
                            result_queue.put({
                                "status": "error",
                                "message": error_msg
                            })
                        if rank == 0:
                            # 只有主进程报告错误
                            result_queue.put({
                                "status": "error",
                                "message": error_msg
                            })
            except Exception as e:
                print(f"工作进程 {rank} 发生错误: {str(e)}")
                import traceback
                traceback.print_exc()
                if rank == 0:
                    # 只有主进程报告错误
                    result_queue.put({
                        "status": "error",
                        "message": str(e)
                    })
        
        # 清理资源
        if pipe is not None:
            del pipe
        
        # 关闭分布式环境
        if world_size > 1 and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            
        print(f"工作进程 {rank} 已退出")
            
    except Exception as e:
        print(f"工作进程 {rank} 初始化失败: {str(e)}")
        # 确保主进程能收到错误通知
        if rank == 0:
            result_queue.put({
                "status": "error",
                "message": f"工作进程初始化失败: {str(e)}"
            }) 