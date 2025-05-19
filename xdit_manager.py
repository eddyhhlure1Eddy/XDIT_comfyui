import os
import sys
import time
import threading
import torch
import torch.distributed
import torch.multiprocessing as mp
import queue
from pathlib import Path
import folder_paths
from PIL import Image
import numpy as np

# 单例管理器
class XDiTDistributedManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(XDiTDistributedManager, cls).__new__(cls)
                cls._instance.initialized = False
                cls._instance.worker_processes = []
                cls._instance.comm_queues = []
                cls._instance.result_queues = []
                cls._instance.model_path = None
                cls._instance.world_size = 0
                cls._instance.vae = None
            return cls._instance
    
    def initialize(self, model_path, world_size=None):
        """初始化分布式环境和工作进程"""
        if self.initialized and self.model_path == model_path:
            return
            
        # 关闭之前的进程（如果有）
        self.shutdown()
        
        self.model_path = model_path
        self.world_size = world_size or torch.cuda.device_count()
        
        if self.world_size <= 0:
            raise ValueError(f"没有可用的GPU。world_size = {self.world_size}")
            
        print(f"正在初始化xDiT分布式环境，使用GPU数量: {self.world_size}")
        
        # 导入工作进程函数
        from .xdit_worker import worker_fn
        
        # 创建工作进程
        for rank in range(self.world_size):
            # 创建通信队列
            comm_queue = mp.Queue()
            result_queue = mp.Queue()
            
            self.comm_queues.append(comm_queue)
            self.result_queues.append(result_queue)
            
            # 启动进程
            p = mp.Process(
                target=worker_fn,
                args=(rank, self.world_size, model_path, comm_queue, result_queue),
                daemon=True
            )
            p.start()
            self.worker_processes.append(p)
            print(f"启动工作进程 {rank}/{self.world_size}")
        
        self.initialized = True
        print("xDiT分布式环境初始化完成")
    
    def generate(self, task_data, timeout=90):
        """Execute generation task"""
        if not self.initialized:
            raise RuntimeError("Please initialize distributed environment first")
        
        # Set task identifier for tracking
        task_id = f"task_{time.time()}"
        task_data["task_id"] = task_id
        print(f"[MANAGER] Sending task {task_id} to worker processes")
        print(f"[MANAGER] Task details: prompt='{task_data.get('prompt', '')}', steps={task_data.get('steps', 0)}, seed={task_data.get('seed', 0)}")
            
        # Send task to all worker processes
        print(f"[MANAGER] Worker queue count: {len(self.comm_queues)}")
        for i, q in enumerate(self.comm_queues):
            print(f"[MANAGER] Sending task to worker {i}")
            q.put(task_data)
        print(f"[MANAGER] Tasks dispatched to all workers")
        
        # Process results from main node (rank 0)
        print(f"[MANAGER] Waiting for result from main worker (timeout: {timeout}s)")
        try:
            # Use shorter timeout
            check_interval = 15
            total_waited = 0
            
            # Print status updates while waiting for result
            while total_waited < timeout:
                try:
                    # Check if result is ready with short timeout
                    result = self.result_queues[0].get(timeout=check_interval)
                    # If we get here, we have a result
                    if result.get("status") == "success":
                        print(f"[MANAGER] Task {task_id} completed successfully")
                    else:
                        print(f"[MANAGER] Task {task_id} failed: {result.get('message', 'unknown error')}")
                    return result
                except queue.Empty:
                    # No result yet, print status update
                    total_waited += check_interval
                    print(f"[MANAGER] Still waiting for result... ({total_waited}/{timeout}s)")
                    self._check_worker_health()
            
            # If we exit the loop, we've reached the timeout
            print(f"[MANAGER] Task {task_id} timed out! No response after {timeout} seconds")
            
            # Check worker process health
            print("[MANAGER] Checking worker health...")
            self._check_worker_health()
            
            # Try to send stop signal to all workers
            print("[MANAGER] Sending emergency stop signal to all workers")
            for q in self.comm_queues:
                try:
                    q.put({"action": "stop"}, block=False)
                except Exception as e:
                    print(f"[MANAGER] Failed to send stop signal: {e}")
            
            return {
                "status": "error",
                "message": f"xDiT generation task timed out ({timeout}s). Possible causes: GPU out of memory, process hung, or internal error"
            }
        except Exception as e:
            print(f"[MANAGER] Unexpected error while waiting for results: {e}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"xDiT generation failed with error: {str(e)}"
            }
    
    def encode_image(self, image, device=None):
        """
        使用VAE将图像编码为潜在表示
        
        Args:
            image: PIL图像
            device: 设备
            
        Returns:
            ComfyUI格式的latent
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 检查VAE是否已加载
        if self.vae is None:
            # 尝试加载VAE
            try:
                from diffusers import AutoencoderKL
                # 查找VAE模型
                vae_path = None
                vae_files = folder_paths.get_filename_list("vae")
                for vae_file in vae_files:
                    if "sdxl" in vae_file.lower() or "flux" in vae_file.lower():
                        vae_path = folder_paths.get_full_path("vae", vae_file)
                        break
                
                if vae_path is None:
                    # 尝试从Hugging Face加载
                    self.vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float16).to(device)
                else:
                    # 从本地加载
                    self.vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16).to(device)
                    
                print(f"VAE模型加载成功: {vae_path}")
            except Exception as e:
                print(f"VAE模型加载失败: {str(e)}")
                raise ValueError("无法加载VAE模型，请检查模型路径")
        
        # 图像预处理
        if not isinstance(image, np.ndarray):
            img_array = np.array(image).astype(np.float32) / 255.0
            img_array = img_array[None].transpose(0, 3, 1, 2)
            input_tensor = torch.from_numpy(img_array).to(device, dtype=torch.float16)
        else:
            img_array = image.astype(np.float32) / 255.0
            img_array = img_array[None].transpose(0, 3, 1, 2)
            input_tensor = torch.from_numpy(img_array).to(device, dtype=torch.float16)
        
        # 执行编码
        with torch.no_grad():
            latent = self.vae.encode(input_tensor).latent_dist.sample()
            latent = latent * 0.18215
        
        # 转换为ComfyUI格式
        return {"samples": latent.cpu()}
    
    def _check_worker_health(self):
        """检查工作进程状态"""
        for i, p in enumerate(self.worker_processes):
            if not p.is_alive():
                print(f"工作进程 {i} 已死亡")
                
    def shutdown(self):
        """关闭分布式环境"""
        if not self.initialized:
            return
            
        print("正在关闭xDiT分布式环境...")
        
        # 发送停止信号
        for q in self.comm_queues:
            try:
                q.put({"action": "stop"}, timeout=5)
            except:
                pass
            
        # 等待进程结束
        for p in self.worker_processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        
        # 清理资源
        self.worker_processes = []
        self.comm_queues = []
        self.result_queues = []
        self.initialized = False
        
        # 释放VAE
        if self.vae is not None:
            del self.vae
            self.vae = None
            torch.cuda.empty_cache()
        
        print("xDiT分布式环境已关闭")
        
    def __del__(self):
        """析构函数，确保资源被释放"""
        self.shutdown()

# 全局管理器实例
manager = XDiTDistributedManager() 