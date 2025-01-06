#!/usr/bin/env python
import runpod
import torch
from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter
from controlnet_aux.midas import MidasDetector
import base64
from io import BytesIO
from PIL import Image

# 全局变量
DEVICE = "cuda"
PIPE = None
MIDAS = None

def load_models():
    global PIPE, MIDAS
    
    if PIPE is not None and MIDAS is not None:
        return
    
    try:
        # 加载Midas
        MIDAS = MidasDetector.from_pretrained(
            "/models/midas",
            local_files_only=True
        ).to(DEVICE)
        
        # 加载Adapter
        adapter = T2IAdapter.from_pretrained(
            "/models/adapter",
            local_files_only=True,
            torch_dtype=torch.float16
        ).to(DEVICE)
        
        # 加载SDXL
        PIPE = StableDiffusionXLAdapterPipeline.from_pretrained(
            "/models/sdxl",
            adapter=adapter,
            torch_dtype=torch.float16,
            local_files_only=True
        ).to(DEVICE)
        
        PIPE.enable_xformers_memory_efficient_attention()
        
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise e

def handler(event):
    """
    处理请求的handler函数
    """
    try:
        # 确保模型已加载
        load_models()
        
        # 获取输入
        if "input" not in event:
            return {"error": "No input provided"}
            
        job_input = event["input"]
        
        if "image" not in job_input or "prompt" not in job_input:
            return {"error": "Missing required input: image or prompt"}
            
        # 处理输入图像
        image = Image.open(BytesIO(base64.b64decode(job_input["image"])))
        prompt = job_input["prompt"]
        
        # 可选参数
        negative_prompt = job_input.get("negative_prompt", None)
        num_inference_steps = job_input.get("num_inference_steps", 30)
        guidance_scale = job_input.get("guidance_scale", 7.5)
        
        # 生成深度图
        depth_image = MIDAS(image)
        
        # 生成图像
        output = PIPE(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=depth_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images[0]
        
        # 转换输出为base64
        buffered = BytesIO()
        output.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "output": {
                "image": image_base64
            }
        }
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({
        "handler": handler
    })