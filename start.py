# start.py
import os
import torch
import base64
from io import BytesIO
from PIL import Image
import runpod
from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter
from controlnet_aux.midas import MidasDetector

# 全局变量存储模型
DEVICE = "cuda"
PIPE = None
MIDAS = None

def init_models():
    global PIPE, MIDAS
    
    # 初始化Midas检测器
    MIDAS = MidasDetector.from_pretrained(
        "valhalla/t2iadapter-aux-models", 
        filename="dpt_large_384.pt"
    ).to(DEVICE)
    
    # 初始化Adapter和Pipeline
    adapter = T2IAdapter.from_pretrained(
        "TencentARC/t2i-adapter-depth-midas-sdxl-1.0",
        torch_dtype=torch.float16
    ).to(DEVICE)
    
    PIPE = StableDiffusionXLAdapterPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        adapter=adapter,
        torch_dtype=torch.float16
    ).to(DEVICE)
    
    # 启用memory efficient attention
    PIPE.enable_xformers_memory_efficient_attention()

def process_image(image_data):
    """处理输入图像数据"""
    if isinstance(image_data, str):
        # 处理Base64编码的图像
        if image_data.startswith("data:image"):
            image_data = image_data.split(",")[1]
        image = Image.open(BytesIO(base64.b64decode(image_data)))
    else:
        # 处理URL或文件路径
        image = Image.open(image_data)
    return image

def image_to_base64(image):
    """将PIL图像转换为Base64"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def handler(event):
    """RunPod处理函数"""
    try:
        # 获取输入参数
        input_data = event["input"]
        
        # 必需参数
        image = process_image(input_data["image"])
        prompt = input_data.get("prompt", "")
        
        # 可选参数
        negative_prompt = input_data.get("negative_prompt", None)
        num_inference_steps = input_data.get("num_inference_steps", 30)
        guidance_scale = input_data.get("guidance_scale", 7.5)
        seed = input_data.get("seed", None)
        
        # 设置随机种子
        if seed is not None:
            torch.manual_seed(seed)
            
        # 生成深度图
        depth_image = MIDAS(image)
        
        # 生成图像
        output = PIPE(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=depth_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]
        
        # 返回结果
        return {
            "status": "success",
            "output": {
                "image": image_to_base64(output),
                "depth_map": image_to_base64(depth_image)
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

# 初始化模型
init_models()

# 启动RunPod
runpod.serverless.start({"handler": handler})
