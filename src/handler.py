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

def init_models():
    global PIPE, MIDAS
    
    # 初始化Midas检测器
    MIDAS = MidasDetector.from_pretrained(
        "/workspace/models/midas",
        filename="dpt_large_384.pt",
        local_files_only=True
    ).to(DEVICE)
    
    # 初始化Adapter和Pipeline
    adapter = T2IAdapter.from_pretrained(
        "/workspace/models/adapter",
        torch_dtype=torch.float16,
        local_files_only=True
    ).to(DEVICE)
    
    PIPE = StableDiffusionXLAdapterPipeline.from_pretrained(
        "/workspace/models/sdxl",
        adapter=adapter,
        torch_dtype=torch.float16,
        local_files_only=True
    ).to(DEVICE)
    
    PIPE.enable_xformers_memory_efficient_attention()

def handler(event):
    try:
        # 确保模型已初始化
        if PIPE is None or MIDAS is None:
            init_models()
            
        # 获取输入参数
        input_data = event["input"]
        
        # 处理输入图像
        image = Image.open(BytesIO(base64.b64decode(input_data["image"])))
        prompt = input_data["prompt"]
        
        # 可选参数
        negative_prompt = input_data.get("negative_prompt", None)
        num_inference_steps = input_data.get("num_inference_steps", 30)
        guidance_scale = input_data.get("guidance_scale", 7.5)
        
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
        
        # 转换为base64
        buffered = BytesIO()
        output.save(buffered, format="PNG")
        output_image = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "status": "success",
            "output": {
                "image": output_image
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

runpod.serverless.start({"handler": handler})