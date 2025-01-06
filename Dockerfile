# 使用CUDA基础镜像
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# 安装Python和基础工具
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    # 添加图像处理所需的库
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx

# 安装依赖
RUN pip3 install --no-cache-dir \
    torch \
    diffusers \
    transformers \
    accelerate \
    safetensors \
    controlnet_aux==0.0.7 \
    xformers \
    Pillow \
    runpod \
    opencv-python

# 创建工作目录
WORKDIR /workspace

# 下载模型文件
RUN python3 -c "from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter; \
    from controlnet_aux.midas import MidasDetector; \
    adapter = T2IAdapter.from_pretrained('TencentARC/t2i-adapter-depth-midas-sdxl-1.0'); \
    pipe = StableDiffusionXLAdapterPipeline.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0'); \
    midas = MidasDetector.from_pretrained('valhalla/t2iadapter-aux-models', filename='dpt_large_384.pt')"

# 复制启动脚本
COPY start.py .

# 设置健康检查
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动服务
CMD ["python3", "start.py"]
