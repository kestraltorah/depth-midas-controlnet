# 使用RunPod的基础镜像
FROM runpod/pytorch:3.10-2.0.1-117

# 设置工作目录
WORKDIR /workspace

# 安装依赖
RUN pip3 install --no-cache-dir \
    diffusers \
    transformers \
    accelerate \
    safetensors \
    controlnet_aux==0.0.7 \
    xformers \
    runpod \
    opencv-python

# 创建模型缓存目录
RUN mkdir -p /workspace/models

# 下载模型 - 修改了加载顺序和方式
RUN python3 -c 'from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter; \
    from controlnet_aux.midas import MidasDetector; \
    print("Downloading models..."); \
    adapter = T2IAdapter.from_pretrained("TencentARC/t2i-adapter-depth-midas-sdxl-1.0"); \
    pipe = StableDiffusionXLAdapterPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", adapter=adapter); \
    midas = MidasDetector.from_pretrained("valhalla/t2iadapter-aux-models", filename="dpt_large_384.pt"); \
    print("Models downloaded successfully!")'

# 添加启动脚本
COPY start.py .

# 启动服务
CMD [ "python3", "-u", "start.py" ]
