FROM runpod/pytorch:2.0.1-py3.10-cuda11.7.1-runtime

WORKDIR /

# 安装依赖
RUN pip install --no-cache-dir \
    runpod \
    diffusers \
    transformers \
    accelerate \
    safetensors \
    controlnet_aux==0.0.7 \
    xformers

# 复制handler
COPY handler.py /handler.py
RUN chmod +x /handler.py

# 启动命令
CMD [ "python", "-u", "/handler.py" ]