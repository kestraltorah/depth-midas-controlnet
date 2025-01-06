FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel

WORKDIR /workspace

# 复制项目文件
COPY builder/requirements.txt /workspace/builder/
COPY src /workspace/src

# 安装依赖
RUN pip install -r /workspace/builder/requirements.txt

# 创建模型目录
RUN mkdir -p /workspace/models

# 下载模型
RUN python3 /workspace/src/download_models.py

# 启动服务
CMD [ "python3", "-u", "/workspace/src/handler.py" ]