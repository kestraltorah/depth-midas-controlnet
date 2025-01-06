FROM runpod/base:0.4.0

WORKDIR /

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY requirements.txt /requirements.txt
COPY src /src

# 安装Python依赖
RUN pip3 install --no-cache-dir -r /requirements.txt

# 创建模型目录
RUN mkdir -p /models

# 下载模型
RUN python3 /src/download.py

# 设置权限
RUN chmod +x /src/handler.py

# 启动服务
CMD [ "python3", "-u", "/src/handler.py" ]