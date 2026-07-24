# 使用官方 Python 3.11 slim 镜像作为基础（体积比完整版小很多，适合生产部署）
FROM python:3.11-slim

# 设置容器内的工作目录，后续所有相对路径操作都基于这里
WORKDIR /app

# 安装系统级编译依赖
# 部分Python包（如cryptography、grpcio）在某些平台上需要编译，装了以防万一
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 先只复制依赖清单，不复制代码
# 这样只要requirements.txt不变，Docker就能复用这一层的缓存，不用每次都重新装一遍依赖
COPY requirements.txt .

# 关键优化：单独先装CPU-only版本的torch
# 默认从PyPI装torch会连GPU（CUDA）版本一起装进来，体积能到几个GB
# 这个项目只用torch做embedding推理，不需要GPU，装CPU版本能大幅减小镜像体积
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 再装其余依赖。因为torch已经装好且版本号匹配，这一步不会重复下载torch
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码（放在依赖安装之后，代码改动不会导致依赖重新安装）
COPY . .

# 声明容器对外监听的端口
EXPOSE 8000

# 容器启动命令
# 注意：必须显式写 --host 0.0.0.0，不能依赖main.py里的127.0.0.1
# 因为0.0.0.0表示监听所有网络接口，外部（包括宿主机）才能访问到容器内的服务
# 而127.0.0.1只能被容器内部自己访问，端口映射也没用
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
