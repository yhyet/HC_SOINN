# 环境配置说明

## 在服务器上创建 cl_lora_2 环境

### 方法 1: 使用完整版本（推荐）

```bash
# 将 environment_cl_lora_2.yaml 上传到服务器后执行
conda env create -f environment_cl_lora_2.yaml
```

### 方法 2: 使用服务器优化版本（更灵活）

```bash
# 将 environment_cl_lora_2_server.yaml 上传到服务器后执行
conda env create -f environment_cl_lora_2_server.yaml
```

### 激活环境

```bash
conda activate cl_lora_2
```

### 安装 CUDA 版本的 PyTorch（如果需要）

如果服务器有 GPU 且需要特定 CUDA 版本，可以：

```bash
# 激活环境后
conda activate cl_lora_2

# 对于 CUDA 11.8
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# 对于 CUDA 12.1
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu121

# 或者使用 conda（推荐）
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 验证安装

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 文件说明

- `environment_cl_lora_2.yaml`: 完整的环境配置，包含所有精确版本号
- `environment_cl_lora_2_server.yaml`: 服务器优化版本，部分依赖使用灵活版本号

## 注意事项

1. **CUDA 版本**: 原环境使用 CUDA 11.8 (`+cu118`)，服务器上需要根据实际 CUDA 版本安装对应的 PyTorch
2. **Python 版本**: 环境使用 Python 3.10.19
3. **核心依赖版本**:
   - torch: 2.0.1
   - torchvision: 0.15.2
   - timm: 0.6.12
   - numpy: 1.26.4
   - scipy: 1.15.3

## 如果遇到问题

1. **PyTorch 安装失败**: 检查服务器 CUDA 版本，使用对应的安装命令
2. **依赖冲突**: 可以尝试先创建基础环境，再逐步安装依赖
3. **网络问题**: 可以使用国内镜像源加速下载

```bash
# 使用清华镜像
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
```

