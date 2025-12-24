# Git Pull 失败解决方案

## 问题描述
从服务器向GitHub拉取代码时出现TLS连接错误：
```
fatal: unable to access 'https://github.com/yhyet/CL_SOINN.git/': gnutls_handshake() failed: The TLS connection was non-properly terminated.
```

## 已尝试的解决方案

### 1. Git配置优化（已执行）
已配置以下git设置来改善连接：
- 增加HTTP缓冲区大小
- 增加超时时间
- 使用HTTP/1.1协议

### 2. 其他解决方案

#### 方案A: 使用GitHub镜像（推荐，如果在中国大陆）
如果服务器位于中国大陆，可以使用GitHub镜像：

```bash
# 使用镜像站点
git remote set-url origin https://mirror.ghproxy.com/https://github.com/yhyet/CL_SOINN.git
git pull --tags origin main

# 或者使用其他镜像
git remote set-url origin https://ghproxy.com/https://github.com/yhyet/CL_SOINN.git
git pull --tags origin main
```

#### 方案B: 配置代理（如果有代理服务器）
如果有可用的HTTP/HTTPS代理：

```bash
# 设置代理（替换为实际代理地址）
git config --global http.proxy http://proxy.example.com:8080
git config --global https.proxy https://proxy.example.com:8080

# 拉取代码
git pull --tags origin main

# 如果不再需要代理，可以取消设置
# git config --global --unset http.proxy
# git config --global --unset https.proxy
```

#### 方案C: 使用SSH协议（需要配置SSH密钥）
如果已配置SSH密钥，可以切换到SSH协议：

```bash
# 切换到SSH协议
git remote set-url origin git@github.com:yhyet/CL_SOINN.git

# 拉取代码
git pull --tags origin main
```

#### 方案D: 手动下载并合并
如果网络问题持续，可以手动下载：

```bash
# 1. 下载最新代码为zip文件（通过浏览器或其他方式）
# 2. 解压到临时目录
# 3. 手动合并更改

# 或者使用wget/curl下载（如果这些工具可以访问GitHub）
cd /tmp
wget https://github.com/yhyet/CL_SOINN/archive/refs/heads/main.zip
unzip main.zip
# 然后手动复制文件或使用git apply
```

#### 方案E: 使用git fetch替代pull
有时fetch比pull更稳定：

```bash
git fetch origin main
git merge origin/main
```

#### 方案F: 检查网络连接和DNS
```bash
# 测试DNS解析
nslookup github.com

# 测试端口连接
telnet github.com 443

# 如果DNS有问题，可以尝试使用8.8.8.8作为DNS服务器
```

#### 方案G: 使用不同的SSL后端
如果gnutls有问题，可以尝试使用openssl：

```bash
# 检查git编译时使用的SSL库
git --version

# 如果可能，重新编译git使用openssl，或者使用系统包管理器安装使用openssl的git版本
```

## 最新状态

网络连接问题已解决，但出现了认证问题。如果仓库是私有的，需要配置认证：

### 配置GitHub认证

#### 使用Personal Access Token (推荐)
```bash
# 1. 在GitHub上生成Personal Access Token
# Settings -> Developer settings -> Personal access tokens -> Tokens (classic)
# 勾选 repo 权限

# 2. 配置git使用token
git config --global credential.helper store
# 或者在URL中包含token
git remote set-url origin https://[YOUR_TOKEN]@github.com/yhyet/CL_SOINN.git
```

#### 使用SSH密钥（长期方案）
```bash
# 1. 生成SSH密钥（如果还没有）
ssh-keygen -t ed25519 -C "your_email@example.com"

# 2. 将公钥添加到GitHub
cat ~/.ssh/id_ed25519.pub
# 复制输出到 GitHub -> Settings -> SSH and GPG keys

# 3. 切换到SSH协议
git remote set-url origin git@github.com:yhyet/CL_SOINN.git

# 4. 测试连接
ssh -T git@github.com

# 5. 拉取代码
git pull --tags origin main
```

## 当前状态检查

检查当前git配置：
```bash
git config --list | grep -E "(http|proxy|ssl|tls|credential)"
git remote -v
```

检查网络连接：
```bash
curl -I https://github.com
ping github.com
```

## 建议

1. **首先尝试方案A（镜像）**：如果服务器在中国大陆，这是最可靠的方案
2. **如果有代理，使用方案B**：配置代理通常能解决网络问题
3. **长期解决方案**：考虑配置SSH密钥并使用SSH协议（方案C）

## 注意事项

- 使用镜像后，记得在拉取完成后将远程URL改回原始地址（如果需要）
- 代理配置会影响所有git操作，记得在不需要时取消设置
- SSH协议需要先在GitHub上配置SSH密钥

