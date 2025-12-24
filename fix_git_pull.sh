#!/bin/bash
# Git Pull 问题修复脚本

echo "=== Git Pull 问题诊断和修复 ==="
echo ""

# 检查当前状态
echo "1. 检查当前远程仓库配置..."
git remote -v
echo ""

echo "2. 检查网络连接..."
if curl -s --max-time 5 -o /dev/null -w "%{http_code}" https://github.com | grep -q "200\|301\|302"; then
    echo "✓ GitHub连接正常"
else
    echo "✗ GitHub连接失败，可能需要使用镜像或代理"
fi
echo ""

echo "3. 尝试的解决方案："
echo ""

# 方案1: 尝试使用镜像（如果在中国）
read -p "是否尝试使用GitHub镜像？(y/n): " use_mirror
if [ "$use_mirror" = "y" ]; then
    echo "切换到镜像..."
    git remote set-url origin https://mirror.ghproxy.com/https://github.com/yhyet/CL_SOINN.git
    echo "尝试拉取..."
    git pull --tags origin main
    if [ $? -eq 0 ]; then
        echo "✓ 使用镜像拉取成功！"
        exit 0
    else
        echo "镜像也失败，尝试其他方案..."
        git remote set-url origin https://github.com/yhyet/CL_SOINN.git
    fi
fi
echo ""

# 方案2: 使用fetch + merge
echo "尝试使用 git fetch + merge..."
git fetch origin main
if [ $? -eq 0 ]; then
    echo "fetch成功，尝试merge..."
    git merge origin/main
    if [ $? -eq 0 ]; then
        echo "✓ 拉取成功！"
        exit 0
    fi
fi
echo ""

# 方案3: 检查是否需要认证
echo "如果仓库是私有的，需要配置认证："
echo "  选项A: 使用Personal Access Token"
echo "    git remote set-url origin https://[YOUR_TOKEN]@github.com/yhyet/CL_SOINN.git"
echo ""
echo "  选项B: 使用SSH（需要先配置SSH密钥）"
echo "    git remote set-url origin git@github.com:yhyet/CL_SOINN.git"
echo ""

echo "=== 诊断完成 ==="
echo ""
echo "如果问题仍然存在，请查看 GIT_PULL_SOLUTIONS.md 获取更多解决方案"

