# 将项目推送到新 GitHub 仓库的脚本
# 请修改下面的变量后运行此脚本

# ========== 配置区域 - 请修改这些变量 ==========
$GITHUB_USERNAME = "yhyet"  # 改为你的 GitHub 用户名
$NEW_REPO_NAME = "LAMDA-PILOT_2"  # 改为你的新仓库名称
# ==============================================

Write-Host "开始配置新仓库..." -ForegroundColor Green
Write-Host "GitHub 用户名: $GITHUB_USERNAME" -ForegroundColor Cyan
Write-Host "新仓库名称: $NEW_REPO_NAME" -ForegroundColor Cyan
Write-Host ""

# 切换到项目目录
Set-Location "D:\LAMDA-PILOT_2"

# 检查当前远程仓库
Write-Host "当前远程仓库配置:" -ForegroundColor Yellow
git remote -v
Write-Host ""

# 询问用户是否要保留旧的远程仓库
$keepOld = Read-Host "是否保留旧的远程仓库作为备份? (输入 'backup' 保留为 upstream, 或直接回车移除)"
if ($keepOld -eq "backup") {
    Write-Host "重命名 origin 为 upstream..." -ForegroundColor Yellow
    git remote rename origin upstream
    Write-Host "✓ 已重命名为 upstream" -ForegroundColor Green
} else {
    Write-Host "移除旧的 origin 远程仓库..." -ForegroundColor Yellow
    git remote remove origin
    Write-Host "✓ 已移除旧的 origin" -ForegroundColor Green
}

# 添加新的远程仓库
Write-Host "添加新的远程仓库..." -ForegroundColor Yellow
$NEW_REPO_URL = "git@github.com:$GITHUB_USERNAME/$NEW_REPO_NAME.git"
git remote add origin $NEW_REPO_URL
Write-Host "✓ 已添加新远程仓库: $NEW_REPO_URL" -ForegroundColor Green
Write-Host ""

# 显示新的远程仓库配置
Write-Host "新的远程仓库配置:" -ForegroundColor Yellow
git remote -v
Write-Host ""

# 检查是否有未提交的更改
$status = git status --porcelain
if ($status) {
    Write-Host "警告: 检测到未提交的更改!" -ForegroundColor Red
    Write-Host "未提交的文件:" -ForegroundColor Yellow
    git status --short
    Write-Host ""
    $commit = Read-Host "是否现在提交这些更改? (y/n)"
    if ($commit -eq "y" -or $commit -eq "Y") {
        $commitMsg = Read-Host "请输入提交信息 (直接回车使用默认信息)"
        if ([string]::IsNullOrWhiteSpace($commitMsg)) {
            $commitMsg = "Update before pushing to new repository"
        }
        git add .
        git commit -m $commitMsg
        Write-Host "✓ 更改已提交" -ForegroundColor Green
    }
}

# 询问是否推送
Write-Host ""
$push = Read-Host "是否现在推送到新仓库? (y/n)"
if ($push -eq "y" -or $push -eq "Y") {
    Write-Host "正在推送到新仓库..." -ForegroundColor Yellow
    git push -u origin main
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✓ 成功推送到新仓库!" -ForegroundColor Green
        Write-Host "仓库地址: https://github.com/$GITHUB_USERNAME/$NEW_REPO_NAME" -ForegroundColor Cyan
    } else {
        Write-Host ""
        Write-Host "✗ 推送失败，请检查:" -ForegroundColor Red
        Write-Host "  1. 是否在 GitHub 上创建了新仓库" -ForegroundColor Yellow
        Write-Host "  2. 仓库名称和用户名是否正确" -ForegroundColor Yellow
        Write-Host "  3. SSH 密钥是否已配置" -ForegroundColor Yellow
    }
} else {
    Write-Host "已配置远程仓库，但未推送。你可以稍后使用以下命令推送:" -ForegroundColor Yellow
    Write-Host "  git push -u origin main" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "完成!" -ForegroundColor Green





