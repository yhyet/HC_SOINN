# 将项目推送到新的 GitHub 仓库 - 操作指南

## 前提条件

1. 确保你在 GitHub 上已经创建了新的空仓库
2. 知道你的 GitHub 用户名
3. 知道新仓库的名称

## 操作步骤

### 步骤 1: 在 GitHub 网页上创建新仓库

1. 访问 https://github.com/new
2. 填写仓库名称（例如：`LAMDA-PILOT_2` 或你喜欢的名称）
3. **不要**初始化 README、.gitignore 或 license（因为本地已有代码）
4. 点击 "Create repository"

### 步骤 2: 移除或重命名旧的远程仓库

如果你想保留旧的远程仓库作为备份，可以重命名它：

```powershell
# 重命名旧的 origin 为 upstream（可选）
git remote rename origin upstream

# 或者直接移除（如果你确定不再需要）
# git remote remove origin
```

### 步骤 3: 添加新的远程仓库

```powershell
# 将 YOUR_USERNAME 替换为你的 GitHub 用户名
# 将 NEW_REPO_NAME 替换为你的新仓库名称
git remote add origin git@github.com:YOUR_USERNAME/NEW_REPO_NAME.git
```

### 步骤 4: 推送代码到新仓库

```powershell
# 推送 main 分支到新仓库
git push -u origin main

# 如果你想推送所有分支和标签
# git push -u origin --all
# git push -u origin --tags
```

## 快速命令模板

假设：
- GitHub 用户名：`your-username`
- 新仓库名称：`your-new-repo-name`

```powershell
# 1. 移除旧远程仓库（或重命名为 upstream）
git remote remove origin

# 2. 添加新远程仓库
git remote add origin git@github.com:your-username/your-new-repo-name.git

# 3. 推送代码
git push -u origin main
```

## 注意事项

- 确保你已提交所有更改，或至少知道哪些更改未提交
- 如果新仓库不是空的，可能需要先 pull 再 push，或者使用 `git push -u origin main --force`（谨慎使用）
- 使用 SSH 方式需要配置 SSH 密钥（你已经配置好了）




