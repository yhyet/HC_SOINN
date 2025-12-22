# Procrustes 距离测量原理与实现说明

## 1. Procrustes 距离的原理

### 基本概念

Procrustes 距离是一种用于比较两个点集形状相似性的度量方法。它通过**相似变换**（Similarity Transformation）将一个点集对齐到另一个点集，然后计算对齐后的残差距离。

### 数学公式

给定两个点集：
- **X1**: 参考点集 [N, D]（Task 1 的特征）
- **X2**: 目标点集 [N, D]（当前任务的特征）

Procrustes 分析的目标是找到最优的**旋转矩阵 R** 和**缩放因子 s**，使得：

```
min_{R, s} ||X2 - s * X1 @ R||_F^2
```

其中：
- **R**: 正交旋转矩阵（R^T @ R = I）
- **s**: 缩放因子（非负实数）
- **||·||_F**: Frobenius 范数

### 计算步骤

#### Step 1: 去中心化（Centering）
消除平移差异：
```python
mu1 = X1.mean(axis=0)
mu2 = X2.mean(axis=0)
X1_centered = X1 - mu1
X2_centered = X2 - mu2
```

#### Step 2: 计算最优旋转矩阵 R
使用 SVD 分解：
```python
M = X1_centered.T @ X2_centered  # [D, D]
U, S, Vt = np.linalg.svd(M, full_matrices=False)
R = U @ Vt  # 正交旋转矩阵
```

#### Step 3: 计算最优缩放因子 s
**修正后的公式**：
```python
X1_rotated = X1_centered @ R
s = trace(X1_rotated^T @ X2_centered) / trace(X1_rotated^T @ X1_rotated)
```

**注意**：如果 s < 0，说明需要反射（reflection），但我们只考虑旋转+缩放，此时使用 Frobenius 范数比作为 fallback。

#### Step 4: 应用变换并计算距离
```python
X1_transformed = s * X1_rotated
diff = X2_centered - X1_transformed
procrustes_dist = ||diff||_F
```

#### Step 5: 归一化
```python
normalized_dist = procrustes_dist / ||X2_centered||_F
```

## 2. 发现的 Bug 及修复

### Bug 1: 缩放因子计算错误（已修复）

**原始实现**：
```python
s = sum(S) / trace(X1_centered^T @ X1_centered)
```

**问题**：
- `sum(S)` 是 `M = X1^T @ X2` 的奇异值之和，不是正确的缩放因子
- 这会导致缩放因子计算错误，影响最终距离

**修复后**：
```python
X1_rotated = X1_centered @ R
s = trace(X1_rotated^T @ X2_centered) / trace(X1_rotated^T @ X1_rotated)
```

### Bug 2: CL-LoRA 特征提取问题（已修复）

**原始实现**：
- 使用 `extract_vector()`，包含 general_lora 和 specific_lora

**问题**：
- 应该只测试 general_lora 的特征漂移，不应该包含 specific_lora

**修复后**：
- 创建自定义特征提取函数，只使用 `general_pos` 位置的 adapter
- 跳过 `specific_pos` 位置的 adapter

## 3. 使用场景

### 在增量学习中的应用

Procrustes 距离用于测量**特征漂移**（Feature Drift）：

1. **Task 1 结束后**：保存所有训练样本（图像和特征）
2. **后续任务**：使用当前模型重新提取 Task 1 样本的特征
3. **计算距离**：计算新旧特征之间的 Procrustes 距离
4. **分析结果**：
   - 距离接近 0：簇结构稳定，特征漂移小
   - 距离较大：簇结构改变，特征漂移大

### 输出结果

- **日志输出**：每个任务、每个类别的 Procrustes 距离
- **CSV 文件**：`logs/{model_name}/{dataset}/{init_cls}/{increment}/procrustes_distances.csv`

## 4. 配置参数

在 JSON 配置文件中启用：
```json
{
    "analyze_cluster_structure_drift": true
}
```

## 5. 注意事项

1. **样本数量**：需要至少 2 个样本才能计算有意义的 Procrustes 距离
2. **特征维度**：适用于任意维度 D 的特征空间
3. **归一化**：最终距离是归一化的（0-1 范围），便于比较不同规模的数据
4. **CL-LoRA 特殊处理**：只使用 general_lora，排除 specific_lora 的影响

