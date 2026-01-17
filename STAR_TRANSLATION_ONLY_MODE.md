# STAR 平移模式说明

## 修改概述

STAR 方法已修改为**只保留平移，去掉旋转和缩放**。

## 修改内容

### 1. `compute_rigid_transform()` 方法

**修改位置**：`utils/STAR.py` 第 39-120 行

**关键修改**：
- **旋转 R**：设置为 `None`（禁用旋转）
- **缩放 s**：设置为 `1.0`（禁用缩放）
- **保留计算逻辑**：仍然计算 R 和 s 的值（用于调试信息），但不使用

**变换公式**：
```
原始公式：W_new = s * (W_old - mu_old) @ R + mu_new
平移模式：W_new = (W_old - mu_old) + mu_new = W_old + (mu_new - mu_old)
```

### 2. 返回类型更新

**修改**：返回类型从 `Tuple[np.ndarray, np.ndarray, np.ndarray, float]` 
改为 `Tuple[Optional[np.ndarray], np.ndarray, np.ndarray, float]`

**原因**：R 现在可以是 `None`

### 3. 边界情况处理

**修改**：当样本数 < 2 时，返回 `(None, zeros, zeros, 1.0)` 而不是单位矩阵

## 变换公式验证

### 平移模式下的变换

当 `R=None` 且 `s=1.0` 时：

```python
# Step A: 去中心化
centers_centered = centers_raw - mu_old

# Step B: 旋转（跳过，因为 R=None）
centers_rotated = centers_centered  # 不变

# Step C: 缩放 & 平移
centers_new_raw = (centers_rotated * 1.0) + mu_new
                = centers_centered + mu_new
                = (centers_raw - mu_old) + mu_new
                = centers_raw + (mu_new - mu_old)  # 纯平移
```

**结果**：变换简化为纯平移：`W_new = W_old + (mu_new - mu_old)`

## 调试信息

修改后的日志输出：

```
[STAR DEBUG] Drift Analysis (Translation Only Mode):
  > Shift (Mean Move): 0.123456 (Avg Norm: 15.23)
  > Scale Change (computed but disabled): 1.023456
  > Rotation Angle (computed but disabled): 0.012345
  > Using: Translation only (R=None, s=1.0)
```

**说明**：
- `Scale Change` 和 `Rotation Angle` 仍然计算（用于分析），但不使用
- 实际使用的是纯平移变换

## 兼容性

### `apply_rigid_transform()` 支持

`utils/hc_soinn_classifier.py` 中的 `apply_rigid_transform()` 已经支持 `R=None`：

```python
# Step B: 旋转
if R is not None:
    centers_rotated = np.dot(centers_centered, R)
else:
    centers_rotated = centers_centered  # 跳过旋转
```

**结论**：无需修改 `apply_rigid_transform()`，代码已兼容。

## 使用效果

### 预期行为

1. **只进行平移对齐**：节点位置按照中心点的偏移量进行平移
2. **不进行旋转**：节点之间的相对角度保持不变
3. **不进行缩放**：节点之间的相对距离保持不变

### 适用场景

- 特征漂移主要是平移（中心点偏移）
- 不需要处理旋转和缩放的情况
- 简化对齐过程，减少计算复杂度

## 恢复完整变换

如果需要恢复旋转和缩放，只需修改 `utils/STAR.py` 第 93-97 行：

```python
# 恢复完整变换
R = R_computed  # 使用计算的旋转矩阵
s = s_computed  # 使用计算的缩放因子
```

---

**修改日期**：2025-12-27  
**修改状态**：✅ 已完成

