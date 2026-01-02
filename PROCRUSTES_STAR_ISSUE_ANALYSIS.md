# Procrustes 距离计算与 STAR 方法问题分析报告

## 问题概述

通过对比 APER Adapter 和 CodaPrompt 两种方法的实验日志，发现了以下关键问题：

### 1. APER Adapter 方法：Procrustes 距离为 0（严重问题）

**现象**：
- 所有任务的 Procrustes 距离都是 `0.0000`
- 特征差异极小：`diff=0.00001287`，`mean_diff=0.00000000`
- 所有 Angle 都是 `3.1416`（π）
- STAR 判断：`Feature drift is NEGLIGIBLE! STAR will have no effect.`

**示例数据（Task 2）**：
```
Class 0: feats_task1[0,0]=0.61738116, feats_current[0,0]=0.61739403, diff=0.00001287
         Dist=0.0000, Scale=1.0000, Angle=3.1416
Class 1: feats_task1[0,0]=-1.18286765, feats_current[0,0]=-1.18287706, diff=0.00000942
         Dist=0.0000, Scale=1.0000, Angle=3.1416
```

**根本原因分析**：

1. **特征几乎完全相同**：
   - 特征差异在 `10^-5` 量级，这在数值上几乎可以忽略
   - 说明 `feature_extractor` 提取的特征与 Task 1 时几乎相同
   - **可能原因**：
     - Adapter 在测试模式下没有正确使用所有 adapter
     - 特征提取时使用了错误的模型状态
     - 存在某种缓存机制导致返回旧特征

2. **Procrustes 距离计算问题**：
   - 当 `feats_task1` 和 `feats_current` 几乎相同时，Procrustes 距离被归一化因子淹没
   - 归一化因子：`norm_factor = sqrt(sum(X2_centered^2))`
   - 如果特征范数很大（~130），即使有微小差异，归一化后的距离也会接近 0

3. **Angle = π 的异常**：
   - 当数据几乎相同时，SVD 可能产生反射矩阵（`det(R) < 0`）
   - 代码修正：`Vt[-1, :] *= -1`，这会导致角度计算异常
   - 角度估算公式：`cos(theta) = (Tr(R) - (D-2)) / 2` 在高维空间中可能不准确

### 2. CodaPrompt 方法：Procrustes 距离正常，但 Angle 异常

**现象**：
- Procrustes 距离正常：`0.15-0.39` 之间
- 特征差异明显：`diff=1.43258333`，`mean_diff=0.00013736`
- **但所有 Angle 都是 `3.1416`（π）**（异常！）

**示例数据（Task 2）**：
```
Class 0: feats_task1[0,0]=1.20314598, feats_current[0,0]=-0.22943740, diff=1.43258333
         Dist=0.1638, Scale=0.9867, Angle=3.1416
Class 1: feats_task1[0,0]=-4.40682030, feats_current[0,0]=-4.05566883, diff=0.35115147
         Dist=0.1587, Scale=0.9901, Angle=3.1416
```

**问题分析**：

1. **Procrustes 距离计算正常**：
   - 距离在合理范围内（0.15-0.39），说明特征确实有漂移
   - Scale 接近 1.0，说明缩放变化很小

2. **Angle = π 的异常**：
   - 特征明显不同，但角度都是 π，这不符合预期
   - **可能原因**：
     - 角度估算公式在高维空间中不准确
     - SVD 产生的旋转矩阵 R 的行列式可能经常为负，触发修正
     - 修正后的 R 导致角度计算异常

### 3. STAR 方法：判断阈值可能过于严格

**STAR 判断逻辑**（`utils/STAR.py:523`）：
```python
if shift_dist < 0.1 and abs(s - 1.0) < 0.01:
    logging.warning("[STAR WARNING] Feature drift is NEGLIGIBLE! STAR will have no effect.")
```

**问题**：
- 对于 APER 方法，`shift_dist` 确实很小（~0.0001），所以判断正确
- 但对于 CodaPrompt 方法，虽然 Procrustes 距离正常，但 STAR 可能没有启用（日志中没有 STAR 相关信息）

## 修复建议

### 1. 修复 APER Adapter 的特征提取问题

**问题**：特征几乎完全相同，说明 `feature_extractor` 没有使用当前任务的模型权重

**检查点**：
1. 确认 `feature_extractor` 闭包正确捕获了 `self._network`
2. 确认在 `compute_procrustes_distances` 调用时，模型已经更新到当前任务
3. 检查 Adapter 在测试模式下是否正确使用所有 adapter

**建议修复**：
```python
# 在 compute_procrustes_distances 中添加模型状态检查
def compute_procrustes_distances(self, cur_task: int) -> None:
    # 确保模型处于 eval 模式
    if hasattr(self.feature_extractor, '__self__'):
        model = getattr(self.feature_extractor.__self__, '_network', None)
        if model is not None:
            model.eval()
    
    # 添加特征提取验证
    # 提取一个样本的特征，检查是否与 Task 1 时不同
    ...
```

### 2. 修复 Procrustes 距离计算的归一化问题

**问题**：当特征范数很大时，归一化因子可能过大，导致距离被淹没

**建议修复**：
```python
# 在 _compute_procrustes_distance 中
# 使用更合理的归一化方式
norm_factor = np.sqrt(np.sum(X2_centered**2)) + 1e-8
# 改为：
norm_factor = np.sqrt(np.mean(X2_centered**2)) + 1e-8  # 使用均值而非总和
# 或者：
norm_factor = np.linalg.norm(X2_centered, ord='fro') / np.sqrt(X2_centered.shape[0])  # 归一化到每个样本
```

### 3. 修复角度计算问题

**问题**：角度估算公式在高维空间中不准确，且 SVD 修正可能导致异常

**建议修复**：
```python
# 在 _compute_procrustes_distance 中
# 当前的角度估算：
trace_R = np.trace(R)
D = X1.shape[1]
cos_theta = (trace_R - (D - 2)) / 2.0
angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))

# 改为更准确的计算：
# 对于旋转矩阵，角度可以通过特征值计算
eigenvals = np.linalg.eigvals(R)
# 对于 2D 旋转，角度 = arctan2(imag(eigenval), real(eigenval))
# 对于高维，使用平均角度
angles = np.angle(eigenvals)
angle = np.mean(np.abs(angles))
```

### 4. 改进 STAR 的漂移判断

**问题**：判断阈值可能过于严格，且没有考虑旋转角度

**建议修复**：
```python
# 在 STAR 的 compute_rigid_transform 中
# 当前判断：
if shift_dist < 0.1 and abs(s - 1.0) < 0.01:
    logging.warning("...")

# 改为考虑旋转角度：
rotation_threshold = 0.01  # 1% 的旋转
if shift_dist < 0.1 and abs(s - 1.0) < 0.01 and theta < rotation_threshold:
    logging.warning("...")
```

## 总结

1. **APER Adapter 方法**：特征提取存在问题，导致特征几乎完全相同，Procrustes 距离为 0
2. **CodaPrompt 方法**：特征提取正常，但角度计算存在问题，所有角度都是 π
3. **STAR 方法**：判断阈值可能过于严格，且没有考虑旋转角度

**优先级**：
1. **高优先级**：修复 APER Adapter 的特征提取问题（这是最严重的问题）
2. **中优先级**：修复角度计算问题（影响诊断信息）
3. **低优先级**：改进 STAR 的漂移判断（影响较小）

