# STAR 使用指南

## 概述

STAR (Structure-Topology Alignment via Residuals) 是一个特征漂移对齐算法，专为类增量学习中的特征空间漂移问题设计。

**核心哲学**：适应而非对抗
- 不通过蒸馏限制 Backbone
- 通过几何变换让分类器主动对齐新空间
- 实现"结构追随"而非"结构对抗"

---

## 快速开始

### 1. 配置文件设置

在实验配置 JSON 文件中添加以下参数：

```json
{
    "use_hc_soinn": true,
    "use_feature_alignment": true,
    "use_full_task_rehearsal": false
}
```

**参数说明**：
- `use_hc_soinn`: 必须为 `true`（STAR 需要 HC-SOINN）
- `use_feature_alignment`: 启用 STAR 对齐
- `use_full_task_rehearsal`: 
  - `false`（推荐）：锚点模式，只保存 SOINN 节点和 NCM 中心对应的样本
  - `true`：全量模式，保存所有训练样本（用于性能上限测试）

### 2. 代码集成

STAR 已集成到以下模型中：
- ✅ `models/coda_prompt.py`
- ⚠️ 其他模型（如 `cllora.py`, `sema.py`）需要类似集成

### 3. 运行实验

正常运行训练脚本即可，STAR 会自动在 `after_task()` 中执行对齐和锚点选择。

---

## 工作原理

### 核心流程

```
Task 0 (Class 1-10):
  1. 正常训练
  2. compress() 生成原型
  3. 选择锚点（SOINN 节点 + NCM 中心）
  4. 保存锚点图像和特征 F_0
  5. 将原型存入 class_clusters_original

Task 1 (Class 11-20):
  1. 对齐旧类（Class 1-10）：
     - 加载锚点图像
     - 用新模型提取特征 F_1
     - 计算变换 (R, mu_old, mu_new)
     - 应用变换到 SOINN 节点（从原始节点开始）
     - 链式覆盖：更新 feats_ref = F_1
  2. 训练新类（Class 11-20）
  3. compress() 生成新类原型
  4. 选择新类锚点
  5. 保存新类锚点图像和特征 F_1

Task 2 (Class 21-30):
  1. 对齐旧类（Class 1-20）：
     - Class 1-10: 使用 F_1 -> F_2 对齐
     - Class 11-20: 使用 F_1 -> F_2 对齐
  2. ...
```

### Plan B（原始骨架还原）

**关键设计**：始终从 `class_clusters_original` 开始变换，避免累积误差。

```python
# 错误方式（累积误差）
Task 1: F_0 -> F_1, 变换节点
Task 2: F_1 -> F_2, 在已变换的节点上再次变换 ❌

# 正确方式（Plan B）
Task 1: F_0 -> F_1, 从原始节点变换
Task 2: F_1 -> F_2, 从原始节点变换 ✅
```

### 全量拓扑映射

**锚点选择策略**：
1. 确定靶心：所有 SOINN 节点的 `center_raw` + NCM 中心的 `class_mu_raw`
2. 最近邻匹配：在训练数据中找到与靶心余弦距离最近的样本
3. 持久化：保存样本图像和当前模型下的特征

**优势**：
- 完整捕捉流形漂移
- 锚点位于数据流形的"关节"位置
- 数量可控（通常 10-25 个/类）

---

## API 参考

### STARAligner 类

#### `__init__(hc_soinn, feature_extractor, device, use_full_task_rehearsal)`

初始化 STAR 对齐器。

**参数**：
- `hc_soinn`: HC-SOINN 分类器实例
- `feature_extractor`: 特征提取函数 `(Tensor) -> Tensor`
- `device`: 计算设备
- `use_full_task_rehearsal`: 是否使用全量模式

#### `compute_rigid_transform(feats_old, feats_new) -> (R, mu_old, mu_new)`

计算正交 Procrustes 变换。

**参数**：
- `feats_old`: 旧特征 [M, D]
- `feats_new`: 新特征 [M, D]

**返回**：
- `R`: 旋转矩阵 [D, D]
- `mu_old`: 旧空间中心 [D]
- `mu_new`: 新空间中心 [D]

#### `select_anchors_for_current_task(dataset, batch_size, num_workers, current_task_classes)`

为当前任务选择锚点。

**参数**：
- `dataset`: 训练数据集
- `batch_size`: 批处理大小
- `num_workers`: 数据加载器工作进程数
- `current_task_classes`: 当前任务的类别集合

#### `align_old_classes(cur_task, current_task_classes)`

对齐所有旧类别的 SOINN 节点。

**参数**：
- `cur_task`: 当前任务编号
- `current_task_classes`: 当前任务的新类别集合（这些类别不需要对齐）

---

## 调试和监控

### 1. 日志输出

STAR 提供详细的日志输出：

```
[STAR] Initialized (Plan B: Re-alignment from Original Nodes)
[STAR] Mode: Anchor-based (saving SOINN nodes + NCM centers)
[STAR] Class 0: Selected 15 anchors (from 18 target points, 500 samples)
[STAR] Procrustes alignment: error_before=0.123456, error_after=0.012345, reduction=90.00%
[STAR] Task 1: Aligned 10 classes, skipped 0 classes
```

### 2. 获取锚点信息

```python
anchor_info = star.get_anchor_info(cls=0)
print(anchor_info)
# {'num_anchors': 15, 'feat_dim': 768, 'image_shape': (3, 224, 224)}
```

### 3. 清除锚点（用于重置）

```python
star.clear_anchors(class_list=[0, 1, 2])  # 清除指定类别
star.clear_anchors()  # 清除所有
```

---

## 性能优化

### 1. 批处理大小

建议 `batch_size=128`，可根据 GPU 内存调整。

### 2. 多进程加载

建议 `num_workers=4-8`，可根据 CPU 核心数调整。

### 3. 全量模式 vs 锚点模式

- **锚点模式**（推荐）：内存占用小，速度快
- **全量模式**：性能上限测试，内存占用大

---

## 常见问题

### Q1: 对齐误差很大怎么办？

**A**: 检查以下几点：
1. 特征提取函数是否正确（是否使用了正确的模型状态）
2. 锚点数量是否足够（建议每个类别至少 10 个）
3. 特征维度是否一致

### Q2: 内存占用过大怎么办？

**A**: 
1. 使用锚点模式（`use_full_task_rehearsal=false`）
2. 减少 `max_prototypes_per_class`
3. 减少锚点数量（修改 `select_anchors_for_current_task` 中的去重逻辑）

### Q3: 对齐后性能反而下降？

**A**: 可能原因：
1. 特征漂移不是刚性的（Procrustes 距离 > 0.1）
2. 锚点选择不当（样本质量差）
3. 变换计算有误（检查维度一致性）

### Q4: 如何验证 Plan B 是否正确工作？

**A**: 检查日志：
```
[STAR] Class X: Re-aligning from ORIGINAL nodes (avoiding cumulative transformation errors)
```
如果看到这条日志，说明 Plan B 已启用。

---

## 实现检查清单

参考 `STAR_IMPLEMENTATION_CHECKLIST.md` 验证实现是否正确。

**关键检查项**：
- ✅ Plan B 逻辑
- ✅ 全量锚点选择
- ✅ 链式覆盖
- ✅ 归一化顺序
- ✅ 推理时机

---

## 参考文献

- Procrustes Analysis
- Orthogonal Procrustes Problem
- Structure-Topology Alignment via Residuals (STAR)

---

## 更新日志

### v1.0 (当前版本)
- ✅ 实现 Plan B（原始骨架还原）
- ✅ 实现全量拓扑映射锚点选择
- ✅ 实现链式覆盖
- ✅ 集成到 coda_prompt.py
- ✅ 完整的调试和监控功能

