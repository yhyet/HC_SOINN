# HC-SOINN 分类器介绍

## 1. 概述

**HC-SOINN (Hierarchical-Cluster SOINN)** 是一个专为类增量学习设计的混合分类器，结合了层次聚类（Hierarchical Clustering）和自组织增量神经网络（Self-Organizing Incremental Neural Network, SOINN）的优势。

### 1.1 设计理念

HC-SOINN 采用**两阶段原型生成策略**：

1. **层次聚类阶段**：先对类内特征进行层次聚类，过滤噪声，得到代表性的簇中心
2. **SOINN 精炼阶段**：在簇中心上应用简化版 SOINN 自组织机制，进行增量学习和动态调整

这种设计既利用了层次聚类的去噪能力，又保留了 SOINN 的自适应特性，能够在类增量学习场景中高效地维护和更新类别原型。

### 1.2 核心优势

- **高效的原型管理**：通过层次聚类和 SOINN 精炼，自动维护每个类别的代表性原型节点
- **增量学习能力**：支持动态添加新类别，无需重新训练整个模型
- **GPU 加速推理**：使用 GPU 矩阵运算加速预测过程
- **多进程并行压缩**：使用多进程并行处理各类的聚类计算，提升训练效率
- **融合距离计算**：结合 NCM（Nearest Class Mean）和子簇中心，提供更准确的分类决策

---

## 2. 核心组件

### 2.1 数据结构

#### NCM 类中心（Nearest Class Mean）
- `class_mu`: 归一化的类中心（用于推理）
- `class_mu_raw`: 未归一化的类中心（用于特征对齐）
- `class_count`: 每个类别的样本计数

#### 类内子簇（Sub-clusters）
- `class_clusters`: 每个类别的 SOINN 节点列表
  - 每个节点包含：归一化中心、原始中心、样本计数
- `class_edges`: SOINN 节点之间的边关系（用于自组织网络）

#### 特征缓冲
- `buffers`: 临时存储当前任务的特征，等待压缩

### 2.2 主要方法

#### `add_features(features, labels)`
- **功能**：将当前任务的特征加入缓冲，并更新全局类中心（NCM）
- **使用场景**：训练过程中，每个 batch 的特征提取后调用

#### `compress()`
- **功能**：周期性压缩（通常在每个 task 结束时调用）
  - 对每个类将缓冲特征进行层次聚类
  - 可选：在簇中心上应用 SOINN 自组织机制
  - 使用多进程并行处理各类聚类
- **使用场景**：每个任务训练结束后调用

#### `predict_topk(query_features, topk, total_classes, device)`
- **功能**：返回 Top-K 类别预测
- **计算流程**：
  1. 计算 NCM 距离：`dist_ncm = 1 - cosine(query, class_mu)`
  2. 计算子簇距离：`dist_sub = min(1 - cosine(query, cluster_center))`
  3. 融合距离：`final_score = alpha * dist_ncm + (1 - alpha) * dist_sub`
  4. 返回 Top-K 类别
- **优化**：使用 GPU 矩阵运算加速

---

## 3. 工作流程

### 3.1 训练阶段

```
每个任务训练流程：
1. 提取特征 → add_features() → 更新缓冲和 NCM 中心
2. 任务结束 → compress() → 层次聚类 + SOINN 精炼 → 生成原型节点
```

**详细步骤**：

1. **特征收集**：
   - 训练过程中，每个 batch 的特征通过 `add_features()` 加入缓冲
   - 同时更新 NCM 类中心（增量均值）

2. **原型生成**（任务结束时）：
   - 调用 `compress()` 对每个类别的缓冲特征进行压缩
   - **层次聚类**：使用 scipy 的层次聚类算法，将特征聚类为 `max_prototypes_per_class` 个簇
   - **SOINN 精炼**（可选）：
     - 在簇中心上应用简化版 SOINN 自组织机制
     - 通过边老化、节点删除等机制，进一步优化原型节点
     - 保留拓扑结构信息（边关系）

### 3.2 推理阶段

```
查询特征 → predict_topk() → 融合距离计算 → Top-K 预测
```

**详细步骤**：

1. **特征归一化**：将查询特征归一化到单位超球面
2. **NCM 距离计算**：计算查询特征到每个类别中心的余弦距离
3. **子簇距离计算**：计算查询特征到每个类别所有子簇中心的最小距离
4. **融合距离**：`final_score = alpha * dist_ncm + (1 - alpha) * dist_sub`
5. **Top-K 选择**：返回距离最小的 Top-K 类别

---

## 4. 关键参数

### 4.1 基础参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_prototypes_per_class` | int | 20 | 每个类别的最大原型数量 |
| `alpha` | float | 0.5 | NCM 和子簇距离的融合权重（0-1） |
| `tau_merge` | float | 0.2 | 簇合并阈值（余弦距离） |
| `tau_reject` | float | 2.0 | 拒绝阈值（未使用） |
| `linkage_method` | str | "average" | 层次聚类链接方法 |
| `distance_metric` | str | "cosine" | 距离度量（cosine/euclidean） |

### 4.2 SOINN 精炼参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_soinn_refinement` | bool | True | 是否启用 SOINN 精炼 |
| `soinn_ad` | int | 20 | 边最大年龄（越小，边越容易被删除） |
| `soinn_lam` | int | 20 | 每 lam 次迭代删除孤立节点（已弃用） |
| `soinn_threshold_scale` | float | 0.5 | 阈值缩放因子 |
| `soinn_max_iter` | int | 3 | SOINN 最大迭代轮数 |
| `soinn_max_degree_for_removal` | int | 1 | 删除节点的最大度阈值 |

**参数调优建议**：

- **`alpha`**：控制 NCM 和子簇的权重
  - 接近 1.0：更依赖全局类中心（适合类别内分布较集中）
  - 接近 0.0：更依赖局部子簇（适合类别内分布较分散）
  - 0.5：平衡两种距离

- **`max_prototypes_per_class`**：控制原型数量
  - 较小值（10-15）：更紧凑，适合简单类别
  - 较大值（20-30）：更精细，适合复杂类别

- **`soinn_max_iter`**：控制 SOINN 精炼强度
  - 较小值（1-2）：快速收敛，但可能不够精细
  - 较大值（3-5）：更精细，但计算成本更高

---

## 5. SOINN 自组织机制

### 5.1 基本原理

SOINN（Self-Organizing Incremental Neural Network）是一种自组织神经网络，能够在增量学习过程中动态调整网络结构。

### 5.2 简化版 SOINN 流程

在 HC-SOINN 中，SOINN 机制应用于**簇中心**（而非原始特征），流程如下：

1. **初始化**：
   - 所有簇中心作为节点
   - 为每个节点找到最近的 1 个邻居建立连接

2. **自组织迭代**（多轮）：
   - 将簇中心作为"样本"输入
   - 找到 winner 和 second winner
   - 建立/更新边关系
   - 边老化：每条边的年龄 +1，超过 `soinn_ad` 则删除
   - 节点更新：使用球面线性插值（SLERP）更新节点位置
   - 删除孤立节点：度 ≤ `max_degree_for_removal` 的节点被删除

3. **最终清理**：
   - 删除所有孤立节点（度=0）
   - 确保图的连通性

### 5.3 关键机制

#### 边老化（Edge Aging）
- 每次迭代，winner 的边年龄 +1
- 年龄超过 `soinn_ad` 的边被删除
- 作用：自动删除不再重要的连接，保持网络结构紧凑

#### 节点删除（Node Removal）
- 删除度 ≤ `max_degree_for_removal` 的节点
- 作用：移除边缘节点，保留核心节点
- 默认 `max_degree_for_removal=1`：只删除孤立节点或尾部节点

#### 球面线性插值（SLERP）
- 在单位超球面上进行线性插值
- 保持节点归一化（模长=1）
- 作用：平滑更新节点位置，避免数值不稳定

---

## 6. 性能优化

### 6.1 GPU 加速推理

`predict_topk()` 使用 PyTorch 的 GPU 矩阵运算：

```python
# 批量计算余弦相似度
sim_ncm = torch.mm(query_t, ncm_centers_t.t())  # [N, C]
dist_ncm = 1.0 - sim_ncm

# 批量计算子簇距离
sim_proto = torch.mm(query_t, all_protos_t.t())  # [N, M]
dist_proto_all = 1.0 - sim_proto
```

**优势**：
- 避免 Python 循环，大幅提升推理速度
- 支持批量预测，充分利用 GPU 并行能力

### 6.2 多进程并行压缩

`compress()` 使用 `ProcessPoolExecutor` 并行处理各类聚类：

```python
with ProcessPoolExecutor() as executor:
    results = list(executor.map(_compress_class_worker, tasks))
```

**优势**：
- 充分利用多核 CPU，加速压缩过程
- 每个类别的聚类计算独立，适合并行化

---

## 7. 使用示例

### 7.1 初始化

```python
from utils.hc_soinn_classifier import HCSOINNClassifier

hc_soinn = HCSOINNClassifier(
    max_prototypes_per_class=20,
    alpha=0.5,
    tau_merge=0.2,
    linkage_method="average",
    distance_metric="cosine",
    use_soinn_refinement=True,
    soinn_ad=20,
    soinn_max_iter=3,
)
```

### 7.2 训练阶段

```python
# 每个 batch 后添加特征
for batch in train_loader:
    features = extract_features(batch)  # [B, D]
    labels = batch['labels']  # [B]
    hc_soinn.add_features(features, labels)

# 任务结束时压缩
hc_soinn.compress()
```

### 7.3 推理阶段

```python
# 批量预测
query_features = extract_features(test_batch)  # [N, D]
topk_preds = hc_soinn.predict_topk(
    query_features, 
    topk=5, 
    total_classes=100,
    device=torch.device("cuda")
)  # [N, 5]
```

---

## 8. 适用场景

### 8.1 类增量学习（Class-Incremental Learning）

HC-SOINN 专为类增量学习设计，特别适合：

- **持续学习场景**：需要不断学习新类别
- **内存受限场景**：不能存储所有训练样本
- **原型学习场景**：使用少量原型代表每个类别

### 8.2 支持的模型

HC-SOINN 已集成到以下模型中：

- **CL-LoRA** (`models/cllora.py`)
- **CoDA-Prompt** (`models/coda_prompt.py`)
- **SEMA** (`models/sema.py`)
- **SimpleCIL** (`models/simplecil_hc_soinn.py`)

### 8.3 数据集

已在以下数据集上验证：

- **CIFAR-100**：100 个类别，10 个任务，每个任务 10 个类别
- **ImageNet-R**：200 个类别，10 个任务，每个任务 20 个类别
- **CUB-200**：200 个类别，10 个任务，每个任务 20 个类别

---

## 9. 技术细节

### 9.1 归一化策略

- **特征归一化**：所有特征归一化到单位超球面（L2 归一化）
- **距离计算**：使用余弦距离（`1 - cosine_similarity`）
- **优势**：对特征尺度不敏感，更适合高维特征空间

### 9.2 层次聚类

- **算法**：使用 scipy 的 `linkage()` 和 `fcluster()`
- **链接方法**：默认 "average"（平均链接）
- **目标**：将特征聚类为 `max_prototypes_per_class` 个簇

### 9.3 融合距离公式

```
final_score = alpha * dist_ncm + (1 - alpha) * dist_sub

其中：
- dist_ncm: 查询特征到类别中心的余弦距离
- dist_sub: 查询特征到最近子簇中心的余弦距离
- alpha: 融合权重（默认 0.5）
```

**设计思路**：
- NCM 距离：捕获全局类别信息
- 子簇距离：捕获局部结构信息
- 融合：平衡全局和局部信息

---

## 10. 调试和监控

### 10.1 日志输出

HC-SOINN 提供详细的日志输出：

```
[HC-SOINN] Compressing 10 classes using multiprocessing...
[HC-SOINN] class 0: hierarchical_clusters=25 -> soinn_refined=18 (reduction: 7)
[HC-SOINN] class 1: hierarchical_clusters=23 -> soinn_refined=17 (reduction: 6)
...
```

### 10.2 快照功能

支持保存和比较不同任务的原型快照：

```python
# 保存快照
hc_soinn.save_cluster_snapshot(task_id=2, class_list=[0, 1, 2, 3, 4])

# 比较快照
hc_soinn.compare_cluster_snapshots(task_id1=1, task_id2=2, class_list=[0, 1, 2, 3, 4])
```

### 10.3 原型统计

```python
# 获取每个类别的原型数量
proto_counts = hc_soinn.prototypes_per_class()
# 输出: {0: 18, 1: 17, 2: 20, ...}
```

---

## 11. 常见问题

### Q1: 如何选择合适的 `alpha` 值？

**A**: 建议从 0.5 开始，根据类别内分布调整：
- 类别内分布集中 → 增大 `alpha`（接近 1.0）
- 类别内分布分散 → 减小 `alpha`（接近 0.0）

### Q2: SOINN 精炼是否必要？

**A**: 取决于类别复杂度：
- 简单类别：可以关闭（`use_soinn_refinement=False`），使用简单的距离阈值合并
- 复杂类别：建议开启，SOINN 精炼能更好地优化原型节点

### Q3: 如何平衡原型数量和性能？

**A**: 
- 原型数量越多，分类精度可能越高，但计算成本也越高
- 建议从 `max_prototypes_per_class=20` 开始，根据精度和速度需求调整

### Q4: 多进程压缩失败怎么办？

**A**: 
- 检查是否有足够的 CPU 核心
- 确保没有 GUI 相关的导入（matplotlib 会自动切换到非 GUI 后端）
- 如果仍有问题，可以修改代码使用单进程模式

---

## 12. 总结

HC-SOINN 是一个高效、灵活的类增量学习分类器，通过结合层次聚类和 SOINN 自组织机制，能够在类增量学习场景中自动维护和更新类别原型。其 GPU 加速推理和多进程并行压缩等优化，使其在实际应用中具有出色的性能和可扩展性。

**核心特点**：
- ✅ 两阶段原型生成（层次聚类 + SOINN 精炼）
- ✅ 融合距离计算（NCM + 子簇）
- ✅ GPU 加速推理
- ✅ 多进程并行压缩
- ✅ 增量学习支持
- ✅ 详细的日志和调试功能

**适用场景**：
- 类增量学习任务
- 内存受限场景
- 需要原型学习的场景

---

## 参考文献

- SOINN: Self-Organizing Incremental Neural Network
- Hierarchical Clustering
- Nearest Class Mean (NCM) Classifier
- Procrustes Analysis (用于特征对齐，已移除 STAR 相关代码)

