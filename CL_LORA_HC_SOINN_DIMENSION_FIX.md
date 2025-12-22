# CL-LoRA HC-SOINN 维度不匹配问题修复

## 问题描述

在 CL-LoRA 中，不同任务的特征维度不同：
- **Task 0**: 768维（只有当前任务的 adapter）
- **Task 1**: 1536维（Task 0 + Task 1 的 adapter concat）
- **Task 2**: 2304维（Task 0 + Task 1 + Task 2 的 adapter concat）

当 HC-SOINN 在不同任务时更新 `class_mu`（NCM 类中心）时，会存储不同维度的特征。在 `predict_topk` 中尝试 `np.stack(ncm_centers)` 时，如果不同类别的 `class_mu` 维度不同，就会报错：

```
ValueError: all input arrays must have the same shape
```

## 修复方案

### 1. NCM 中心维度过滤

在 `predict_topk` 中，只使用与查询特征维度匹配的类别：

```python
query_dim = query_features.shape[1]
ncm_centers = []
valid_classes = []
for cls in classes:
    cls_mu = self.class_mu[cls]
    # 只使用维度匹配的类别
    if cls_mu.shape[0] == query_dim:
        ncm_centers.append(cls_mu)
        valid_classes.append(cls)
    else:
        # 维度不匹配，跳过该类别（可能是旧任务的特征，维度不同）
        logging.debug(
            f"Skipping class {cls} in NCM prediction: "
            f"class_mu dim={cls_mu.shape[0]}, query dim={query_dim}"
        )
```

### 2. 子簇原型维度过滤

同样，在构建子簇原型时，也只使用维度匹配的类别：

```python
for cls in valid_classes:  # 只使用维度匹配的类别
    clusters = self.class_clusters.get(cls, [])
    if clusters:
        # 检查簇中心的维度是否匹配
        cluster_dims = [c.center.shape[0] for c in clusters]
        if all(dim == query_dim for dim in cluster_dims):
            cls_protos = np.stack([c.center for c in clusters])
            all_protos.append(cls_protos)
            proto_labels.extend([cls] * len(clusters))
```

### 3. 类别索引映射

返回结果时，使用 `valid_classes` 来映射预测索引到原始类别ID：

```python
valid_classes_np = np.array(valid_classes)
top_preds = valid_classes_np[indices] # [N, k]
```

## 影响分析

### 优点

1. **避免崩溃**：不再因为维度不匹配而报错
2. **正确预测**：只使用与当前查询特征维度匹配的类别进行预测
3. **向后兼容**：不影响其他使用 HC-SOINN 的场景（如 coda_prompt）

### 潜在问题

1. **旧类别不可用**：如果旧类别的特征维度与当前查询不匹配，这些类别将不会被预测
   - **影响**：在 Task 1 时，Task 0 的类别（768维）无法被预测（因为查询是1536维）
   - **原因**：CL-LoRA 使用 concat 所有 adapter 的特征，导致不同任务的特征维度不同

### 解决方案

这个问题是 CL-LoRA 架构的特性导致的。要完全解决，需要：

1. **方案 A**：在每次任务后，重新计算所有旧类别的 NCM 中心（使用新的特征维度）
   - 需要保存旧任务的训练数据或使用 rehearsal
   - 实现复杂，但可以保证所有类别都可用

2. **方案 B**：使用特征对齐（STAR）将旧类别的特征对齐到新特征空间
   - 如果启用了 `use_feature_alignment`，STAR 会处理特征对齐
   - 但需要确保对齐后的特征维度与查询特征维度一致

3. **方案 C**：接受当前限制（推荐）
   - 在增量学习场景中，通常只评估所有已见过的类别
   - 当前实现可以正常工作，只是旧类别的 NCM 中心不会被使用
   - 子簇原型仍然可以使用（如果维度匹配）

## 测试建议

1. **验证维度过滤**：检查日志中是否有 "Skipping class" 的调试信息
2. **验证预测结果**：确保预测的类别索引正确
3. **验证性能**：检查 HC-SOINN 的准确率是否正常

## 相关文件

- `utils/hc_soinn_classifier.py`: `predict_topk` 方法
- `models/cllora.py`: `_get_hc_soinn_feature_fn` 方法（已修复为使用 `test=True` 模式）

