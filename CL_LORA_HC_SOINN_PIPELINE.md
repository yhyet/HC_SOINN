# CL-LoRA + HC-SOINN 完整 Pipeline 文档

## 1. 发现的 Bug 及修复

### Bug 1: 特征提取模式不一致（已修复）⚠️ **关键Bug**

**问题描述**：
- CL-LoRA 的 `backbone.forward()` 在训练和测试模式下返回的特征不同：
  - **训练模式** (`test=False`)：只返回当前任务的 adapter 特征（768维）
  - **测试模式** (`test=True`)：返回所有 adapter 的 concat 特征（768 * (task_num+1)维）
- 原始实现中，`_get_hc_soinn_feature_fn()` 使用 `extract_vector()`，它调用 `backbone(x)` 时默认 `test=False`
- 这导致：
  - **训练时**：提取的特征是 768维（只有当前任务的 adapter）
  - **测试时**：HC-SOINN 期望的特征是 768 * (task_num+1)维（所有 adapter 的 concat）
  - **结果**：特征维度不匹配，导致 HC-SOINN 性能异常低

**修复方案**：
```python
def _get_hc_soinn_feature_fn(self):
    def feature_fn(x):
        # 使用 test=True 模式，提取所有 adapter 的 concat 特征
        feats = backbone(x, test=True, use_init_ptm=self.use_init_ptm)
        return feats
    return feature_fn
```

**修复位置**：`models/cllora.py` 的 `_get_hc_soinn_feature_fn()` 方法

---

## 2. 完整 Pipeline 流程

### 2.1 初始化阶段 (`__init__`)

```python
# 1. 初始化 HC-SOINN 分类器
if self.use_hc_soinn:
    self.hc_soinn = HCSOINNClassifier(...)

# 2. 初始化 STAR 对齐器（如果启用）
if self.use_feature_alignment and self.use_hc_soinn:
    self.star = STARAlignment(...)

# 3. 初始化簇结构分析器（如果启用）
if self.analyze_cluster_structure_drift:
    self.cluster_analyzer = ClusterStructureAnalyzer(...)
```

### 2.2 训练阶段 (`incremental_train`)

**调用顺序**：
```
trainer.py:
  for task in range(nb_tasks):
    model.incremental_train(data_manager)  # 1. 训练
    eval_results = model.eval_task()       # 2. 评估
    model.after_task()                     # 3. 后处理
```

**`incremental_train` 内部流程**：

```python
def incremental_train(self, data_manager):
    # Step 1: 更新任务信息
    self._cur_task += 1
    self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
    self._network.update_fc(self._total_classes)
    
    # Step 2: 准备数据
    self.train_dataset = data_manager.get_dataset(...)
    self.train_loader = DataLoader(...)
    self.test_loader = DataLoader(...)
    
    # Step 3: 训练模型
    self._train(self.train_loader, self.test_loader)
    
    # Step 4: 更新 FC 层（使用原型网络）
    self._network.add_fc()
    self.replace_fc(self.train_loader_for_protonet)
    
    # Step 5: 构建 HC-SOINN bank（关键步骤）
    if self.use_hc_soinn:
        self._build_hc_soinn_bank()
        # 注意：这里只添加特征，不压缩
        # compress() 在 after_task() 中调用
```

**`_build_hc_soinn_bank` 详细流程**：

```python
def _build_hc_soinn_bank(self):
    # 1. 获取特征提取函数（使用 test=True 模式）
    feature_fn = self._get_hc_soinn_feature_fn()
    
    # 2. 提取当前任务新类别的特征
    current_task_dataset = data_manager.get_dataset(
        np.arange(self._known_classes, self._total_classes),
        source="train", mode="test"  # 使用 test 模式关闭数据增强
    )
    
    # 3. 批量提取特征并添加到 HC-SOINN
    for inputs, targets in current_task_loader:
        feats = feature_fn(inputs)  # 使用 test=True 模式
        self.hc_soinn.add_features(feats, targets)
    
    # 注意：这里不调用 compress()，因为：
    # - compress() 需要在 after_task() 中调用
    # - 这样可以确保在压缩前完成所有后处理（如 STAR 对齐）
```

### 2.3 评估阶段 (`eval_task`)

```python
def eval_task(self):
    results = {}
    
    # 1. FC 分类器评估
    y_pred, y_true = self._eval_cnn(self.test_loader)
    results["fc"] = self._evaluate(y_pred, y_true)
    
    # 2. HC-SOINN 分类器评估（如果启用）
    if self.use_hc_soinn:
        y_pred_hc, y_true_hc = self._eval_hc_soinn(self.test_loader)
        results["hc_soinn"] = self._evaluate(y_pred_hc, y_true_hc)
    
    return results
```

**`_eval_hc_soinn` 详细流程**：

```python
def _eval_hc_soinn(self, loader):
    self._network.eval()  # 确保模型处于 eval 模式
    feature_fn = self._get_hc_soinn_feature_fn()  # 使用 test=True 模式
    
    with torch.no_grad():
        for inputs, targets in loader:
            # 提取特征（使用 test=True 模式，所有 adapter 的 concat）
            feats = feature_fn(inputs)
            
            # 使用 HC-SOINN 预测
            topk_pred = self.hc_soinn.predict_topk(
                feats, self.topk, self._total_classes
            )
    
    return y_pred, y_true
```

### 2.4 后处理阶段 (`after_task`)

**调用顺序**（在 `trainer.py` 中）：
```python
model.incremental_train(data_manager)  # 训练
eval_results = model.eval_task()       # 评估
model.after_task()                     # 后处理 ← 这里
```

**`after_task` 详细流程**：

```python
def after_task(self):
    # ========== Step 1: STAR 特征漂移对齐 ==========
    # 目的：将旧类别的 SOINN 节点从旧模型空间对齐到新模型空间
    if self.star is not None:
        self.star.align_old_classes(self._cur_task)
    
    # ========== Step 2: 压缩 HC-SOINN ==========
    # 目的：为当前任务的新类别生成 SOINN 原型节点
    # 注意：这里压缩的是在 _build_hc_soinn_bank 中添加的特征
    if self.use_hc_soinn:
        self.hc_soinn.compress()
    
    # ========== Step 3: 选择锚点（用于下一轮对齐）==========
    if self.star is not None:
        self.star.select_anchors_for_current_task(...)
    
    # ========== Step 4: 簇结构分析 ==========
    if self.analyze_cluster_structure_drift:
        if self._cur_task == 0:
            # 保存 Task 1 的样本（用于后续计算 Procrustes 距离）
            self.cluster_analyzer.save_task1_samples(...)
        else:
            # 计算 Procrustes 距离
            self.cluster_analyzer.compute_procrustes_distances(self._cur_task)
    
    # ========== Step 5: 更新已知类别数 ==========
    self._known_classes = self._total_classes
```

---

## 3. 关键设计决策

### 3.1 为什么在 `_build_hc_soinn_bank` 中不调用 `compress()`？

**原因**：
1. `compress()` 需要在所有后处理完成后调用
2. STAR 对齐需要在压缩前完成（对齐旧类别的节点）
3. 压缩应该在 `after_task()` 中统一处理

**流程**：
```
incremental_train():
  _build_hc_soinn_bank()  # 添加特征，不压缩
  
after_task():
  star.align_old_classes()  # 对齐旧类别
  hc_soinn.compress()       # 压缩当前任务的特征
  star.select_anchors()     # 选择锚点
```

### 3.2 为什么特征提取使用 `test=True` 模式？

**原因**：
1. **一致性**：训练和测试时使用相同的特征空间
2. **完整性**：测试模式返回所有 adapter 的 concat 特征，包含所有任务的信息
3. **正确性**：HC-SOINN 需要完整的特征表示来进行分类

**对比**：
- **训练模式** (`test=False`)：只返回当前任务的 adapter 特征（768维）
- **测试模式** (`test=True`)：返回所有 adapter 的 concat 特征（768 * (task_num+1)维）

### 3.3 为什么使用 `mode="test"` 的数据集？

**原因**：
1. **关闭数据增强**：数据增强会让 SOINN 的节点变得更多更杂乱
2. **特征一致性**：确保提取的特征与推理时一致
3. **性能优化**：减少不必要的节点数量

---

## 4. 特征维度变化

### 4.1 CL-LoRA 的特征维度

**Task 0**：
- 训练模式：768维（只有当前任务的 adapter）
- 测试模式：768维（只有当前任务的 adapter）

**Task 1**：
- 训练模式：768维（只有当前任务的 adapter）
- 测试模式：1536维（Task 0 + Task 1 的 adapter concat）

**Task 2**：
- 训练模式：768维（只有当前任务的 adapter）
- 测试模式：2304维（Task 0 + Task 1 + Task 2 的 adapter concat）

**HC-SOINN 使用**：测试模式的特征（所有 adapter 的 concat）

---

## 5. 调试建议

### 5.1 检查特征维度

```python
# 在 _build_hc_soinn_bank 中添加日志
logging.info(f"Feature shape: {feats.shape}")
logging.info(f"Expected feature dim: {self._network.feature_dim}")
```

### 5.2 检查 HC-SOINN 状态

```python
# 在 after_task 中添加日志
if self.use_hc_soinn:
    logging.info(f"HC-SOINN class clusters: {len(self.hc_soinn.class_clusters)}")
    for cls_id, clusters in self.hc_soinn.class_clusters.items():
        logging.info(f"  Class {cls_id}: {len(clusters)} clusters")
```

### 5.3 验证特征提取模式

```python
# 在 _get_hc_soinn_feature_fn 中添加日志
def feature_fn(x):
    feats = backbone(x, test=True, use_init_ptm=self.use_init_ptm)
    logging.info(f"Extracted feature shape: {feats.shape}, cur_task: {self._cur_task}")
    return feats
```

---

## 6. 与 coda_prompt 的对比

### 6.1 特征提取方式

**coda_prompt**：
```python
def _get_soinn_feature_fn(self):
    def feature_fn(x):
        feats = self._network(x, pen=True, train=False)
        return feats
    return feature_fn
```

**cl_lora（修复后）**：
```python
def _get_hc_soinn_feature_fn(self):
    def feature_fn(x):
        feats = backbone(x, test=True, use_init_ptm=self.use_init_ptm)
        return feats
    return feature_fn
```

### 6.2 主要差异

1. **coda_prompt**：使用 `_network(x, pen=True, train=False)` 提取特征
2. **cl_lora**：使用 `backbone(x, test=True, use_init_ptm=self.use_init_ptm)` 提取特征
3. **关键**：两者都确保在测试模式下提取特征，保持一致性

---

## 7. 总结

### 7.1 修复的关键 Bug

1. ✅ **特征提取模式不一致**：修复了 `_get_hc_soinn_feature_fn()` 使用 `test=True` 模式
2. ✅ **特征维度不匹配**：确保训练和测试时使用相同的特征空间

### 7.2 完整的 Pipeline

1. **初始化**：创建 HC-SOINN、STAR、ClusterAnalyzer
2. **训练**：训练模型 → 构建 HC-SOINN bank（添加特征）
3. **评估**：使用 FC 和 HC-SOINN 分类器评估
4. **后处理**：STAR 对齐 → HC-SOINN 压缩 → 选择锚点 → 簇结构分析

### 7.3 关键要点

1. **特征提取**：始终使用 `test=True` 模式，确保特征维度一致
2. **数据模式**：使用 `mode="test"` 关闭数据增强
3. **压缩时机**：在 `after_task()` 中统一处理，确保所有后处理完成后再压缩

