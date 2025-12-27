# STAR 性能断崖式下跌 Bug 修复

## 问题描述

在 Task 2 时，HC-SOINN 性能从 96.96% 断崖式下跌到 83.43%，下降了约 13.5 个百分点。

## 根本原因

**对齐时机错误**：STAR 对齐在评估之后执行，导致评估时使用的是未对齐的节点。

### 原始调用顺序（错误）

```
trainer.py:
  for task in range(nb_tasks):
    model.incremental_train(data_manager)  # 训练
    eval_results = model.eval_task()        # 评估 ← 使用未对齐的节点
    model.after_task()                      # 对齐旧类别 ← 太晚了！
```

**问题分析**：
- Task 0 结束后：选择锚点（但没有旧类别需要对齐）
- Task 1 训练完成后：
  - 立即评估（此时 Task 0 的节点还未对齐）
  - 评估完成后，才执行 `after_task()` 对齐 Task 0 的节点
- Task 2 训练完成后：
  - 立即评估（此时 Task 0-1 的节点还未对齐）← **性能断崖式下跌**
  - 评估完成后，才执行 `after_task()` 对齐 Task 0-1 的节点

## 修复方案

**在评估前对齐**：将 STAR 对齐移到 `_build_classifiers()` 中，在评估之前执行。

### 修复后的调用顺序（正确）

```
trainer.py:
  for task in range(nb_tasks):
    model.incremental_train(data_manager)
      └─> _build_classifiers()
          └─> STAR 对齐旧类别 ← 在评估前对齐
    eval_results = model.eval_task()        # 评估 ← 使用对齐后的节点 ✅
    model.after_task()
      └─> 压缩新类别
      └─> 选择锚点
```

## 代码修改

### 1. `_build_classifiers()` 中添加对齐

```python
def _build_classifiers(self):
    # ========== Step 0: STAR 特征漂移对齐（在评估前对齐旧类别）==========
    # 关键修复：在评估之前对齐旧类别，确保评估时使用的是对齐后的节点
    if self.star is not None and self._cur_task > 0:
        current_task_classes = set(range(self._known_classes, self._total_classes))
        self.star.align_old_classes(
            cur_task=self._cur_task,
            current_task_classes=current_task_classes
        )
    
    # 然后构建其他分类器...
```

### 2. `after_task()` 中移除重复对齐

```python
def after_task(self):
    # ========== Step 1: 特征漂移对齐（针对旧类别）==========
    # 注意：对齐已在 _build_classifiers() 中提前执行（在评估前）
    # 这里不再重复对齐，只处理新类别的压缩和锚点选择
    
    # ========== Step 2: 压缩 HC-SOINN（生成当前任务的节点）==========
    if self.use_hc_soinn:
        self.hc_soinn.compress()
    
    # ========== Step 3: 为当前任务选择锚点（用于下一轮对齐）==========
    if self.star is not None:
        self.star.select_anchors_for_current_task(...)
```

## 修复效果

修复后，每个任务的评估流程：

1. **Task 0**：
   - 训练 → 构建分类器（无旧类别需要对齐）→ 评估 → 压缩 → 选择锚点

2. **Task 1**：
   - 训练 → 构建分类器（对齐 Task 0 的类别）→ 评估（使用对齐后的节点）→ 压缩 → 选择锚点

3. **Task 2**：
   - 训练 → 构建分类器（对齐 Task 0-1 的类别）→ 评估（使用对齐后的节点）✅ → 压缩 → 选择锚点

## 验证方法

1. **检查日志**：应该看到 `[STAR] Pre-evaluation alignment: Aligning old classes before evaluation`
2. **性能曲线**：Task 2 及之后的性能应该稳定，不再出现断崖式下跌
3. **对齐误差**：检查 `[STAR] Procrustes alignment: error_before=..., error_after=...` 日志

## 相关文件

- `models/coda_prompt.py`：修复对齐时机
- `utils/STAR.py`：对齐逻辑（无需修改）

## 注意事项

1. **对齐时机**：必须在评估前对齐，否则评估结果不准确
2. **避免重复对齐**：`after_task()` 中不再重复对齐
3. **Task 0 特殊处理**：Task 0 没有旧类别，不需要对齐

---

**修复日期**：2025-12-27
**修复状态**：✅ 已完成

