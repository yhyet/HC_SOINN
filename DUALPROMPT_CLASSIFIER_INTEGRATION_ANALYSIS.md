# DualPrompt 分类器集成分析文档

## 一、数据结构差异分析

### 1. 网络结构差异

#### DualPrompt (`PromptVitNet`)
- **网络类**: `PromptVitNet` (在 `utils/inc_net.py`)
- **Backbone**: `VisionTransformer` (在 `backbone/vit_dualprompt.py`)
- **Forward 返回值**: 字典 `{"logits": tensor, "pre_logits": tensor}`
- **分类头位置**: 集成在 backbone 内部 (`self.head`)，通过 `forward_head()` 方法
- **特征提取**: **没有独立的 `extract_vector()` 方法**
- **独立分类器**: **没有 `fc` 层或 `ncm_fc` 层**

```python
# dualprompt forward 结构
def forward(self, x, task_id=-1, train=False):
    res = self.forward_features(x, task_id=task_id, cls_features=cls_features, train=train)
    res = self.forward_head(res)  # 包含 logits 计算
    return res  # 返回 {"logits": ..., "pre_logits": ...}
```

#### CodaPrompt (`CodaPromptVitNet`)
- **网络类**: `CodaPromptVitNet` (在 `utils/inc_net.py`)
- **Backbone**: `VisionTransformer` (在 `backbone/vit_coda_promtpt.py`)
- **Forward 返回值**: 
  - `pen=False`: 返回 logits (tensor)
  - `pen=True`: 返回特征 (tensor)
  - `train=True`: 返回 (logits, prompt_loss) 元组
- **分类头位置**: **独立的 `self.fc` 层**（可以是 Linear 或 KACLayer）
- **特征提取**: **有 `extract_vector()` 方法**，用于 NCM 分类器
- **NCM 分类器**: **有独立的 `self.ncm_fc` 层**（CosineLinear）

```python
# codaprompt forward 结构
def forward(self, x, pen=False, train=False):
    # ... 提取特征 ...
    if not pen:
        out = self.fc(out)  # 独立的 fc 层
    if train:
        return out, prompt_loss
    else:
        return out

def extract_vector(self, x):
    # 专门用于提取特征向量
    # ...
    return out  # [B, D] 特征向量
```

### 2. Forward 方法签名差异

| 方法 | DualPrompt | CodaPrompt |
|------|-----------|------------|
| **Forward 参数** | `forward(x, task_id=-1, train=False)` | `forward(x, pen=False, train=False)` |
| **Forward 返回值** | 字典 `{"logits": ..., "pre_logits": ...}` | Tensor 或 (Tensor, Tensor) 元组 |
| **特征提取方法** | 无 | `extract_vector(x)` |
| **分类器评估** | `output["logits"]` | `network(x)` 或 `network(x, pen=True)` |

### 3. 分类器集成状态

#### CodaPrompt 已集成的分类器
1. **KAC 分类器**: 通过 `self.fc = KACLayer(...)` 替换标准 Linear
2. **NCM 分类器**: 通过 `self.ncm_fc = CosineLinear(...)` 独立层
3. **HC-SOINN 分类器**: 通过 `self.hc_soinn = HCSOINNClassifier(...)` 独立对象
4. **KNN 分类器**: 通过 `self.knn = KNNClassifier(...)` 独立对象

#### DualPrompt 当前状态
- **仅使用 backbone 内部的分类头** (`self.head`)
- **没有独立的分类器层**
- **没有特征提取方法**

## 二、关键差异点总结

### 1. 架构设计差异

| 特性 | DualPrompt | CodaPrompt |
|------|-----------|------------|
| **分类头位置** | 集成在 backbone 内部 | 独立于 backbone |
| **特征提取** | 需要从 `pre_logits` 获取 | 有专门的 `extract_vector()` 方法 |
| **分类器层** | 无独立层 | 有 `fc` 和 `ncm_fc` 独立层 |
| **Forward 接口** | 返回字典 | 返回 tensor 或元组 |

### 2. 训练流程差异

#### DualPrompt 训练流程
```python
# models/dualprompt.py _init_train()
output = self._network(inputs, task_id=self._cur_task, train=True)
logits = output["logits"][:, :self._total_classes]
logits[:, :self._known_classes] = float('-inf')
loss = F.cross_entropy(logits, targets.long())
```

#### CodaPrompt 训练流程
```python
# models/coda_prompt.py _init_train()
logits, prompt_loss = self._network(inputs, train=True)
logits = logits[:, :self._total_classes]
logits[:, :self._known_classes] = float('-inf')
loss_supervised = F.cross_entropy(logits, targets.long())
loss = loss_supervised + prompt_loss.sum()
```

### 3. 评估流程差异

#### DualPrompt 评估流程
```python
# models/dualprompt.py _eval_cnn()
outputs = self._network(inputs, task_id=self._cur_task)["logits"][:, :self._total_classes]
```

#### CodaPrompt 评估流程
```python
# models/coda_prompt.py _eval_fc()
outputs = self._network(inputs)[:, :eval_classes]  # 直接返回 logits

# models/coda_prompt.py _eval_ncm_fc()
features = self._network.extract_vector(inputs)  # 使用 extract_vector
ncm_output = self._network.ncm_fc(features)  # 使用独立的 ncm_fc
```

## 三、集成三个分类器到 DualPrompt 的关键问题

### 1. 需要添加的特征提取方法

**问题**: DualPrompt 的 `PromptVitNet` 没有 `extract_vector()` 方法

**解决方案**: 
- 需要在 `PromptVitNet` 中添加 `extract_vector()` 方法
- 从 backbone 的 forward 结果中提取 `pre_logits` 作为特征向量
- 参考 CodaPrompt 的实现，但需要适配 DualPrompt 的字典返回值

```python
# 建议的实现（在 PromptVitNet 类中）
def extract_vector(self, x):
    """提取特征向量，用于 NCM、KNN、HC-SOINN 分类器"""
    with torch.no_grad():
        if self.original_backbone is not None:
            cls_features = self.original_backbone(x)['pre_logits']
        else:
            cls_features = None
    
    res = self.backbone(x, task_id=-1, cls_features=cls_features, train=False)
    # 从字典中提取 pre_logits
    features = res['pre_logits']  # [B, D]
    return features
```

### 2. 需要添加独立的分类器层

**问题**: DualPrompt 的分类头在 backbone 内部，需要添加独立分类器层

**解决方案**:
- 添加 `ncm_fc` 层（CosineLinear）用于 NCM 分类器
- **不需要添加独立的 `fc` 层**（保持使用 backbone 内部的 `self.head`）
- 但要支持 KAC 分类器，可能需要修改 backbone 的分类头

**注意**: 
- DualPrompt 的 backbone 已经有 `self.head`，这是分类头
- 如果要用 KAC 替换，需要修改 backbone 的 `reset_classifier()` 方法
- 或者保持 backbone 不变，添加额外的 `fc` 层（但这会改变架构）

### 3. Forward 方法的兼容性

**问题**: DualPrompt 的 forward 返回字典，CodaPrompt 的分类器代码可能不兼容

**解决方案**:
- 在 DualPrompt 的 Learner 中，保持现有的 forward 调用方式
- 在评估分类器时，使用 `extract_vector()` 提取特征
- 确保所有分类器都使用统一的特征提取接口

### 4. NCM 分类器的集成

**需要添加**:
- `self.ncm_fc = CosineLinear(embed_dim, nb_classes, sigma=False)` 在 `PromptVitNet.__init__()`
- `_build_ncm_classifier()` 方法（参考 CodaPrompt）
- `_eval_ncm_fc()` 方法（使用 `extract_vector()` 提取特征）

**关键点**:
- DualPrompt 的 embed_dim 可以从 backbone 获取（通常是 768）
- 特征提取需要使用 `extract_vector()`，而不是直接从 forward 获取

### 5. HC-SOINN 分类器的集成

**需要添加**:
- `self.hc_soinn = HCSOINNClassifier(...)` 在 Learner 的 `__init__()`
- `_build_hc_soinn_bank()` 方法（参考 CodaPrompt）
- `_eval_hc_soinn()` 方法（使用特征提取函数）

**关键点**:
- 特征提取函数需要从 `extract_vector()` 获取特征
- 需要适配 DualPrompt 的 forward 接口（字典返回值）

### 6. KAC 分类器的集成

**问题**: DualPrompt 的分类头在 backbone 内部，直接修改可能影响现有架构

**两种方案**:

**方案 A: 修改 backbone 的分类头**（推荐用于完全替换）
- 修改 `backbone/vit_dualprompt.py` 的 `reset_classifier()` 方法
- 支持 KACLayer 作为分类头
- 需要确保 forward_head() 兼容 KACLayer

**方案 B: 添加独立的 fc 层**（推荐用于并行使用）
- 在 `PromptVitNet` 中添加 `self.fc` 层（KACLayer）
- 保持 backbone 的 `self.head` 不变
- 在评估时可以选择使用哪个分类器

**注意**: 如果使用方案 B，训练时需要考虑使用哪个分类器。

### 7. 训练流程的修改

**需要修改**:
- `_init_train()`: 保持现有的训练流程（使用 backbone 的 logits）
- 如果使用 KAC 替换，需要修改 optimizer 的参数列表
- 添加 `_build_classifiers()` 方法（训练结束后构建分类器）

### 8. 评估流程的修改

**需要修改**:
- `eval_task()`: 添加对三个分类器的评估
- 保持现有的 `_eval_cnn()` 用于 backbone 分类头评估
- 添加 `_eval_ncm_fc()`, `_eval_hc_soinn()`, `_eval_kac()` 方法

## 四、集成步骤建议

### 步骤 1: 添加特征提取方法
1. 在 `PromptVitNet` 类中添加 `extract_vector()` 方法
2. 在 `PromptVitNet` 类中添加 `feature_dim` 属性（768）

### 步骤 2: 添加 NCM 分类器
1. 在 `PromptVitNet.__init__()` 中添加 `self.ncm_fc = CosineLinear(...)`
2. 在 Learner 的 `__init__()` 中初始化 `_class_means = None`
3. 添加 `_build_ncm_classifier()` 方法
4. 添加 `_eval_ncm_fc()` 方法
5. 在 `_build_classifiers()` 中调用 `_build_ncm_classifier()`
6. 在 `eval_task()` 中添加 NCM 评估

### 步骤 3: 添加 HC-SOINN 分类器
1. 在 Learner 的 `__init__()` 中添加 `self.hc_soinn = HCSOINNClassifier(...)`
2. 添加 `_build_hc_soinn_bank()` 方法
3. 添加 `_eval_hc_soinn()` 方法
4. 在 `_build_classifiers()` 中调用 `_build_hc_soinn_bank()`
5. 在 `eval_task()` 中添加 HC-SOINN 评估
6. 在 `after_task()` 中添加 HC-SOINN 的压缩和锚点选择（如果使用 STAR）

### 步骤 4: 添加 KAC 分类器（可选）
1. **如果方案 A（替换 backbone 分类头）**:
   - 修改 `backbone/vit_dualprompt.py` 的 `reset_classifier()` 方法
   - 支持 KACLayer 作为分类头
   - 修改 optimizer 的参数列表

2. **如果方案 B（添加独立 fc 层）**:
   - 在 `PromptVitNet.__init__()` 中添加 `self.fc = KACLayer(...)`
   - 修改训练流程以使用 `self.fc`（或保持使用 backbone 的 head）
   - 添加 `_eval_kac()` 方法

### 步骤 5: 统一构建分类器
1. 添加 `_build_classifiers()` 方法（在训练结束后调用）
2. 在 `incremental_train()` 的末尾调用 `_build_classifiers()`

### 步骤 6: 修改评估方法
1. 修改 `eval_task()` 方法，添加三个分类器的评估
2. 保持现有的 backbone 分类头评估作为基准

## 五、注意事项

### 1. 保持向后兼容
- 确保不破坏现有的 DualPrompt 训练和评估流程
- 分类器集成应该是可选的（通过配置参数控制）

### 2. 特征一致性
- 确保所有分类器使用相同的特征提取方式
- 使用 `extract_vector()` 统一提取特征

### 3. 设备一致性
- 确保特征提取和分类器计算在正确的设备上（CPU/GPU）

### 4. 数据格式一致性
- NCM 分类器使用 `CosineLinear`，需要归一化特征
- HC-SOINN 分类器使用 numpy 数组
- KAC 分类器使用 torch tensor

### 5. 任务流程一致性
- 分类器构建应该在训练结束后、评估开始前
- 确保分类器构建的顺序：NCM → HC-SOINN → KAC（如果独立）

### 6. 配置参数
- 参考 CodaPrompt 的配置格式，添加三个分类器的配置参数
- 确保配置参数有合理的默认值

## 六、参考代码位置

### CodaPrompt 相关代码
- `models/coda_prompt.py`: Learner 类的完整实现
- `utils/inc_net.py`: `CodaPromptVitNet` 类的实现（第 770-900 行）
- `exps/coda_prompt_hc_soinn.json`: HC-SOINN 配置示例
- `exps/coda_prompt_kac.json`: KAC 配置示例

### DualPrompt 相关代码
- `models/dualprompt.py`: Learner 类的当前实现
- `utils/inc_net.py`: `PromptVitNet` 类的实现（第 741-768 行）
- `backbone/vit_dualprompt.py`: VisionTransformer 的 DualPrompt 实现
- `exps/dualprompt.json`: 当前配置示例

### 分类器实现
- `utils/ncm_classifier.py` 或 `backbone/linears.py`: CosineLinear (NCM)
- `utils/hc_soinn_classifier.py`: HCSOINNClassifier
- `utils/kac_classifier.py`: KACLayer


