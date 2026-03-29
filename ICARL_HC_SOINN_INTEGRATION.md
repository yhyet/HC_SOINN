# iCaRL + HC-SOINN 集成方案

## 一、iCaRL 方法概述

### 核心机制

1. **Exemplar Memory（样本记忆）**
   - 存储每个类别的代表性样本
   - 在训练新任务时，将exemplar memory与新任务数据一起训练
   - 使用herding selection策略选择exemplar

2. **Knowledge Distillation（知识蒸馏）**
   - 使用旧模型（`_old_network`）的输出作为软标签
   - 防止灾难性遗忘
   - 损失函数：`loss = loss_clf + loss_kd`

3. **NCM分类器（Nearest Class Mean）**
   - 计算每个类别的特征均值（`_class_means`）
   - 在评估时使用最近类均值进行分类
   - 特征归一化后计算余弦距离

### 当前实现

- **训练流程**：`_init_train()`（第一个任务）或 `_update_representation()`（后续任务）
- **Exemplar构建**：`build_rehearsal_memory()` → `_construct_exemplar()` / `_reduce_exemplar()`
- **类均值计算**：在构建exemplar时同时计算 `_class_means`
- **评估**：`eval_task()` → `_eval_cnn()` + `_eval_nme()`

## 二、HC-SOINN 集成方案

### 集成思路

HC-SOINN可以作为iCaRL的**增强分类器**，与NCM分类器并行使用：

1. **保持iCaRL的核心机制不变**
   - Exemplar memory机制继续使用
   - Knowledge distillation继续使用
   - 训练流程不变

2. **添加HC-SOINN作为额外分类器**
   - 在训练后构建HC-SOINN bank
   - 在评估时同时使用NCM和HC-SOINN
   - HC-SOINN使用多原型表示，比NCM的单原型更灵活

### 实现步骤

#### 1. 初始化HC-SOINN分类器

在 `Learner.__init__()` 中添加：

```python
from utils.hc_soinn_classifier import HCSOINNClassifier

# HC-SOINN plugin
self.use_hc_soinn = args.get("use_hc_soinn", False)
if self.use_hc_soinn:
    logging.info("Initializing HC-SOINNClassifier for iCaRL")
    self.hc_soinn = HCSOINNClassifier(
        max_prototypes_per_class=args.get("hcsoinn_max_proto_per_class", 20),
        alpha=args.get("hcsoinn_alpha", 0.5),
        tau_merge=args.get("hcsoinn_tau_merge", 0.2),
        tau_reject=args.get("hcsoinn_tau_reject", 2.0),
        linkage_method=args.get("hcsoinn_linkage", "average"),
        distance_metric=args.get("hcsoinn_distance", "cosine"),
        use_soinn_refinement=args.get("hcsoinn_use_soinn_refinement", True),
        soinn_ad=args.get("hcsoinn_soinn_ad", 20),
        soinn_lam=args.get("hcsoinn_soinn_lam", 20),
        soinn_threshold_scale=args.get("hcsoinn_soinn_threshold_scale", 0.5),
        soinn_max_iter=args.get("hcsoinn_soinn_max_iter", 3),
    )
```

#### 2. 添加特征提取函数

iCaRL已经有 `_extract_vectors()` 方法，但需要适配HC-SOINN：

```python
def _get_hc_soinn_feature_fn(self):
    """
    获取HC-SOINN特征提取函数
    使用与NCM相同的特征提取方式（归一化后的特征）
    """
    def feature_fn(x):
        """x: torch.Tensor [B, C, H, W]"""
        self._network.eval()
        with torch.no_grad():
            vectors, _ = self._extract_vectors(
                DataLoader(
                    [(None, x_i, None) for x_i in x],
                    batch_size=self.args["batch_size"],
                    shuffle=False,
                    num_workers=0
                )
            )
            # 归一化（与NCM保持一致）
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
        return vectors
    return feature_fn
```

#### 3. 构建HC-SOINN Bank

在 `incremental_train()` 的 `after_task()` 之后添加：

```python
def _build_hc_soinn_bank(self):
    """
    构建HC-SOINN bank：使用当前任务的训练数据
    累积存储机制：保留旧类别的节点，添加新类别的节点
    """
    if not self.use_hc_soinn:
        return
    
    logging.info(f"Building HC-SOINN bank: adding new classes ({self._known_classes}-{self._total_classes-1})")
    
    # 获取当前任务的训练数据（包括exemplar memory）
    train_dataset = self.data_manager.get_dataset(
        np.arange(self._known_classes, self._total_classes),
        source="train",
        mode="test"
    )
    
    # 如果有exemplar memory，也包含进去
    if len(self._data_memory) > 0:
        exemplar_dataset = self.data_manager.get_dataset(
            [],
            source="train",
            mode="test",
            appendent=(self._data_memory, self._targets_memory)
        )
        # 合并数据集（需要实现合并逻辑）
        # 这里简化处理，只使用新任务数据
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=self.args["batch_size"],
        shuffle=False,
        num_workers=num_workers
    )
    
    feature_fn = self._get_hc_soinn_feature_fn()
    
    # 提取特征
    embedding_list, label_list = [], []
    with torch.no_grad():
        for _, (_, inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self._device)
            vectors, _ = self._extract_vectors(
                DataLoader([(None, inputs[i], targets[i]) for i in range(len(inputs))],
                          batch_size=len(inputs), shuffle=False, num_workers=0)
            )
            # 归一化
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            embedding_list.append(vectors)
            label_list.append(targets.cpu().numpy())
    
    if len(embedding_list) > 0:
        feats_np = np.concatenate(embedding_list, axis=0)
        lbs_np = np.concatenate(label_list, axis=0)
        
        # 添加到HC-SOINN
        self.hc_soinn.add_features(feats_np, lbs_np)
        
        # 压缩（生成原型节点）
        try:
            self.hc_soinn.compress()
            logging.info("HC-SOINN bank built successfully")
        except Exception as e:
            logging.error(f"HC-SOINN compress error: {e}", exc_info=True)
```

#### 4. 添加HC-SOINN评估方法

```python
def _eval_hc_soinn(self, loader):
    """
    使用HC-SOINN分类器进行评估
    """
    if not self.use_hc_soinn:
        return None, None
    
    self._network.eval()
    feature_fn = self._get_hc_soinn_feature_fn()
    
    y_pred, y_true = [], []
    
    with torch.no_grad():
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            
            # 提取特征
            vectors, _ = self._extract_vectors(
                DataLoader([(None, inputs[i], targets[i]) for i in range(len(inputs))],
                          batch_size=len(inputs), shuffle=False, num_workers=0)
            )
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            
            # 使用HC-SOINN预测
            topk_pred = self.hc_soinn.predict_topk(
                vectors, k=1, return_distances=False
            )
            
            y_pred.append(topk_pred)
            y_true.append(targets.cpu().numpy())
    
    if len(y_pred) > 0:
        return np.concatenate(y_pred), np.concatenate(y_true)
    else:
        logging.warning("No predictions generated from HC-SOINN evaluation")
        return None, None
```

#### 5. 修改评估方法

在 `eval_task()` 中添加HC-SOINN评估：

```python
def eval_task(self):
    y_pred, y_true = self._eval_cnn(self.test_loader)
    cnn_accy = self._evaluate(y_pred, y_true)

    if hasattr(self, "_class_means"):
        y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
        nme_accy = self._evaluate(y_pred, y_true)
    else:
        nme_accy = None

    # 添加HC-SOINN评估
    results = {
        "fc": cnn_accy,
        "ncm": nme_accy,
    }
    
    if getattr(self, "use_hc_soinn", False) and self.hc_soinn is not None:
        y_pred_hc, y_true_hc = self._eval_hc_soinn(self.test_loader)
        if y_pred_hc is not None:
            results["hc_soinn"] = self._evaluate(y_pred_hc, y_true_hc)
    
    return results
```

## 三、关键点说明

### 1. 特征提取一致性

- HC-SOINN和NCM使用相同的特征提取方式
- 都使用归一化后的特征（L2归一化）
- 使用 `_extract_vectors()` 方法提取特征

### 2. 数据来源

- **训练时**：使用新任务数据 + exemplar memory
- **构建HC-SOINN bank时**：可以使用新任务数据，也可以包含exemplar memory
- **评估时**：使用测试集

### 3. 累积存储

- HC-SOINN支持增量添加新类别
- 旧类别的节点保留，新类别的节点添加
- 在 `compress()` 时会对所有类别进行压缩

### 4. 与NCM的关系

- NCM：单原型（类均值）
- HC-SOINN：多原型（多个节点）
- 两者可以并行使用，提供不同的分类视角

## 四、配置文件示例

```json
{
    "model_name": "icarl",
    "use_hc_soinn": true,
    "hcsoinn_max_proto_per_class": 20,
    "hcsoinn_alpha": 0.5,
    "hcsoinn_tau_merge": 0.2,
    "hcsoinn_tau_reject": 2.0,
    "hcsoinn_linkage": "average",
    "hcsoinn_distance": "cosine",
    "hcsoinn_use_soinn_refinement": true,
    "hcsoinn_soinn_ad": 20,
    "hcsoinn_soinn_lam": 20,
    "hcsoinn_soinn_threshold_scale": 0.5,
    "hcsoinn_soinn_max_iter": 3
}
```

## 五、优势

1. **保持iCaRL优势**：exemplar memory和知识蒸馏机制不变
2. **增强分类能力**：HC-SOINN的多原型表示比NCM的单原型更灵活
3. **易于集成**：只需添加评估时的分类器，不影响训练流程
4. **可选择性**：可以通过配置开关控制是否使用HC-SOINN

