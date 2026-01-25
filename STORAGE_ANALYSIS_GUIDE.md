# 存储占用分析实验指南

## 实验目标

对比三种配置下的存储空间占用：
1. **CodaPrompt** (基础模型)
2. **CodaPrompt + HC-SOINN**
3. **CodaPrompt + HC-SOINN + STAR**

## 存储组件说明

### 1. 基础模型 (CodaPrompt)
- **Backbone参数**: ViT backbone的权重
- **Prompt参数**: 可学习的prompt tokens
- **FC层参数**: 分类头权重

### 2. HC-SOINN存储
- **节点存储** (每个节点):
  - `center`: 归一化中心向量 (D维 float32)
  - `center_raw`: 原始中心向量 (D维 float32)
  - `count`: 样本计数 (int32)
- **NCM中心存储** (每个类别):
  - `class_mu`: 归一化类中心 (D维 float32)
  - `class_mu_raw`: 原始类中心 (D维 float32)
- **原始节点备份** (用于STAR):
  - `class_clusters_original`: 原始节点的深拷贝

**计算公式**:
```
HC-SOINN存储 = (节点数 × 2 × D × 4 bytes) + (节点数 × 4 bytes) + (类别数 × 2 × D × 4 bytes) + (原始备份)
```

### 3. STAR存储
- **锚点图像** (每个锚点):
  - 图像数据: H × W × C × 1 byte (uint8)
- **参考特征** (每个锚点):
  - `feats_ref`: D维 float32特征向量
- **EMA漂移量** (每个锚点):
  - `ema_delta`: D维 float32漂移向量
- **中心参考** (如果存在):
  - `centers_raw_ref`: 节点原始中心的参考 (M × D float32)

**计算公式**:
```
STAR存储 = (锚点数 × 图像大小) + (锚点数 × D × 4 bytes) + (锚点数 × D × 4 bytes) + (中心参考)
```

## 实验配置

### 配置文件设置

#### 配置1: CodaPrompt (基础)
```json
{
    "model_name": "coda_prompt",
    "use_hc_soinn": false,
    "use_feature_alignment": false,
    "enable_storage_analysis": true
}
```

#### 配置2: CodaPrompt + HC-SOINN
```json
{
    "model_name": "coda_prompt",
    "use_hc_soinn": true,
    "use_feature_alignment": false,
    "enable_storage_analysis": true,
    "hcsoinn_max_proto_per_class": 20
}
```

#### 配置3: CodaPrompt + HC-SOINN + STAR
```json
{
    "model_name": "coda_prompt",
    "use_hc_soinn": true,
    "use_feature_alignment": true,
    "use_full_task_rehearsal": false,
    "star_mode": "rigid",
    "star_lambda": 0.3,
    "enable_storage_analysis": true,
    "hcsoinn_max_proto_per_class": 20
}
```

## 需要统计的数值

### 1. 模型参数存储
- Backbone参数量 (个)
- Prompt参数量 (个)
- FC层参数量 (个)
- **总参数量** (个)
- **总存储大小** (MB)

### 2. HC-SOINN存储
- **总节点数** (个)
- **平均每类节点数** (个/类)
- **节点存储明细**:
  - 归一化中心存储 (MB)
  - 原始中心存储 (MB)
  - 计数存储 (MB)
- **NCM中心存储** (MB)
- **原始备份存储** (MB)
- **HC-SOINN总存储** (MB)

### 3. STAR存储
- **总锚点数** (个)
- **平均每类锚点数** (个/类)
- **存储明细**:
  - 图像存储 (MB)
  - 参考特征存储 (MB)
  - EMA漂移量存储 (MB)
  - 中心参考存储 (MB)
- **STAR总存储** (MB)

### 4. 总体统计
- **总存储占用** (MB)
- **各组件占比** (%):
  - 模型参数占比
  - HC-SOINN占比
  - STAR占比

## 实验步骤

### 步骤1: 准备配置文件
创建三个配置文件：
- `coda_prompt_base.json` (基础)
- `coda_prompt_hc_soinn.json` (HC-SOINN)
- `coda_prompt_hc_soinn_star.json` (HC-SOINN + STAR)

### 步骤2: 运行实验
对每个配置运行完整的增量学习流程，确保：
- 所有任务都训练完成
- 存储分析在每个任务结束后自动执行

### 步骤3: 收集数据
从日志文件中提取存储分析报告，记录：
- 每个任务结束后的存储占用
- 最终任务的总存储占用

### 步骤4: 数据分析
对比三种配置：
- 绘制存储占用对比图
- 分析各组件存储占比
- 计算存储增长趋势

## 输出示例

存储分析器会在每个任务结束后输出如下报告：

```
================================================================================
存储占用分析报告 - Task 2
================================================================================

【模型参数】
  Backbone: 86,000,000 参数 (328.13 MB)
  Prompt:   1,000,000 参数 (3.81 MB)
  FC:       30,000 参数 (0.11 MB)
  总计:     87,030,000 参数 (332.05 MB)

【HC-SOINN】
  类别数: 20
  总节点数: 380
  平均每类节点数: 19.0
  特征维度: 768
  存储明细:
    - nodes_center_mb: 1.1641 MB
    - nodes_center_raw_mb: 1.1641 MB
    - nodes_count_mb: 0.0014 MB
    - ncm_mu_mb: 0.1229 MB
    - ncm_mu_raw_mb: 0.1229 MB
    - original_backup_mb: 2.3281 MB
  总计: 4.9035 MB

【STAR】
  类别数: 20
  总锚点数: 400
  平均每类锚点数: 20.0
  特征维度: 768
  存储明细:
    - images_mb: 57.6000 MB
    - feats_ref_mb: 1.1719 MB
    - ema_delta_mb: 1.1719 MB
    - centers_raw_ref_mb: 0.1229 MB
  总计: 60.0667 MB

【总计】
  总存储: 397.02 MB
  模型占比: 83.6%
  HC-SOINN占比: 1.2%
  STAR占比: 15.1%
================================================================================
```

## 关键指标

### 存储效率指标
1. **每类存储占用**: 总存储 / 类别数 (MB/类)
2. **节点存储效率**: HC-SOINN存储 / 节点数 (MB/节点)
3. **锚点存储效率**: STAR存储 / 锚点数 (MB/锚点)

### 增长趋势
- 存储占用随任务数增长的趋势
- 各组件存储占比的变化
- 存储效率的变化

## 注意事项

1. **图像尺寸**: 根据数据集调整 `image_shape` 参数
   - CIFAR-100: (32, 32, 3)
   - ImageNet: (224, 224, 3)
   - 其他数据集: 根据实际情况设置

2. **特征维度**: 不同backbone的特征维度不同
   - ViT-Base: 768
   - ViT-Large: 1024
   - 需要根据实际模型调整

3. **节点数量**: HC-SOINN的节点数受 `max_prototypes_per_class` 限制
   - 默认值: 20
   - 可根据实验需求调整

4. **锚点数量**: STAR的锚点数量取决于：
   - `star_mode`: "rigid" 或 "trajectory"
   - `use_full_task_rehearsal`: true/false
   - HC-SOINN节点数

## 结果分析建议

1. **对比分析**:
   - 基础模型 vs 添加HC-SOINN的存储增量
   - HC-SOINN vs 添加STAR的存储增量
   - 总存储占用对比

2. **效率分析**:
   - 每类存储占用对比
   - 存储增长速率对比
   - 存储占比分析

3. **可扩展性分析**:
   - 存储占用随类别数增长的趋势
   - 大规模数据集下的存储需求预估

