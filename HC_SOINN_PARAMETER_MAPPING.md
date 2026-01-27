# HC-SOINN 论文参数与代码实现对应关系

## 一、层次聚类初始化阶段（Hierarchical Initialization）

| 论文中的参数/概念 | 代码中的参数 | JSON配置键 | 说明 |
|-----------------|------------|-----------|------|
| $K_{init}$ (目标聚类数) | `max_prototypes_per_class` | `hcsoinn_max_proto_per_class` | 控制层次聚类后初始节点数量，也是最终原型的最大数量 |
| Linkage方法 (average linkage/UPGMA) | `linkage_method` | `hcsoinn_linkage` | 层次聚类的链接准则，论文使用 "average"，代码默认也是 "average" |
| 距离度量 (cosine distance) | `distance_metric` | `hcsoinn_distance` | 计算簇间距离的度量方式，论文使用 "cosine"，代码默认也是 "cosine" |
| - | `tau_merge` | `hcsoinn_tau_merge` | **论文未提及**：合并相近簇的阈值（代码中可能用于后处理，默认0.2） |
| - | `tau_reject` | `hcsoinn_tau_reject` | **论文未提及**：拒绝异常簇的阈值（代码中可能用于后处理，默认2.0） |

## 二、SOINN 精炼阶段（Spherical SOINN Refinement）

| 论文中的参数/概念 | 代码中的参数 | JSON配置键 | 说明 |
|-----------------|------------|-----------|------|
| $age_{max}$ (边最大年龄) | `soinn_ad` | `hcsoinn_soinn_ad` | 边年龄超过此值时被删除，论文中对应 $age_{max}$，代码默认20 |
| $T_{soinn}$ (迭代次数) | `soinn_max_iter` | `hcsoinn_soinn_max_iter` | SOINN精炼的最大迭代轮数，论文中对应 $T_{soinn}$，代码默认3 |
| $\eta_1, \eta_2$ (学习率) | - | - | **自动计算**：代码中根据迭代次数和样本索引自动计算，$\eta_1 = 1/(t + iteration \times N + 1)$，$\eta_2 = 1/(100 \times (t + iteration \times N + 1))$ |
| - | `soinn_threshold_scale` | `hcsoinn_soinn_threshold_scale` | **论文未明确提及**：阈值缩放因子，控制是否插入新节点（0.5-0.8常用，默认0.5） |
| - | `soinn_lam` | `hcsoinn_soinn_lam` | **论文未提及**：每 lam 次迭代删除孤立节点（代码中已弃用，默认20） |
| - | `soinn_max_degree_for_removal` | `hcsoinn_soinn_max_degree_for_removal` | **论文未提及**：删除节点的最大度阈值，度 ≤ 此值的节点会被删除（默认1） |
| - | `use_soinn_refinement` | `hcsoinn_use_soinn_refinement` | **论文未提及**：是否启用SOINN精炼的开关（默认True） |

## 三、推理阶段（Dual-View Inference）

| 论文中的参数/概念 | 代码中的参数 | JSON配置键 | 说明 |
|-----------------|------------|-----------|------|
| $\alpha$ (平衡因子) | `alpha` | `hcsoinn_alpha` | 平衡全局视图（NCM）和局部视图（子原型）的权重，$\alpha \in [0,1]$，代码默认0.5 |

## 四、参数对应关系总结

### 核心对应关系（论文中明确提及）

1. **$K_{init}$ ↔ `hcsoinn_max_proto_per_class`**
   - 论文：层次聚类目标聚类数
   - 代码：每类最大原型数（默认20，你的配置是60）

2. **Linkage方法 ↔ `hcsoinn_linkage`**
   - 论文：average linkage (UPGMA)
   - 代码：默认 "average"

3. **距离度量 ↔ `hcsoinn_distance`**
   - 论文：cosine distance
   - 代码：默认 "cosine"

4. **$age_{max}$ ↔ `hcsoinn_soinn_ad`**
   - 论文：边最大年龄
   - 代码：默认20

5. **$T_{soinn}$ ↔ `hcsoinn_soinn_max_iter`**
   - 论文：SOINN迭代次数
   - 代码：默认3，你的配置是1

6. **$\alpha$ ↔ `hcsoinn_alpha`**
   - 论文：双视图推理的平衡因子
   - 代码：默认0.5

### 代码中额外实现但论文未明确提及的参数

1. **`hcsoinn_soinn_threshold_scale`** (0.8)
   - 作用：控制SOINN精炼时是否插入新节点的阈值缩放因子
   - 论文中：虽然提到了节点插入判断，但没有明确这个缩放因子
   - 影响：值越大，越不容易插入新节点，原型更紧凑

2. **`hcsoinn_soinn_max_degree_for_removal`** (默认1)
   - 作用：删除孤立节点的度阈值
   - 论文中：提到了删除孤立节点，但没有明确度阈值

3. **`hcsoinn_tau_merge`** (0.2) 和 **`hcsoinn_tau_reject`** (2.0)
   - 作用：可能用于层次聚类后的后处理（合并/拒绝簇）
   - 论文中：未明确提及

## 五、你的配置参数解读

根据你的配置文件 `coda_prompt_hc_soinn_star.json`：

```json
{
    "hcsoinn_max_proto_per_class": 60,        // K_init = 60，比默认值20更大，更精细
    "hcsoinn_alpha": 0.5,                     // α = 0.5，平衡全局和局部视图
    "hcsoinn_linkage": "average",             // 使用average linkage，与论文一致
    "hcsoinn_distance": "cosine",             // 使用cosine距离，与论文一致
    "hcsoinn_soinn_ad": 20,                   // age_max = 20，与论文一致
    "hcsoinn_soinn_max_iter": 1,              // T_soinn = 1，比默认值3更少，快速收敛
    "hcsoinn_soinn_threshold_scale": 0.8,     // 阈值缩放0.8，较保守，不容易插入新节点
    "hcsoinn_use_soinn_refinement": true      // 启用SOINN精炼
}
```

**配置特点**：
- 更大的原型数（60 vs 20）：适合更复杂的类别分布
- 更少的SOINN迭代（1 vs 3）：快速收敛，但可能不够精细
- 更保守的阈值缩放（0.8 vs 0.5）：生成更紧凑的原型，减少冗余节点

