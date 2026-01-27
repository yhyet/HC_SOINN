# t-SNE/UMAP 可视化使用指南

## 问题说明

t-SNE 每次运行都会产生不同的坐标系，无法进行跨任务的特征漂移观察和跨实验的横向对比。

## 解决方案

使用 **UMAP** 替代 t-SNE，因为 UMAP 支持 `transform` 新数据，可以实现固定坐标系。

## 功能说明

### 1. 固定坐标系观察特征漂移

- **第一个任务（Task 0）**：自动 fit UMAP 模型并保存到 `{vis_dir}/umap_model.pkl`
- **后续任务（Task 1-9）**：自动加载已保存的 UMAP 模型，使用 `transform` 方法将新特征映射到固定坐标系
- 这样所有任务的可视化都在**相同的坐标系**中，可以清晰观察特征漂移

### 2. 横向对比 hcsoinn 和 hcsoinn+star

#### 方法一：使用参考 UMAP 模型（推荐）

在 `hcsoinn+star` 的配置文件中添加 `reference_umap_model_path` 参数，指向 `hcsoinn` 实验的 UMAP 模型：

```json
{
    "visualize_tsne": true,
    "reference_umap_model_path": "logs/coda_prompt/cifar224/10/10/hcsoinn_visualizations/umap_model.pkl"
}
```

这样 `hcsoinn+star` 实验会使用 `hcsoinn` 实验的 UMAP 模型，两个实验的可视化将在**相同的坐标系**中，便于横向对比。

#### 方法二：手动对比

1. 先运行 `hcsoinn` 实验，生成 UMAP 模型
2. 复制 UMAP 模型路径
3. 在 `hcsoinn+star` 配置文件中添加 `reference_umap_model_path` 参数
4. 运行 `hcsoinn+star` 实验

## 文件结构

```
logs/coda_prompt/cifar224/10/10/
├── hcsoinn_visualizations/          # hcsoinn 实验的可视化
│   ├── umap_model.pkl               # UMAP 模型（Task 0 时生成）
│   ├── tsne_task0.png
│   ├── tsne_task1.png
│   └── ...
└── hcsoinn_star_visualizations/      # hcsoinn+star 实验的可视化
    ├── umap_model.pkl               # UMAP 模型（如果使用参考模型，则不会生成）
    ├── tsne_task0.png
    ├── tsne_task1.png
    └── ...
```

## 配置示例

### hcsoinn 实验配置（coda_prompt_hc_soinn.json）

```json
{
    "use_hc_soinn": true,
    "use_feature_alignment": false,
    "visualize_tsne": true
}
```

### hcsoinn+star 实验配置（coda_prompt_hc_soinn_star.json）

```json
{
    "use_hc_soinn": true,
    "use_feature_alignment": true,
    "visualize_tsne": true,
    "reference_umap_model_path": "logs/coda_prompt/cifar224/10/10/hcsoinn_visualizations/umap_model.pkl"
}
```

## 注意事项

1. **必须先运行 hcsoinn 实验**，生成 UMAP 模型后，才能在 hcsoinn+star 中使用
2. 如果 UMAP 不可用（未安装），会自动 fallback 到 t-SNE（但无法固定坐标系）
3. UMAP 模型路径是相对于项目根目录的
4. 使用参考模型时，两个实验的可视化将使用相同的坐标系，便于观察 STAR 对齐的效果

## 安装 UMAP

```bash
pip install umap-learn
```

## 可视化内容

每个任务的可视化包含：
- **样本点**：正常大小圆点（s=30）
- **HC-SOINN 点**：更大圆点（s=120），带黑色边框，显示原型点
- **HC-SOINN 边**：连接相邻原型点的边（linewidth=2.0），每个节点只显示前2个邻居
- **NCM 点**：×标记（s=150），带黑色边框，表示类中心


