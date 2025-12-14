import logging
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg', force=True)  # 强制使用非 GUI 后端，避免多进程环境下的 tkinter 错误
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from utils.inc_net import SimpleVitNetKNN
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
from utils.hc_soinn_classifier import HCSOINNClassifier


num_workers = 8
batch_size = 128


class Learner(BaseLearner):
    """
    SimpleCIL + HC-SOINN 分类头

    - 训练阶段仅提取特征并交给 HC-SOINN 生成原型
    - 压缩仅在任务结束时进行（compress），无在线新模式发现
    - 推理阶段使用 NCM+簇中心融合距离
    """

    def __init__(self, args):
        super().__init__(args)
        self._network = SimpleVitNetKNN(args, True)
        self.args = args

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
            soinn_max_degree_for_removal=args.get("hcsoinn_soinn_max_degree_for_removal", 1),
        )

    def after_task(self):
        # 每个 task 结束时压缩，并更新已知类别
        # 在更新 _known_classes 之前保存当前任务的新类别范围（用于可视化）
        current_task_classes = list(range(self._known_classes, self._total_classes))
        
        try:
            self.hc_soinn.compress()
        except Exception as e:
            logging.error(f"HC-SOINN compress error: {e}", exc_info=True)
        
        self._known_classes = self._total_classes
        
        # 执行 HC-SOINN t-SNE 可视化（传入当前任务的新类别范围）
        try:
            self._visualize_hc_soinn_tsne(current_task_classes)
        except Exception as e:
            logging.error(f"HC-SOINN t-SNE visualization error: {e}", exc_info=True)

    def _extract_class_features(self, loader, model):
        """
        提取各类特征用于 HC-SOINN；要求 loader 关闭数据增强（mode='test'）
        """
        model.eval()
        feats, lbs = [], []
        with torch.no_grad():
            for _, data, label in loader:
                data = data.to(self._device)
                label = label.to(self._device)
                emb = model.backbone(data)
                feats.append(emb.cpu())
                lbs.append(label.cpu())

        if len(feats) == 0:
            return
        feats = torch.cat(feats, dim=0).numpy()
        lbs = torch.cat(lbs, dim=0).numpy()
        self.hc_soinn.add_features(feats, lbs)

        proto_info = self.hc_soinn.prototypes_per_class()
        logging.info(f"HC-SOINN prototypes per class: {proto_info}")

    def incremental_train(self, data_manager):
        logging.info(
            f"Starting incremental_train: cur_task={self._cur_task}, known_classes={self._known_classes}"
        )
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        logging.info(
            f"After update: cur_task={self._cur_task}, total_classes={self._total_classes}, "
            f"known_classes={self._known_classes}"
        )
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="train"
        )
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        # 关闭数据增强用于特征提取
        train_dataset_for_hc = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="test",
        )
        self.train_loader_for_hc = DataLoader(
            train_dataset_for_hc,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        if len(self._multiple_gpus) > 1:
            logging.info("Using multiple GPUs")
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._train(self.train_loader, self.test_loader, self.train_loader_for_hc)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader, train_loader_for_hc):
        # 这里只负责特征提取与 HC-SOINN 更新
        self._network.to(self._device)
        self._extract_class_features(train_loader_for_hc, self._network)

    def _eval_cnn(self, loader):
        """
        使用 HC-SOINN 进行分类评估
        """
        self._network.eval()
        y_pred, y_true = [], []

        with torch.no_grad():
            for _, (_, inputs, targets) in enumerate(loader):
                inputs = inputs.to(self._device)
                features = self._network.extract_vector(inputs)
                features = tensor2numpy(features)

                topk_pred = self.hc_soinn.predict_topk(
                    features, self.topk, self._total_classes, device=self._device
                )
                y_pred.append(topk_pred)
                y_true.append(targets.numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)

    def eval_task(self):
        """
        评估任务：返回与 trainer.py 兼容的结果字典
        """
        y_pred, y_true = self._eval_cnn(self.test_loader)
        acc = self._evaluate(y_pred, y_true)
        # 使用 "hc_soinn" 作为 key 以在日志中正确显示 HC-SOINN
        return {"hc_soinn": acc}

    def _visualize_hc_soinn_tsne(self, target_classes=None):
        """
        HC-SOINN t-SNE 可视化：对当前 task 新出现的类别进行可视化
        显示样本点、NCM 点（全局类中心）和聚类 SOINN 点（子簇原型）
        
        参数:
            target_classes: 要可视化的类别列表。如果为 None，则使用当前任务的新类别。
        """
        # 确保使用非 GUI 后端，避免多进程环境下的 tkinter 错误
        matplotlib.use('Agg', force=True)
        
        # 如果没有指定类别，则使用当前任务的新类别
        if target_classes is None:
            target_classes = list(range(self._known_classes, self._total_classes))
        
        if len(target_classes) == 0:
            logging.warning(f"No classes to visualize for task {self._cur_task}, skipping visualization")
            return
        
        logging.info(f"Visualizing HC-SOINN for task {self._cur_task} classes: {target_classes}")
        
        # 检查 HC-SOINN 是否有数据
        if not hasattr(self.hc_soinn, 'class_mu') or len(self.hc_soinn.class_mu) == 0:
            logging.warning("HC-SOINN class centers are empty, skipping visualization")
            return
        
        # 获取可用的类别（有 NCM 中心的类别）
        available_classes = [cls for cls in target_classes if cls in self.hc_soinn.class_mu]
        if len(available_classes) == 0:
            logging.warning("No target classes found in HC-SOINN, skipping visualization")
            return
        
        # 检查 data_manager 是否可用
        if not hasattr(self, 'data_manager'):
            logging.warning("DataManager not found, skipping visualization")
            return
        
        try:
            # 1. 提取选定类别的训练样本特征
            train_dataset = self.data_manager.get_dataset(
                np.array(available_classes),
                source="train",
                mode="test"  # 使用 test 模式关闭数据增强
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=num_workers
            )
            
            sample_features = []
            sample_labels = []
            
            self._network.eval()
            with torch.no_grad():
                for _, inputs, targets in train_loader:
                    inputs = inputs.to(self._device)
                    feats = self._network.extract_vector(inputs)
                    feats_np = feats.detach().cpu().numpy()
                    sample_features.append(feats_np)
                    sample_labels.append(targets.numpy())
            
            if len(sample_features) == 0:
                logging.warning("No sample features extracted, skipping visualization")
                return
            
            sample_features = np.concatenate(sample_features, axis=0)
            sample_labels = np.concatenate(sample_labels, axis=0)
            
            # 只保留目标类别的样本
            mask = np.isin(sample_labels, available_classes)
            sample_features = sample_features[mask]
            sample_labels = sample_labels[mask]
            
            if len(sample_features) == 0:
                logging.warning("No samples found for target classes, skipping visualization")
                return
            
            # 归一化样本特征
            sample_features = sample_features / (np.linalg.norm(sample_features, axis=1, keepdims=True) + 1e-8)
            
            # 2. 收集 NCM 点（全局类中心）
            ncm_features = []
            ncm_labels = []
            
            for cls in available_classes:
                if cls in self.hc_soinn.class_mu:
                    ncm_features.append(self.hc_soinn.class_mu[cls])
                    ncm_labels.append(cls)
            
            if len(ncm_features) == 0:
                logging.warning("No NCM centers found, skipping visualization")
                return
            
            ncm_features = np.stack(ncm_features, axis=0)
            ncm_labels = np.array(ncm_labels)
            
            # 归一化 NCM 特征
            ncm_features = ncm_features / (np.linalg.norm(ncm_features, axis=1, keepdims=True) + 1e-8)
            
            # 3. 收集聚类 SOINN 点（子簇原型）和边信息
            cluster_features = []
            cluster_labels = []
            cluster_class_mapping = []  # 记录每个 cluster 属于哪个类，以及在该类中的索引
            class_cluster_edges = {}  # 存储每个类的边信息：{cls: {node_idx: set(neighbor_idx)}}
            
            for cls in available_classes:
                if cls in self.hc_soinn.class_clusters:
                    clusters = self.hc_soinn.class_clusters[cls]
                    if len(clusters) > 0:
                        cls_cluster_centers = np.stack([c.center for c in clusters], axis=0)
                        cluster_features.append(cls_cluster_centers)
                        cluster_labels.extend([cls] * len(clusters))
                        # 记录每个 cluster 的类别和索引
                        start_idx = len(cluster_class_mapping)
                        for i in range(len(clusters)):
                            cluster_class_mapping.append((cls, i))  # (类别, 在该类中的索引)
                        
                        # 获取该类的边信息（如果存在）
                        if hasattr(self.hc_soinn, 'class_edges') and cls in self.hc_soinn.class_edges:
                            class_cluster_edges[cls] = self.hc_soinn.class_edges[cls]
            
            if len(cluster_features) > 0:
                cluster_features = np.concatenate(cluster_features, axis=0)
                cluster_labels = np.array(cluster_labels)
                # 归一化聚类特征
                cluster_features = cluster_features / (np.linalg.norm(cluster_features, axis=1, keepdims=True) + 1e-8)
            else:
                cluster_features = np.empty((0, ncm_features.shape[1]))
                cluster_labels = np.array([])
            
            # 4. 合并所有特征进行 t-SNE 降维
            all_features_list = [sample_features, ncm_features]
            all_labels_list = [sample_labels, ncm_labels]
            
            if len(cluster_features) > 0:
                all_features_list.append(cluster_features)
                all_labels_list.append(cluster_labels)
            
            all_features = np.vstack(all_features_list)
            
            if len(all_features) < 2:
                logging.warning("Not enough features for t-SNE, skipping visualization")
                return
            
            # 计算分割点
            sample_end = len(sample_features)
            ncm_end = sample_end + len(ncm_features)
            
            logging.info(f"Computing t-SNE for {len(all_features)} points ({len(sample_features)} samples + {len(ncm_features)} NCM + {len(cluster_features)} clusters)...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_features) - 1), max_iter=1000)
            embeddings = tsne.fit_transform(all_features)
            
            # 分离不同类型的嵌入
            sample_embeddings = embeddings[:sample_end]
            ncm_embeddings = embeddings[sample_end:ncm_end]
            cluster_embeddings = embeddings[ncm_end:] if len(cluster_features) > 0 else None
            
            # 5. 创建图形
            plt.figure(figsize=(16, 12))
            
            # 为每个类别定义颜色（使用类别在 available_classes 中的索引，而不是全局类别编号）
            num_classes = len(available_classes)
            # 使用更多颜色以支持更多类别
            if num_classes <= 10:
                colors = plt.cm.tab10(np.linspace(0, 1, 10))[:num_classes]
            else:
                colors = plt.cm.tab20(np.linspace(0, 1, 20))[:num_classes]
            
            # 创建类别到颜色的映射
            cls_to_color = {cls: colors[i] for i, cls in enumerate(available_classes)}
            
            # 绘制样本点（稍大一些，便于观察）
            for idx, cls in enumerate(available_classes):
                cls_mask = sample_labels == cls
                if np.any(cls_mask):
                    plt.scatter(
                        sample_embeddings[cls_mask, 0],
                        sample_embeddings[cls_mask, 1],
                        c=[cls_to_color[cls]],
                        marker='.',  # 使用点标记
                        s=30,  # 从15增大到30，使样本点更清晰
                        alpha=0.4,  # 从0.3增大到0.4，稍微增强可见度
                        label=f'Samples (Class {cls})' if idx == 0 else '',
                        edgecolors='none',
                        zorder=1  # 在最底层
                    )
            
            # 绘制聚类 SOINN 点（中等大小圆点）
            if cluster_embeddings is not None and len(cluster_embeddings) > 0:
                # 首先绘制边（在节点之前，这样边在节点下面）
                for cls in available_classes:
                    if cls in class_cluster_edges and len(class_cluster_edges[cls]) > 0:
                        # 找到该类在 cluster_embeddings 中的索引范围
                        cls_mask = cluster_labels == cls
                        cls_indices = np.where(cls_mask)[0]
                        if len(cls_indices) == 0:
                            continue
                        
                        # 建立类内索引到全局 cluster_embeddings 索引的映射
                        cls_cluster_start = cls_indices[0]
                        cls_edges = class_cluster_edges[cls]
                        
                        # 绘制边
                        for node_idx, neighbors in cls_edges.items():
                            if node_idx >= len(cls_indices):
                                continue  # 跳过无效索引（可能因为截断导致）
                            global_idx_i = cls_cluster_start + node_idx
                            if global_idx_i >= len(cluster_embeddings):
                                continue
                            
                            for neighbor_idx in neighbors:
                                if neighbor_idx >= len(cls_indices):
                                    continue
                                global_idx_j = cls_cluster_start + neighbor_idx
                                if global_idx_j >= len(cluster_embeddings):
                                    continue
                                
                                # 只绘制一次边（i < j 避免重复）
                                if node_idx < neighbor_idx:
                                    x_coords = [cluster_embeddings[global_idx_i, 0], cluster_embeddings[global_idx_j, 0]]
                                    y_coords = [cluster_embeddings[global_idx_i, 1], cluster_embeddings[global_idx_j, 1]]
                                    plt.plot(
                                        x_coords, y_coords,
                                        color=cls_to_color[cls],
                                        alpha=0.3,  # 边的透明度
                                        linewidth=0.8,
                                        zorder=2  # 在样本点之上，但在节点之下
                                    )
                
                # 然后绘制节点
                for idx, cls in enumerate(available_classes):
                    cls_mask = cluster_labels == cls
                    if np.any(cls_mask):
                        plt.scatter(
                            cluster_embeddings[cls_mask, 0],
                            cluster_embeddings[cls_mask, 1],
                            c=[cls_to_color[cls]],
                            marker='o',  # 圆点
                            s=60,
                            alpha=0.7,
                            label=f'Cluster Prototypes' if idx == 0 else '',
                            edgecolors='black',
                            linewidths=0.5,
                            zorder=3  # 在样本点和边之上
                        )
            
            # 绘制 NCM 点（大星形，最明显）
            for idx, cls in enumerate(available_classes):
                cls_mask = ncm_labels == cls
                if np.any(cls_mask):
                    plt.scatter(
                        ncm_embeddings[cls_mask, 0],
                        ncm_embeddings[cls_mask, 1],
                        c=[cls_to_color[cls]],
                        marker='*',  # 星形
                        s=500,
                        alpha=0.9,
                        label=f'Class {cls} (NCM)',
                        edgecolors='black',
                        linewidths=2.0,
                        zorder=10  # 确保 NCM 点在最上层
                    )
            
            plt.title(f'HC-SOINN t-SNE Visualization (Task {self._cur_task})', fontsize=14, fontweight='bold')
            plt.xlabel('t-SNE Dimension 1', fontsize=12)
            plt.ylabel('t-SNE Dimension 2', fontsize=12)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # 保存图片
            model_name = self.args.get("model_name", "simplecil_hc_soinn")
            dataset = self.args.get("dataset", "cifar224")
            init_cls = self.args.get("init_cls", 10)
            increment = self.args.get("increment", 10)
            
            vis_dir = f"logs/{model_name}/{dataset}/{init_cls}/{increment}/hcsoinn_visualizations"
            os.makedirs(vis_dir, exist_ok=True)
            
            vis_path = os.path.join(vis_dir, f"tsne_task{self._cur_task}.png")
            plt.savefig(vis_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"HC-SOINN t-SNE visualization saved to {vis_path}")
            logging.info(f"  - Samples: {len(sample_features)}")
            logging.info(f"  - NCM centers: {len(ncm_features)}")
            logging.info(f"  - Cluster prototypes: {len(cluster_features)}")
            
        except Exception as e:
            logging.error(f"Error in HC-SOINN t-SNE visualization: {e}", exc_info=True)


