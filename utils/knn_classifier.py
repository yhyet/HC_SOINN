import logging
from typing import Dict, List, Optional

import numpy as np
import torch
from scipy.spatial.distance import cdist


class KNNClassifier:
    """
    可插拔的 KNN 分类器：
    - 在增量任务场景中维护一个“特征库”（feature bank），跨任务累积训练样本的特征向量。
    - 在推理阶段对查询特征与特征库计算距离，进行基于投票的 top-k 类别预测。

    设计目标：
    - 与不同 Learner 解耦（simplecil、coda_prompt 等），作为通用组件插拔使用；
    - 支持“使用全量样本投票”（追求最高精度）或“仅使用前 K 个最近邻投票”（效率优先）；
    - 支持多种距离度量（欧氏、余弦等，底层由 scipy.cdist 提供）。
    """

    def __init__(
        self,
        metric: str = "euclidean",
        use_all_samples: bool = True,
        k_neighbors: Optional[int] = None,
    ) -> None:
        """
        参数:
        - metric: 距离度量方式，支持 'euclidean'、'cosine' 等（由 cdist 支持）。
        - use_all_samples: 是否使用“全量样本”进行投票；为 True 时忽略 k_neighbors。
        - k_neighbors: 若不使用全量样本投票，则取前 K 个最近邻参与投票（K 至少会覆盖 top-k 的需要）。
        """
        self.metric = metric
        self.use_all_samples = use_all_samples
        self.k_neighbors = k_neighbors
        # 特征库存储结构：{class_id: [np.ndarray([Ni, D]), ...]}
        self.class_to_features: Dict[int, List[np.ndarray]] = {}

    def clear(self) -> None:
        """清空整个特征库。"""
        self.class_to_features.clear()

    def add_features(self, features: np.ndarray, labels: np.ndarray) -> None:
        """
        将一批特征加入特征库（按类别归档）。

        参数:
        - features: 形状 [N, D] 的 numpy 数组，表示 N 条样本的 D 维特征。
        - labels:   形状 [N] 的 numpy 数组，样本对应的类别 id（int）。
        """
        unique_labels = np.unique(labels)
        for cls in unique_labels:
            mask = labels == cls
            cls_feats = features[mask]
            if cls not in self.class_to_features:
                self.class_to_features[cls] = []
            # 每个类按批次追加，后续在使用时会 concatenate
            self.class_to_features[cls].append(cls_feats)

    def add_from_loader(self, loader, feature_fn, device: torch.device) -> None:
        """
        从 DataLoader 提取特征并写入特征库。

        参数:
        - loader: 形如 (idx, inputs, targets) 的增量学习训练/缓存数据加载器。
        - feature_fn: 函数，输入 torch.Tensor([B, ...])，输出 torch.Tensor([B, D]) 特征。
        - device: 模型/特征提取所在设备。
        """
        feats, lbs = [], []
        with torch.no_grad():
            for _, inputs, targets in loader:
                inputs = inputs.to(device)
                batch_feats = feature_fn(inputs).detach().cpu().numpy()
                feats.append(batch_feats)
                lbs.append(targets.numpy())
        # 兼容空批次情况
        feats_np = np.concatenate(feats) if len(feats) else np.zeros((0, 0))
        lbs_np = np.concatenate(lbs) if len(lbs) else np.zeros((0,), dtype=np.int64)
        if feats_np.shape[0] > 0:
            self.add_features(feats_np, lbs_np)
        logging.info(
            f"KNN bank updated. Classes in bank: {sorted(list(self.class_to_features.keys()))}"
        )

    def _gather_bank(self):
        """
        将按类别分散存储的特征拼接为两个扁平数组：
        - bank_feats: [M, D]
        - bank_labels: [M]
        其中 M 为特征库中样本总数。
        """
        all_features, all_labels = [], []
        for cls, chunks in self.class_to_features.items():
            if len(chunks) == 0:
                continue
            feats = np.concatenate(chunks)
            labels = np.full((feats.shape[0],), cls, dtype=np.int64)
            all_features.append(feats)
            all_labels.append(labels)
        if len(all_features) == 0:
            return None, None
        return np.concatenate(all_features), np.concatenate(all_labels)

    def predict_topk(self, query_features: np.ndarray, topk: int, total_classes: int) -> np.ndarray:
        """
        使用特征库对查询特征进行 KNN 预测，返回 top-k 类别索引。

        参数:
        - query_features: [N, D] 查询特征矩阵。
        - topk: 返回前 k 个类别。
        - total_classes: 当前已学习的类别总数（用于在样本不足时补齐不重复的类别索引）。

        细节:
        - 若 use_all_samples=True 或未设置 k_neighbors，则用“全量样本投票”；
        - 否则仅用前 K 个最近邻样本投票（K 至少会覆盖 topk）。
        - 同一类别的票数相同（平票）时，用最小距离更近者优先。
        """
        bank_feats, bank_labels = self._gather_bank()
        if bank_feats is None or bank_feats.shape[0] == 0:
            # 当特征库为空：返回形状一致的占位预测（0..topk-1），并裁剪至 total_classes
            fallback = np.arange(min(topk, max(total_classes, 1)), dtype=np.int64)
            return np.tile(fallback, (query_features.shape[0], 1))

        preds: List[List[int]] = []
        for q in query_features:
            # 计算查询与库中所有样本的距离（受 metric 控制）
            # 对于cosine距离，cdist会自动进行L2归一化
            dists = cdist(q.reshape(1, -1), bank_feats, metric=self.metric)[0]
            if self.use_all_samples or self.k_neighbors is None:
                # 全量样本投票：但为了稳定性，仍然使用加权投票（距离越近权重越大）
                # 或者使用所有样本，但按距离加权
                nn_labels = bank_labels
                nn_dists = dists
                # 使用距离的倒数作为权重，距离越小权重越大
                weights = 1.0 / (dists + 1e-8)  # 避免除零
            else:
                # 仅取前 K 个最近邻投票（同时保证 K >= topk）
                K = max(self.k_neighbors, topk)
                idx = np.argsort(dists)[:K]
                nn_labels = bank_labels[idx]
                nn_dists = dists[idx]
                weights = 1.0 / (nn_dists + 1e-8)  # 避免除零

            # 统计每个类别的加权票数与最小距离（用于平票打破）
            votes: Dict[int, float] = {}  # 改为float以支持加权投票
            min_dist: Dict[int, float] = {}
            for lbl, dist, weight in zip(nn_labels, nn_dists, weights):
                votes[lbl] = votes.get(lbl, 0.0) + weight  # 使用加权投票
                if lbl not in min_dist:
                    min_dist[lbl] = dist
                else:
                    min_dist[lbl] = min(min_dist[lbl], dist)

            # 排序：先按加权票数降序，再按最小距离升序
            sorted_classes = sorted(votes.keys(), key=lambda c: (-votes[c], min_dist[c]))
            top = sorted_classes[:topk]
            if len(top) < topk:
                # 若类别数不足，补齐其它未出现的类别索引（不重复），保证输出形状一致
                remaining = [c for c in range(total_classes) if c not in top]
                top.extend(remaining[: topk - len(top)])
            preds.append(top)

        return np.array(preds, dtype=np.int64)


