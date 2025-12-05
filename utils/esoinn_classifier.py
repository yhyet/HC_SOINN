import logging
from typing import Dict, Optional

import numpy as np
import torch
from scipy.spatial.distance import cdist


def _normalize_vector(v: np.ndarray) -> np.ndarray:
    """归一化到单位长度（零向量则保持不变）"""
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return v.astype(np.float32, copy=True)
    return (v / norm).astype(np.float32, copy=True)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """余弦距离 = 1 - cos_sim，范围约 [0, 2]"""
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    cos_sim = float(np.dot(a_norm, b_norm))
    return 1.0 - cos_sim


class _ESOINNNode:
    """ESOINN 单个节点结构（在单位球面上存储权重向量）"""

    def __init__(self, weight: np.ndarray) -> None:
        # 所有权重统一归一化，便于以余弦距离工作
        self.weight = _normalize_vector(weight)
        self.density = 0.0  # 累积密度信号
        self.win_count = 0  # 胜利次数


class _ESOINNPerClass:
    """
    无监督 ESOINN 核心（单一类别）

    - 采用余弦距离（cosine distance）
    - 使用 OR 规则进行新节点插入
    - 周期性删除老边与噪声节点
    """

    def __init__(
        self,
        dim: int,
        max_edge_age: int = 50,
        iter_threshold: int = 100,
        c1: float = 0.001,
    ) -> None:
        self.dim = dim
        self.max_edge_age = int(max_edge_age)
        self.iter_threshold = int(iter_threshold)
        self.c1 = float(c1)

        # 核心结构
        self.nodes: Dict[int, _ESOINNNode] = {}
        self.edges: Dict[int, Dict[int, int]] = {}
        self._next_node_id = 0
        self._signals_processed = 0

    # --------- 内部工具函数 ---------
    def _get_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        # 统一使用余弦距离
        return _cosine_distance(x, y)

    def _find_winners(self, x: np.ndarray):
        """寻找第一赢家(s1)和第二赢家(s2)，返回 (s1_id, s2_id, dist_s1)"""
        if len(self.nodes) < 2:
            return None, None, None

        ids = list(self.nodes.keys())
        weights = np.stack([self.nodes[i].weight for i in ids], axis=0)
        # 使用余弦距离（scipy 内部自动做归一化）
        dists = cdist(x[None, :], weights, metric="cosine")[0]

        order = np.argsort(dists)
        s1_id = ids[order[0]]
        s2_id = ids[order[1]]
        dist_s1 = float(dists[order[0]])
        return s1_id, s2_id, dist_s1

    def _similarity_threshold(self, node_id: int) -> float:
        """
        计算节点的相似度阈值 T_i
        - 若有邻居: T_i = 邻居中最大距离
        - 否则: T_i = 与其他所有节点最近的距离
        """
        if node_id not in self.nodes:
            return float("inf")

        neighbors = list(self.edges.get(node_id, {}).keys())
        w = self.nodes[node_id].weight

        if len(neighbors) > 0:
            neighbor_weights = np.stack([self.nodes[n].weight for n in neighbors], axis=0)
            dists = cdist(w[None, :], neighbor_weights, metric="cosine")[0]
            return float(np.max(dists))
        else:
            other_ids = [i for i in self.nodes.keys() if i != node_id]
            if not other_ids:
                return float("inf")
            other_weights = np.stack([self.nodes[i].weight for i in other_ids], axis=0)
            dists = cdist(w[None, :], other_weights, metric="cosine")[0]
            return float(np.min(dists))

    # --------- 训练单个样本 ---------
    def train_single(self, x: np.ndarray) -> None:
        self._signals_processed += 1
        # 训练时也先归一化到单位球面
        x = _normalize_vector(x.astype(np.float32, copy=False))

        # 初始化阶段：少于 2 个节点时直接插入
        if len(self.nodes) < 2:
            node = _ESOINNNode(x)
            self.nodes[self._next_node_id] = node
            self.edges[self._next_node_id] = {}
            self._next_node_id += 1
            return

        # 1) 找 winner / second winner
        s1, s2, dist_s1 = self._find_winners(x)
        if s1 is None or s2 is None:
            # 理论上不会发生，但做下保护
            node = _ESOINNNode(x)
            self.nodes[self._next_node_id] = node
            self.edges[self._next_node_id] = {}
            self._next_node_id += 1
            return

        # 2) 阈值
        T1 = self._similarity_threshold(s1)
        T2 = self._similarity_threshold(s2)

        # 3) 新节点判定（OR 规则）
        if dist_s1 > T1 or dist_s1 > T2:
            node = _ESOINNNode(x)
            self.nodes[self._next_node_id] = node
            self.edges[self._next_node_id] = {}
            self._next_node_id += 1
            return

        # 4) 更新边年龄、连接 s1-s2
        if s1 not in self.edges:
            self.edges[s1] = {}
        if s2 not in self.edges:
            self.edges[s2] = {}

        # 所有与 s1 相连的边年龄 +1
        for nbr in list(self.edges[s1].keys()):
            self.edges[s1][nbr] += 1
            self.edges[nbr][s1] += 1

        # 确保 s1 与 s2 有连边
        self.edges[s1][s2] = 0
        self.edges[s2][s1] = 0

        # 5) 更新 s1 密度、权重及邻居权重
        node_s1 = self.nodes[s1]
        node_s1.density += 1.0 / (1.0 + self._get_distance(x, node_s1.weight))
        node_s1.win_count += 1

        lr_winner = 1.0 / float(node_s1.win_count)
        node_s1.weight += lr_winner * (x - node_s1.weight)

        # 邻居微小移动
        for nbr in self.edges[s1].keys():
            if nbr == s1:
                continue
            node_n = self.nodes[nbr]
            # 防止 win_count 为 0
            win = max(1, node_n.win_count)
            node_n.weight += (self.c1 / float(win)) * (x - node_n.weight)

        # 6) 周期性清理
        if (self._signals_processed % self.iter_threshold) == 0:
            self._remove_old_edges()
            self._remove_noise()

    # --------- 结构清理 ---------
    def _remove_old_edges(self) -> None:
        # 删除超过最大年龄的边
        for i in list(self.edges.keys()):
            for j in list(self.edges[i].keys()):
                if self.edges[i][j] > self.max_edge_age:
                    del self.edges[i][j]
                    if j in self.edges and i in self.edges[j]:
                        del self.edges[j][i]

        # 删除没有任何边的孤立节点
        to_remove = [i for i in self.nodes.keys() if len(self.edges.get(i, {})) == 0]
        for i in to_remove:
            self.nodes.pop(i, None)
            self.edges.pop(i, None)

    def _remove_noise(self) -> None:
        """基于平均胜利次数和度数的简单去噪"""
        if not self.nodes:
            return

        total_wins = sum(node.win_count for node in self.nodes.values())
        avg_wins = total_wins / float(len(self.nodes)) if self.nodes else 0.0

        to_remove = []
        for i, node in self.nodes.items():
            deg = len(self.edges.get(i, {}))
            if deg <= 2 and node.win_count < 0.5 * avg_wins:
                to_remove.append(i)

        for i in to_remove:
            if i in self.edges:
                for nbr in list(self.edges[i].keys()):
                    if nbr in self.edges and i in self.edges[nbr]:
                        del self.edges[nbr][i]
                del self.edges[i]
            self.nodes.pop(i, None)

    # --------- 导出原型 ---------
    def get_prototypes(self) -> np.ndarray:
        if not self.nodes:
            return np.zeros((0, self.dim), dtype=np.float32)
        # 按 id 排序，保证稳定性
        ids = sorted(self.nodes.keys())
        return np.stack([self.nodes[i].weight for i in ids], axis=0)


class ESOINNClassifier:
    """
    Supervised ESOINN Classifier

    - 对每个类别训练一个 ESOINN（_ESOINNPerClass）
    - 合并所有类的节点作为最终原型集
    - 使用 1-NN / 最近原型进行 top-k 预测
    - 与 KNN / SOINN 分类器类似，可作为增量学习中的“可插拔分类头”
    """

    def __init__(
        self,
        max_edge_age: int = 50,
        iter_threshold: int = 100,
        c1: float = 0.001,
    ) -> None:
        self.max_edge_age = int(max_edge_age)
        self.iter_threshold = int(iter_threshold)
        self.c1 = float(c1)

        # 每个类的 ESOINN 模型
        self._class_models: Dict[int, _ESOINNPerClass] = {}

        # 聚合后的原型库
        self.prototypes: Optional[np.ndarray] = None  # [M, D]
        self.prototype_labels: Optional[np.ndarray] = None  # [M]

    # --------- 管理接口 ---------
    def clear(self) -> None:
        """清空整个原型库与内部 ESOINN 模型。"""
        self._class_models.clear()
        self.prototypes = None
        self.prototype_labels = None

    def add_features(self, features: np.ndarray, labels: np.ndarray) -> None:
        """
        使用一批特征更新 ESOINN 原型库（可多次调用以增量添加）。

        参数:
            features: [N, D] numpy array
            labels:   [N]    numpy array (int64)
        """
        features = np.asarray(features)
        labels = np.asarray(labels)
        if features.shape[0] == 0:
            return
        if features.shape[0] != labels.shape[0]:
            raise ValueError("features 和 labels 的样本数量必须一致")

        dim = features.shape[1]
        unique_labels = np.unique(labels)

        for cls in unique_labels:
            mask = labels == cls
            cls_feats = features[mask].astype(np.float32)
            if cls not in self._class_models:
                self._class_models[cls] = _ESOINNPerClass(
                    dim=dim,
                    max_edge_age=self.max_edge_age,
                    iter_threshold=self.iter_threshold,
                    c1=self.c1,
                )
            model = self._class_models[cls]

            # 将该类样本打乱，模拟在线输入顺序
            idx_perm = np.random.permutation(cls_feats.shape[0])
            for x in cls_feats[idx_perm]:
                model.train_single(x)

        self._update_prototypes()

    def add_from_loader(self, loader, feature_fn, device: torch.device) -> None:
        """
        从 DataLoader 中提取特征并更新 ESOINN 原型库。

        参数:
            loader: 形如 (idx, inputs, targets) 的增量学习数据加载器
            feature_fn: 输入 torch.Tensor([B, ...]) -> torch.Tensor([B, D])
            device: 特征提取运行设备
        """
        feats, lbs = [], []
        with torch.no_grad():
            for _, inputs, targets in loader:
                inputs = inputs.to(device)
                batch_feats = feature_fn(inputs).detach().cpu().numpy()
                feats.append(batch_feats)
                lbs.append(targets.numpy())

        feats_np = np.concatenate(feats) if len(feats) else np.zeros((0, 0), dtype=np.float32)
        lbs_np = np.concatenate(lbs) if len(lbs) else np.zeros((0,), dtype=np.int64)
        if feats_np.shape[0] > 0:
            self.add_features(feats_np, lbs_np)

        logging.info(
            f"ESOINN bank updated. Classes in bank: {sorted(list(self._class_models.keys()))}"
        )

    def _update_prototypes(self) -> None:
        """合并所有类的 ESOINN 节点作为最终原型集合。"""
        all_protos = []
        all_labels = []

        for cls, model in self._class_models.items():
            protos = model.get_prototypes()
            if protos.shape[0] == 0:
                continue
            all_protos.append(protos)
            all_labels.extend([cls] * protos.shape[0])

        if len(all_protos) == 0:
            self.prototypes = None
            self.prototype_labels = None
            return

        self.prototypes = np.concatenate(all_protos, axis=0).astype(np.float32)
        self.prototype_labels = np.asarray(all_labels, dtype=np.int64)

    # --------- 预测接口 ---------
    def predict_topk(
        self,
        query_features: np.ndarray,
        topk: int,
        total_classes: int,
        device: Optional[torch.device] = None,
    ) -> np.ndarray:
        """
        使用 ESOINN 原型进行最近原型分类，返回 top-k 类别。

        为简化起见，目前实现为 CPU 上的欧式距离计算，device 参数暂未使用。
        """
        if self.prototypes is None or self.prototype_labels is None or len(self.prototype_labels) == 0:
            # 原型库为空：返回占位预测（0..topk-1），并裁剪至 total_classes
            fallback = np.arange(min(topk, max(total_classes, 1)), dtype=np.int64)
            return np.tile(fallback, (query_features.shape[0], 1))

        query_features = np.asarray(query_features, dtype=np.float32)
        # 查询特征也做归一化，保持与节点一致
        if query_features.ndim == 2 and query_features.shape[0] > 0:
            norms = np.linalg.norm(query_features, axis=1, keepdims=True)
            query_features = query_features / (norms + 1e-8)
        N = query_features.shape[0]
        protos = self.prototypes
        labels = self.prototype_labels

        # [N, M] 余弦距离
        dists = cdist(query_features, protos, metric="cosine")

        preds = []
        for i in range(N):
            # 每个类别的最小原型距离
            class_min_dist: Dict[int, float] = {}
            for j, cls in enumerate(labels):
                d = float(dists[i, j])
                if cls not in class_min_dist or d < class_min_dist[cls]:
                    class_min_dist[cls] = d

            # 按距离从小到大排序
            sorted_classes = sorted(class_min_dist.keys(), key=lambda c: class_min_dist[c])
            top = sorted_classes[:topk]

            # 若可选类别不足 topk，用其他类别补齐
            if len(top) < topk:
                remaining = [c for c in range(total_classes) if c not in top]
                top.extend(remaining[: topk - len(top)])

            preds.append(top)

        return np.asarray(preds, dtype=np.int64)

    def prototypes_per_class(self) -> Dict[int, int]:
        """便捷统计：每个类的原型数"""
        if self.prototype_labels is None or len(self.prototype_labels) == 0:
            return {}
        unique, counts = np.unique(self.prototype_labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))



