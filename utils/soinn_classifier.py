"""
Supervised SOINN Classifier (S-SOINN)
可插拔的 SOINN 分类器：
- 在增量任务场景中维护一个"原型库"（prototype bank），跨任务累积训练样本的特征向量。
- 对每个类别单独训练一个简化的 SOINN 网络，生成原型。
- 在推理阶段对查询特征与原型库计算距离，进行基于 1-NN 的类别预测。

设计目标：
- 与不同 Learner 解耦（simplecil、coda_prompt 等），作为通用组件插拔使用；
- 支持增量学习场景，每个任务可以添加新的类别原型。
"""

import logging
from typing import Dict, List, Optional
import numpy as np
import torch
from collections import defaultdict
import math
import random

# 辅助函数：计算余弦距离
def cosine_distance(a, b):
    """
    计算两个向量的余弦距离
    余弦距离 = 1 - 余弦相似度
    返回范围: [0, 2]，0表示完全相同，2表示完全相反
    """
    # 归一化向量
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    # 计算余弦相似度
    cosine_sim = np.dot(a_norm, b_norm)
    # 返回余弦距离
    return 1.0 - cosine_sim

# 辅助函数：归一化向量
def normalize_vector(v):
    """归一化向量到单位长度"""
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return v  # 零向量保持不变
    return v / norm

# 辅助函数：在单位球面上进行插值（球面线性插值，SLERP）
def spherical_interpolate(v1, v2, t):
    """
    在单位球面上进行球面线性插值（Spherical Linear Interpolation, SLERP）
    
    参数:
        v1, v2: 归一化的单位向量
        t: 插值参数 [0, 1]，0返回v1，1返回v2
    
    返回:
        归一化的插值向量
    """
    # 确保输入是归一化的
    v1_norm = normalize_vector(v1)
    v2_norm = normalize_vector(v2)
    
    # 计算两个向量的点积（余弦相似度）
    dot = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    
    # 如果两个向量几乎相同，直接返回
    if abs(dot) > 0.9995:
        return normalize_vector((1 - t) * v1_norm + t * v2_norm)
    
    # 计算角度
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    
    # SLERP 公式
    w1 = np.sin((1 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta
    
    result = w1 * v1_norm + w2 * v2_norm
    return normalize_vector(result)


class _SOINN_per_class:
    """
    内部类: 对单一类别训练的简化 SOINN
    只实现论文中 Adjusted SOINN 的第一层（删除 within-class insertion）
    """

    def __init__(self, ad=20, lam=20, max_nodes=None, seed=None, threshold_scale=0.5):
        """
        参数:
            ad (int): 边最大年龄，边-age > ad 时删除该边
            lam (int): 每 lam 个样本后删除孤立节点（度 <= 1）
            max_nodes (int or None): 若不为 None，限制单类节点最大数
            seed (int or None): 随机种子（用于初始化）
            threshold_scale (float): 阈值缩放因子，用于控制节点插入的宽松程度（默认0.5，越小越容易插入）
        """
        self.ad = int(ad)
        self.lam = int(lam)
        self.max_nodes = None if max_nodes is None else int(max_nodes)
        self.threshold_scale = float(threshold_scale)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # 节点权重列表：每个节点是一个 numpy 向量
        self.nodes = []  # list of np.ndarray
        # 边结构：字典 mapping node_idx -> {neighbor_idx: age, ...}
        self.edges = defaultdict(dict)
        # 节点被选为 winner 的次数（用于可选扩展）
        self.win_count = []

    def fit(self, X):
        """
        使用类内样本 X（shape (N, D)）训练 SOINN（增量式）。
        返回: 无（训练完成后，self.nodes 包含原型）。
        """
        X = np.asarray(X)
        n_samples, dim = X.shape

        # 若样本数少于2，则把样本直接当作节点（归一化）
        if n_samples == 0:
            return
        if n_samples == 1:
            self.nodes = [normalize_vector(X[0].copy())]
            self.win_count = [1]
            return

        # 初始化：随机选两个不同的样本作为初始两个节点（归一化）
        idxs = np.random.choice(n_samples, size=2, replace=False)
        self.nodes = [normalize_vector(X[idxs[0]].copy()), normalize_vector(X[idxs[1]].copy())]
        self.win_count = [1, 1]
        self.edges = defaultdict(dict)

        # 增量学习主循环
        # 目标节点数：样本数的一半
        target_nodes = max(2, n_samples // 2)
        
        for t, x in enumerate(X, start=1):
            # 1) 计算每个节点到 x 的距离，找到 winner(最小) 和 second winner
            # 使用余弦距离（输入样本也需要归一化）
            x_norm = normalize_vector(x)
            dists = np.array([cosine_distance(x_norm, w) for w in self.nodes])
            if len(dists) == 0:
                # 无节点 -> 直接插入（归一化）
                self.nodes.append(x_norm.copy())
                self.win_count.append(1)
                continue
            sorted_idx = np.argsort(dists)
            s1 = sorted_idx[0]
            s2 = sorted_idx[1] if len(sorted_idx) > 1 else sorted_idx[0]

            # 2) 计算相似度阈值 T_s1 和 T_s2（使用邻居最大距或最小全局距）
            T1 = self._similarity_threshold(s1) * self.threshold_scale
            T2 = self._similarity_threshold(s2) * self.threshold_scale
            
            # 动态调整阈值：如果当前节点数少于目标节点数，降低阈值（更容易插入）
            current_node_ratio = len(self.nodes) / max(target_nodes, 1)
            if current_node_ratio < 0.5:
                # 节点数不足目标的一半，大幅降低阈值
                dynamic_scale = 0.3
            elif current_node_ratio < 0.8:
                # 节点数接近目标，适度降低阈值
                dynamic_scale = 0.5
            else:
                # 节点数足够，使用原始阈值
                dynamic_scale = 1.0
            
            T1 *= dynamic_scale
            T2 *= dynamic_scale

            # 3) between-class 插入判断：若距离大于阈值，插入新节点
            # 使用更宽松的条件：只要距离大于任一阈值的缩放版本就插入
            # 或者如果节点数远少于目标，强制插入一些节点
            force_insert = (len(self.nodes) < target_nodes * 0.3) and (t % max(1, n_samples // target_nodes) == 0)
            
            if dists[s1] > T1 or dists[s2] > T2 or force_insert:
                # 插入新的节点 x（归一化）
                # 若设置了 max_nodes，且已达上限，可用替换或跳过插入策略（此处简单跳过插入）
                if (self.max_nodes is None) or (len(self.nodes) < self.max_nodes):
                    self.nodes.append(x_norm.copy())
                    self.win_count.append(1)
                # 插入后不继续更新已有节点（按论文行为）
                continue

            # 4) 如果 s1 与 s2 未连接，建立连接，年龄置 0
            if s2 not in self.edges[s1]:
                self.edges[s1][s2] = 0
                self.edges[s2][s1] = 0
            # 5) 所有从 s1 出发的边的年龄 +1
            # 注意在迭代时不能在原对象上直接修改迭代集合，否则发生并发修改异常
            neighbors_of_s1 = list(self.edges[s1].keys())
            for nbr in neighbors_of_s1:
                self.edges[s1][nbr] += 1
                # 若某条边年龄 > ad，删除该边
                if self.edges[s1][nbr] > self.ad:
                    # 删除两端边
                    try:
                        del self.edges[s1][nbr]
                    except KeyError:
                        pass
                    try:
                        del self.edges[nbr][s1]
                    except KeyError:
                        pass

            # 6) 更新 winner 与其直接邻居的权重（在单位球面上移动到输入样本方向）
            # 使用球面插值（SLERP）在余弦空间中更合理
            eta1 = 1.0 / float(t)          # winner 学习率随时间衰减
            eta2 = 1.0 / (100.0 * float(t)) # 邻居学习率更小
            
            # 对 winner 使用球面插值
            self.nodes[s1] = spherical_interpolate(self.nodes[s1], x_norm, eta1)
            
            # 对邻居也使用球面插值
            for nbr in list(self.edges[s1].keys()):
                self.nodes[nbr] = spherical_interpolate(self.nodes[nbr], x_norm, eta2)
            
            self.win_count[s1] += 1

            # 7) 每 lam 次迭代，删除孤立节点（度 <= 1）
            if (t % self.lam) == 0:
                self._remove_isolated_nodes()

        # 训练完成：可以再次移除孤立节点以清理
        self._remove_isolated_nodes()

    def _similarity_threshold(self, i):
        """
        计算节点 i 的相似度阈值 T_i（使用余弦距离）:
         - 若节点有邻居: T_i = max_j cosine_distance(W_i, W_j) （邻居中最大距离）
         - 若没有邻居: T_i = min_{c != i} cosine_distance(W_i, W_c) （与其他节点最近距离）
        """
        if len(self.nodes) == 0:
            return float('inf')
        if len(self.edges[i]) > 0:
            # 计算到所有邻居的距离，返回最大值
            dmax = 0.0
            for nbr in self.edges[i]:
                d = cosine_distance(self.nodes[i], self.nodes[nbr])
                if d > dmax:
                    dmax = d
            # 若只有一个邻居，dmax 为该邻居距离
            return dmax if dmax > 0 else 1e-8
        else:
            # 若没有邻居，返回与其他节点的最小距离
            if len(self.nodes) == 1:
                return float('inf')
            dmin = float('inf')
            for j, w in enumerate(self.nodes):
                if j == i:
                    continue
                d = cosine_distance(self.nodes[i], w)
                if d < dmin:
                    dmin = d
            return dmin if dmin < float('inf') else 1e-8

    def _remove_isolated_nodes(self):
        """
        删除度 <= 1 的节点（孤立或尾部节点）
        删除时需要重索引节点并重建边结构
        """
        if len(self.nodes) <= 2:
            return

        keep_mask = [True] * len(self.nodes)
        # 计算度并标记需要删除的节点
        for i in range(len(self.nodes)):
            deg = len(self.edges[i])
            if deg <= 1:
                keep_mask[i] = False

        # 如果全部都被标记删除（通常不会发生），则保留至少两个节点
        if all(not k for k in keep_mask):
            # 保留前两个
            keep_mask = [False] * len(self.nodes)
            keep_mask[0] = True
            if len(self.nodes) > 1:
                keep_mask[1] = True

        # 重建节点列表与边
        old_to_new = {}
        new_nodes = []
        new_win_count = []
        for old_idx, keep in enumerate(keep_mask):
            if keep:
                old_to_new[old_idx] = len(new_nodes)
                new_nodes.append(self.nodes[old_idx])
                new_win_count.append(self.win_count[old_idx])

        new_edges = defaultdict(dict)
        for old_i, nbrs in self.edges.items():
            if old_i not in old_to_new:
                continue
            new_i = old_to_new[old_i]
            for old_j, age in nbrs.items():
                if old_j not in old_to_new:
                    continue
                new_j = old_to_new[old_j]
                new_edges[new_i][new_j] = age

        # 确保所有节点都是归一化的（虽然理论上已经是，但为了健壮性）
        self.nodes = [normalize_vector(n.copy()) for n in new_nodes]
        self.win_count = new_win_count[:]
        self.edges = new_edges


class SOINNClassifier:
    """
    Supervised SOINN Classifier (S-SOINN)
    可插拔的 SOINN 分类器：
    - 对每个类别单独训练一个简化的 SOINN（_SOINN_per_class）
    - 合并所有类的节点作为最终原型集
    - 使用 1-NN 进行预测
    """

    def __init__(
        self,
        ad: int = 20,
        lam: int = 20,
        max_nodes_per_class: Optional[int] = None,
        seed: Optional[int] = None,
        threshold_scale: float = 0.5,
        k_neighbors: int = 3,
    ) -> None:
        """
        参数:
            ad (int): 边最大年龄（默认 20）
            lam (int): 删除孤立节点频率（默认 20）
            max_nodes_per_class (int or None): 限制单类节点最大数（可选）
            seed (int or None): 随机种子
            threshold_scale (float): 阈值缩放因子，用于控制节点插入的宽松程度（默认0.5）
            k_neighbors (int): K-NN中的K值，用于预测时投票（默认3）
        """
        self.ad = int(ad)
        self.lam = int(lam)
        self.max_nodes_per_class = None if max_nodes_per_class is None else int(max_nodes_per_class)
        self.seed = seed
        self.threshold_scale = float(threshold_scale)
        self.k_neighbors = int(k_neighbors)

        # 训练后保存的原型与标签
        self.prototypes = None  # numpy array shape (M, D)
        self.prototype_labels = None  # numpy array shape (M,)

        # 每个类的 soinn 模型（若需要检查）
        self._class_models = {}

    def clear(self) -> None:
        """清空整个原型库。"""
        self.prototypes = None
        self.prototype_labels = None
        self._class_models.clear()

    def add_features(self, features: np.ndarray, labels: np.ndarray) -> None:
        """
        将一批特征加入原型库（按类别训练 SOINN 并生成原型）。

        参数:
        - features: 形状 [N, D] 的 numpy 数组，表示 N 条样本的 D 维特征。
        - labels:   形状 [N] 的 numpy 数组，样本对应的类别 id（int）。
        """
        features = np.asarray(features)
        labels = np.asarray(labels)
        
        if features.shape[0] != labels.shape[0]:
            raise ValueError("features 和 labels 的样本数量必须一致")

        unique_labels = np.unique(labels)
        
        # 对每个类别分别训练 SOINN
        for cls in unique_labels:
            mask = labels == cls
            cls_feats = features[mask]
            
            if cls not in self._class_models:
                # 创建新的 SOINN 模型
                model = _SOINN_per_class(
                    ad=self.ad,
                    lam=self.lam,
                    max_nodes=self.max_nodes_per_class,
                    seed=self.seed,
                    threshold_scale=self.threshold_scale
                )
                self._class_models[cls] = model
            else:
                model = self._class_models[cls]
            
            # 增量训练 SOINN（可以多次调用 add_features 来增量添加数据）
            model.fit(cls_feats)

        # 更新原型库
        self._update_prototypes()

    def add_from_loader(self, loader, feature_fn, device: torch.device) -> None:
        """
        从 DataLoader 提取特征并训练 SOINN 生成原型。

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
            f"SOINN bank updated. Classes in bank: {sorted(list(self._class_models.keys()))}"
        )

    def _update_prototypes(self):
        """更新原型库：合并所有类的 SOINN 节点作为最终原型集合"""
        all_protos = []
        all_labels = []

        for cls, model in self._class_models.items():
            if len(model.nodes) == 0:
                continue
            # model.nodes 为 list of np.ndarray
            class_protos = np.vstack([w.reshape(1, -1) for w in model.nodes])
            all_protos.append(class_protos)
            all_labels.extend([cls] * class_protos.shape[0])

        if len(all_protos) == 0:
            # 保持原型为 None，在 predict_topk 中会处理
            self.prototypes = None
            self.prototype_labels = np.array([], dtype=np.int64)
            return

        self.prototypes = np.vstack(all_protos)  # shape (M, D)
        self.prototype_labels = np.array(all_labels, dtype=np.int64)

    def predict_topk(self, query_features: np.ndarray, topk: int, total_classes: int) -> np.ndarray:
        """
        使用训练好的原型进行预测（K-NN投票），返回 top-k 类别索引。

        参数:
        - query_features: [N, D] 查询特征矩阵。
        - topk: 返回前 k 个类别。
        - total_classes: 当前已学习的类别总数（用于在样本不足时补齐不重复的类别索引）。

        返回:
        - [N, topk] 的 numpy 数组，每行为一个查询的 top-k 类别索引。
        """
        if self.prototypes is None or len(self.prototypes) == 0:
            # 当原型库为空：返回形状一致的占位预测（0..topk-1），并裁剪至 total_classes
            fallback = np.arange(min(topk, max(total_classes, 1)), dtype=np.int64)
            return np.tile(fallback, (query_features.shape[0], 1))

        query_features = np.asarray(query_features)
        preds = []
        k = self.k_neighbors
        
        for q in query_features:
            # 计算查询与所有原型的距离（余弦距离）
            # 归一化查询向量和原型
            q_norm = q / (np.linalg.norm(q) + 1e-8)
            prototypes_norm = self.prototypes / (np.linalg.norm(self.prototypes, axis=1, keepdims=True) + 1e-8)
            # 计算余弦相似度
            cosine_sims = np.dot(prototypes_norm, q_norm)
            # 转换为余弦距离（1 - 余弦相似度）
            dists = 1.0 - cosine_sims
            
            # 找到最近的 K 个原型（K-NN）
            k_indices = np.argsort(dists)[:min(k, len(dists))]
            k_labels = self.prototype_labels[k_indices]
            k_dists = dists[k_indices]
            
            # 使用 K-NN 投票：统计每个类别的票数（使用距离加权）
            class_votes = {}  # {label: weighted_vote}
            class_min_dist = {}  # {label: min_distance}
            
            for label, dist in zip(k_labels, k_dists):
                # 使用距离的倒数作为权重（距离越小权重越大）
                weight = 1.0 / (dist + 1e-8)
                if label not in class_votes:
                    class_votes[label] = weight
                    class_min_dist[label] = dist
                else:
                    class_votes[label] += weight
                    class_min_dist[label] = min(class_min_dist[label], dist)
            
            # 如果K-NN中没有覆盖所有类别，补充其他类别的信息（使用最小距离）
            for i, label in enumerate(self.prototype_labels):
                if label not in class_votes:
                    class_min_dist[label] = dists[i]
            
            # 排序：先按加权票数降序，再按最小距离升序
            sorted_classes = sorted(class_votes.keys(), key=lambda c: (-class_votes[c], class_min_dist.get(c, float('inf'))))
            
            # 如果还有其他类别没有出现在K-NN中，按最小距离排序后添加到末尾
            other_classes = [c for c in range(total_classes) if c not in sorted_classes]
            other_classes_sorted = sorted(other_classes, key=lambda c: class_min_dist.get(c, float('inf')))
            sorted_classes.extend(other_classes_sorted)
            
            # 取前 topk 个类别
            top = sorted_classes[:topk]
            
            # 若类别数不足，补齐其它类别索引（不重复），保证输出形状一致
            if len(top) < topk:
                remaining = [c for c in range(total_classes) if c not in top]
                top.extend(remaining[: topk - len(top)])
            
            preds.append(top)

        return np.array(preds, dtype=np.int64)

    def prototypes_per_class(self):
        """便捷方法: 获取每个类生成了多少原型"""
        if self.prototype_labels is None or len(self.prototype_labels) == 0:
            return {}
        unique, counts = np.unique(self.prototype_labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

