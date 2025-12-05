"""
HC-SOINN 分类器（Hierarchical-Cluster SOINN）

- 使用 NCM 维护全局类中心 mu_c（增量均值）
- 类内使用层次聚类生成有限数量的子簇原型（仅在 task 结束时压缩）
- 推理时：先看 NCM，再结合最近子簇中心做融合距离

设计原则：
- 不进行在线新模式/新类发现，压缩仅在任务结束时调用 compress()
- 接口与现有 soinn/esoinn 分类器类似：add_features -> compress -> predict_topk
"""

from typing import Dict, List, Optional
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cdist
import logging


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return v.astype(np.float32, copy=True)
    return (v / norm).astype(np.float32, copy=True)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    return 1.0 - float(np.dot(a_norm, b_norm))


class _Cluster:
    def __init__(self, center: np.ndarray, count: int):
        self.center = _normalize(center)
        self.count = int(count)


class HCSOINNClassifier:
    """
    Hierarchical-Cluster SOINN 分类器
    """

    def __init__(
        self,
        max_prototypes_per_class: Optional[int] = 20,
        alpha: float = 0.5,
        tau_merge: float = 0.2,
        tau_reject: float = 2.0,
        linkage_method: str = "average",
        distance_metric: str = "cosine",
    ) -> None:
        self.max_prototypes_per_class = None if max_prototypes_per_class is None else int(
            max_prototypes_per_class
        )
        self.alpha = float(alpha)
        self.tau_merge = float(tau_merge)
        self.tau_reject = float(tau_reject)
        self.linkage_method = linkage_method
        self.distance_metric = distance_metric

        # 类中心（NCM）与样本计数
        self.class_mu: Dict[int, np.ndarray] = {}
        self.class_count: Dict[int, int] = {}

        # 类内子簇
        self.class_clusters: Dict[int, List[_Cluster]] = {}

        # 任务内缓存特征（仅在 compress 时聚类）
        self.buffers: Dict[int, List[np.ndarray]] = {}

    # ------------------------------------------------------------------ #
    # 数据添加与压缩
    # ------------------------------------------------------------------ #
    def add_features(self, features: np.ndarray, labels: np.ndarray) -> None:
        """
        将当前任务的特征加入缓冲，并更新全局类中心（NCM）
        """
        features = np.asarray(features, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)
        if features.shape[0] == 0:
            return
        for cls in np.unique(labels):
            cls_mask = labels == cls
            cls_feats = features[cls_mask]
            if cls_feats.shape[0] == 0:
                continue

            # 更新缓冲
            if cls not in self.buffers:
                self.buffers[cls] = []
            self.buffers[cls].append(cls_feats)

            # 更新 NCM
            cls_count_old = self.class_count.get(cls, 0)
            cls_sum_old = self.class_mu[cls] * cls_count_old if cls in self.class_mu else 0
            cls_sum_new = cls_sum_old + cls_feats.sum(axis=0)
            cls_count_new = cls_count_old + cls_feats.shape[0]
            self.class_count[cls] = cls_count_new
            self.class_mu[cls] = _normalize(cls_sum_new / float(max(cls_count_new, 1)))

    def compress(self) -> None:
        """
        周期性压缩（通常在每个 task 结束时调用）：
        - 对每个类将缓冲特征（可含旧簇中心）做层次聚类
        - 生成受限数量的子簇原型
        """
        for cls, chunk_list in list(self.buffers.items()):
            if len(chunk_list) == 0:
                continue

            # 聚合缓冲特征
            feats = np.concatenate(chunk_list, axis=0).astype(np.float32, copy=False)

            # 可选：将旧簇中心也纳入，避免完全遗忘已有结构
            if cls in self.class_clusters and len(self.class_clusters[cls]) > 0:
                old_centers = np.stack([c.center for c in self.class_clusters[cls]], axis=0)
                feats = np.concatenate([feats, old_centers], axis=0)

            # 归一化，便于余弦距离
            feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)

            target_k = feats.shape[0] if self.max_prototypes_per_class is None else min(
                self.max_prototypes_per_class, feats.shape[0]
            )

            clusters = self._hierarchical_cluster(feats, target_k)
            clusters = self._merge_close_clusters(clusters, self.tau_merge)

            # 按距离均值排序，保持确定性
            clusters = sorted(clusters, key=lambda c: c.count, reverse=True)
            if self.max_prototypes_per_class is not None:
                clusters = clusters[: self.max_prototypes_per_class]

            self.class_clusters[cls] = clusters
            self.buffers[cls] = []  # 清空缓冲

            logging.info(f"[HC-SOINN] class {cls}: prototypes={len(clusters)}")

    # ------------------------------------------------------------------ #
    # 预测
    # ------------------------------------------------------------------ #
    def predict_topk(
        self, query_features: np.ndarray, topk: int, total_classes: int, device=None
    ) -> np.ndarray:
        """
        返回 shape [N, topk] 的类别索引
        """
        query_features = np.asarray(query_features, dtype=np.float32)
        if query_features.shape[0] == 0:
            return np.zeros((0, topk), dtype=np.int64)

        # 归一化查询
        q = query_features / (np.linalg.norm(query_features, axis=1, keepdims=True) + 1e-8)
        classes = sorted(self.class_mu.keys())
        if len(classes) == 0:
            fallback = np.arange(min(topk, max(total_classes, 1)), dtype=np.int64)
            return np.tile(fallback, (query_features.shape[0], 1))

        preds = []
        for x in q:
            scores = {}
            for cls in classes:
                d_ncm = _cosine_distance(x, self.class_mu[cls])
                d_proto = d_ncm  # 默认退化为 NCM
                cluster_list = self.class_clusters.get(cls, [])
                if len(cluster_list) > 0:
                    centers = np.stack([c.center for c in cluster_list], axis=0)
                    dists = cdist(x[None, :], centers, metric="cosine")[0]
                    d_proto = float(np.min(dists))
                score = self.alpha * d_ncm + (1.0 - self.alpha) * d_proto
                scores[cls] = score

            # 拒识逻辑
            sorted_cls = sorted(scores.keys(), key=lambda c: scores[c])
            top = [c for c in sorted_cls if scores[c] <= self.tau_reject][:topk]
            if len(top) < topk:
                # 若不足，用按分数排序的类补齐（即便超过阈值）
                remaining = [c for c in sorted_cls if c not in top]
                top.extend(remaining[: topk - len(top)])
            preds.append(top)

        return np.asarray(preds, dtype=np.int64)

    # ------------------------------------------------------------------ #
    # 工具函数
    # ------------------------------------------------------------------ #
    def prototypes_per_class(self) -> Dict[int, int]:
        return {c: len(v) for c, v in self.class_clusters.items() if len(v) > 0}

    def _hierarchical_cluster(self, feats: np.ndarray, target_k: int) -> List[_Cluster]:
        if feats.shape[0] == 0:
            return []
        if feats.shape[0] <= target_k:
            return [_Cluster(f, 1) for f in feats]

        Z = linkage(feats, method=self.linkage_method, metric=self.distance_metric)
        cluster_ids = fcluster(Z, t=target_k, criterion="maxclust")

        clusters: Dict[int, List[np.ndarray]] = {}
        for cid in np.unique(cluster_ids):
            clusters[cid] = feats[cluster_ids == cid]

        out: List[_Cluster] = []
        for cid, vectors in clusters.items():
            center = vectors.mean(axis=0)
            out.append(_Cluster(center, vectors.shape[0]))
        return out

    def _merge_close_clusters(self, clusters: List[_Cluster], tau: float) -> List[_Cluster]:
        """
        按阈值合并距离过近的簇（余弦距离）
        """
        if len(clusters) <= 1:
            return clusters

        centers = np.stack([c.center for c in clusters], axis=0)
        dmat = cdist(centers, centers, metric="cosine")

        merged = [False] * len(clusters)
        new_clusters: List[_Cluster] = []

        for i in range(len(clusters)):
            if merged[i]:
                continue
            close_idxs = [i]
            for j in range(i + 1, len(clusters)):
                if dmat[i, j] <= tau:
                    merged[j] = True
                    close_idxs.append(j)
            # 合并 close_idxs
            total_count = sum(clusters[k].count for k in close_idxs)
            weighted_center = sum(clusters[k].center * clusters[k].count for k in close_idxs) / float(
                max(total_count, 1)
            )
            new_clusters.append(_Cluster(weighted_center, total_count))

        return new_clusters


