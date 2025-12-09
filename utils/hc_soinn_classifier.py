"""
HC-SOINN 分类器（Hierarchical-Cluster SOINN）

- 使用 NCM 维护全局类中心 mu_c（增量均值）
- 类内先做层次聚类得到簇中心，然后在簇中心上应用 SOINN 自组织机制
- 推理时：先看 NCM，再结合最近子簇中心做融合距离

设计思路：
- 层次聚类先过滤噪声，得到代表性簇中心
- 在簇中心上应用简化版 SOINN 自组织网络，进行增量学习和动态调整
- 这样既利用了层次聚类的去噪能力，又保留了 SOINN 的自适应特性

优化：
- 推理加速：predict_topk 使用 GPU 矩阵运算替代 Python 循环
- 压缩加速：compress 使用 ProcessPoolExecutor 多进程并行处理各类聚类
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cdist
import logging
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import random


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


def _hierarchical_cluster(feats: np.ndarray, target_k: int, linkage_method: str, distance_metric: str) -> List[_Cluster]:
    if feats.shape[0] == 0:
        return []
    if feats.shape[0] <= target_k:
        return [_Cluster(f, 1) for f in feats]

    try:
        Z = linkage(feats, method=linkage_method, metric=distance_metric)
        cluster_ids = fcluster(Z, t=target_k, criterion="maxclust")
    except Exception:
        # fallback if linkage fails (e.g. too few points)
        return [_Cluster(f, 1) for f in feats]

    clusters: Dict[int, List[np.ndarray]] = {}
    for cid in np.unique(cluster_ids):
        clusters[cid] = feats[cluster_ids == cid]

    out: List[_Cluster] = []
    for cid, vectors in clusters.items():
        center = vectors.mean(axis=0)
        out.append(_Cluster(center, vectors.shape[0]))
    return out


def _merge_close_clusters(clusters: List[_Cluster], tau: float) -> List[_Cluster]:
    """
    按阈值合并距离过近的簇（余弦距离）
    当 use_soinn_refinement=False 时使用
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


def _spherical_interpolate(v1: np.ndarray, v2: np.ndarray, t: float) -> np.ndarray:
    """
    球面线性插值（SLERP）在单位球面上
    """
    v1_norm = _normalize(v1)
    v2_norm = _normalize(v2)
    
    dot = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    
    if abs(dot) > 0.9995:
        return _normalize((1 - t) * v1_norm + t * v2_norm)
    
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    
    w1 = np.sin((1 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta
    
    result = w1 * v1_norm + w2 * v2_norm
    return _normalize(result)


def _simplified_soinn_on_clusters(
    cluster_centers: List[np.ndarray],
    cluster_counts: List[int],
    ad: int = 20,
    lam: int = 20,
    threshold_scale: float = 0.5,
    max_iterations: int = 3,
) -> List[_Cluster]:
    """
    在簇中心上应用简化版 SOINN 自组织机制
    
    参数:
        cluster_centers: 簇中心列表 [M, D]
        cluster_counts: 每个簇的样本计数 [M]
        ad: 边最大年龄
        lam: 每 lam 次迭代删除孤立节点
        threshold_scale: 阈值缩放因子
        max_iterations: 最大迭代轮数（因为簇中心数量少，可以多轮迭代）
    
    返回:
        经过自组织调整后的簇列表
    """
    if len(cluster_centers) == 0:
        return []
    if len(cluster_centers) == 1:
        return [_Cluster(cluster_centers[0], cluster_counts[0])]
    
    # 初始化：所有簇中心作为节点
    nodes = [_normalize(c.copy()) for c in cluster_centers]
    win_counts = cluster_counts.copy()
    edges = defaultdict(dict)
    
    # 初始化边：为每个节点找到最近的邻居建立连接
    n_nodes = len(nodes)
    if n_nodes >= 2:
        for i in range(n_nodes):
            dists = []
            indices = []
            for j in range(n_nodes):
                if j != i:
                    dists.append(_cosine_distance(nodes[i], nodes[j]))
                    indices.append(j)
            if len(dists) > 0:
                nearest_idx = np.argmin(dists)
                j = indices[nearest_idx]
                edges[i][j] = 0
                edges[j][i] = 0
    
    # 自组织迭代：将簇中心作为"样本"输入，进行多轮自组织调整
    # 这里我们使用簇中心本身作为输入，进行多轮迭代以优化节点位置
    all_inputs = cluster_centers.copy()  # 使用原始簇中心作为输入
    
    for iteration in range(max_iterations):
        # 打乱输入顺序
        indices = list(range(len(all_inputs)))
        if len(indices) > 1:
            random.shuffle(indices)
        
        for t, idx in enumerate(indices, start=1):
            x = all_inputs[idx]
            x_norm = _normalize(x)
            
            if len(nodes) < 2:
                continue
            
            # 1) 找到 winner 和 second winner
            dists = np.array([_cosine_distance(x_norm, w) for w in nodes])
            sorted_idx = np.argsort(dists)
            s1 = sorted_idx[0]
            s2 = sorted_idx[1] if len(sorted_idx) > 1 else sorted_idx[0]
            
            # 2) 计算相似度阈值
            def _similarity_threshold(i):
                if len(edges[i]) > 0:
                    dmax = 0.0
                    for nbr in edges[i]:
                        d = _cosine_distance(nodes[i], nodes[nbr])
                        if d > dmax:
                            dmax = d
                    return dmax if dmax > 0 else 1e-8
                else:
                    dmin = float('inf')
                    for j in range(len(nodes)):
                        if j != i:
                            d = _cosine_distance(nodes[i], nodes[j])
                            if d < dmin:
                                dmin = d
                    return dmin if dmin < float('inf') else 1e-8
            
            T1 = _similarity_threshold(s1) * threshold_scale
            T2 = _similarity_threshold(s2) * threshold_scale
            
            # 3) 插入判断（简化：因为簇中心数量已经较少，插入逻辑可以更保守）
            if dists[s1] > T1 or dists[s2] > T2:
                # 如果距离很大，可以考虑插入，但这里我们简化：不插入新节点
                # 因为簇中心已经经过层次聚类筛选，数量应该已经合理
                continue
            
            # 4) 建立/更新边
            if s2 not in edges[s1]:
                edges[s1][s2] = 0
                edges[s2][s1] = 0
            
            # 5) 边老化
            neighbors_of_s1 = list(edges[s1].keys())
            for nbr in neighbors_of_s1:
                edges[s1][nbr] += 1
                edges[nbr][s1] += 1
                if edges[s1][nbr] > ad:
                    del edges[s1][nbr]
                    if s1 in edges[nbr]:
                        del edges[nbr][s1]
            
            # 6) 更新节点权重（使用 SLERP）
            eta1 = 1.0 / float(t + iteration * len(all_inputs) + 1)
            eta2 = 1.0 / (100.0 * float(t + iteration * len(all_inputs) + 1))
            
            nodes[s1] = _spherical_interpolate(nodes[s1], x_norm, eta1)
            for nbr in list(edges[s1].keys()):
                nodes[nbr] = _spherical_interpolate(nodes[nbr], x_norm, eta2)
            
            win_counts[s1] += 1
            
            # 7) 定期删除孤立节点
            if (t % lam) == 0:
                to_remove = []
                for i in range(len(nodes)):
                    if len(edges[i]) <= 1:
                        to_remove.append(i)
                
                # 保留至少 2 个节点
                if len(to_remove) > 0 and len(nodes) - len(to_remove) >= 2:
                    # 重建节点和边
                    keep_mask = [i not in to_remove for i in range(len(nodes))]
                    old_to_new = {}
                    new_nodes = []
                    new_win_counts = []
                    for old_idx, keep in enumerate(keep_mask):
                        if keep:
                            old_to_new[old_idx] = len(new_nodes)
                            new_nodes.append(nodes[old_idx])
                            new_win_counts.append(win_counts[old_idx])
                    
                    new_edges = defaultdict(dict)
                    for old_i, nbrs in edges.items():
                        if old_i not in old_to_new:
                            continue
                        new_i = old_to_new[old_i]
                        for old_j in nbrs.keys():
                            if old_j in old_to_new:
                                new_j = old_to_new[old_j]
                                new_edges[new_i][new_j] = edges[old_i][old_j]
                    
                    nodes = new_nodes
                    win_counts = new_win_counts
                    edges = new_edges
    
    # 最终清理孤立节点
    to_remove = []
    for i in range(len(nodes)):
        if len(edges[i]) == 0:
            to_remove.append(i)
    
    if len(to_remove) > 0 and len(nodes) - len(to_remove) >= 1:
        keep_mask = [i not in to_remove for i in range(len(nodes))]
        nodes = [nodes[i] for i in range(len(nodes)) if keep_mask[i]]
        win_counts = [win_counts[i] for i in range(len(win_counts)) if keep_mask[i]]
    
    # 转换为 _Cluster 对象
    result = []
    for node, count in zip(nodes, win_counts):
        result.append(_Cluster(node, count))
    
    return result


def _compress_class_worker(args):
    """
    Worker function for parallel compression
    args: (cls, feats, target_k, tau_merge, linkage_method, distance_metric, max_prototypes, 
           use_soinn_refinement, soinn_ad, soinn_lam, soinn_threshold_scale, soinn_max_iter)
    """
    (cls, feats, target_k, tau_merge, linkage_method, distance_metric, max_prototypes,
     use_soinn_refinement, soinn_ad, soinn_lam, soinn_threshold_scale, soinn_max_iter) = args
    
    # 归一化
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    
    # 1. 层次聚类得到初始簇中心
    clusters = _hierarchical_cluster(feats, target_k, linkage_method, distance_metric)
    
    # 2. 在簇中心上应用 SOINN 自组织机制（如果启用）
    if use_soinn_refinement and len(clusters) > 1:
        cluster_centers = [c.center for c in clusters]
        cluster_counts = [c.count for c in clusters]
        
        # 应用简化版 SOINN 自组织
        clusters = _simplified_soinn_on_clusters(
            cluster_centers,
            cluster_counts,
            ad=soinn_ad,
            lam=soinn_lam,
            threshold_scale=soinn_threshold_scale,
            max_iterations=soinn_max_iter,
        )
    else:
        # 否则使用简单的距离阈值合并
        clusters = _merge_close_clusters(clusters, tau_merge)
    
    # 按样本计数排序，保持确定性
    clusters = sorted(clusters, key=lambda c: c.count, reverse=True)
    if max_prototypes is not None:
        clusters = clusters[: max_prototypes]
        
    return cls, clusters


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
        use_soinn_refinement: bool = True,
        soinn_ad: int = 20,
        soinn_lam: int = 20,
        soinn_threshold_scale: float = 0.5,
        soinn_max_iter: int = 3,
    ) -> None:
        self.max_prototypes_per_class = None if max_prototypes_per_class is None else int(
            max_prototypes_per_class
        )
        self.alpha = float(alpha)
        self.tau_merge = float(tau_merge)
        self.tau_reject = float(tau_reject)
        self.linkage_method = linkage_method
        self.distance_metric = distance_metric
        
        # SOINN 自组织精炼参数
        self.use_soinn_refinement = bool(use_soinn_refinement)
        self.soinn_ad = int(soinn_ad)
        self.soinn_lam = int(soinn_lam)
        self.soinn_threshold_scale = float(soinn_threshold_scale)
        self.soinn_max_iter = int(soinn_max_iter)

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
        - 使用多进程加速
        """
        tasks = []
        for cls, chunk_list in list(self.buffers.items()):
            if len(chunk_list) == 0:
                continue

            # 聚合缓冲特征
            feats = np.concatenate(chunk_list, axis=0).astype(np.float32, copy=False)

            # 可选：将旧簇中心也纳入，避免完全遗忘已有结构
            if cls in self.class_clusters and len(self.class_clusters[cls]) > 0:
                old_centers = np.stack([c.center for c in self.class_clusters[cls]], axis=0)
                feats = np.concatenate([feats, old_centers], axis=0)

            target_k = feats.shape[0] if self.max_prototypes_per_class is None else min(
                self.max_prototypes_per_class, feats.shape[0]
            )
            
            tasks.append((
                cls, feats, target_k, self.tau_merge, 
                self.linkage_method, self.distance_metric, self.max_prototypes_per_class,
                self.use_soinn_refinement, self.soinn_ad, self.soinn_lam,
                self.soinn_threshold_scale, self.soinn_max_iter
            ))

        # 并行处理
        if tasks:
            logging.info(f"[HC-SOINN] Compressing {len(tasks)} classes using multiprocessing...")
            # 使用 ProcessPoolExecutor 进行并行处理
            # max_workers 默认是 CPU 核心数
            with ProcessPoolExecutor() as executor:
                results = list(executor.map(_compress_class_worker, tasks))
            
            for cls, clusters in results:
                self.class_clusters[cls] = clusters
                self.buffers[cls] = []  # 清空缓冲
                logging.info(f"[HC-SOINN] class {cls}: prototypes={len(clusters)}")
        else:
            logging.info("[HC-SOINN] No buffer to compress.")

    # ------------------------------------------------------------------ #
    # 预测 (GPU 加速版)
    # ------------------------------------------------------------------ #
    def predict_topk(
        self, query_features: np.ndarray, topk: int, total_classes: int, device=None
    ) -> np.ndarray:
        """
        返回 shape [N, topk] 的类别索引
        使用 GPU 矩阵运算加速
        """
        if query_features.shape[0] == 0:
            return np.zeros((0, topk), dtype=np.int64)
            
        if device is None:
            device = torch.device("cpu")

        query_features = np.asarray(query_features, dtype=np.float32)
        
        # 准备 NCM 中心矩阵
        classes = sorted(self.class_mu.keys())
        if len(classes) == 0:
            fallback = np.arange(min(topk, max(total_classes, 1)), dtype=np.int64)
            return np.tile(fallback, (query_features.shape[0], 1))
            
        # 确保涵盖所有可能的类别索引（0 到 total_classes-1）
        # 如果 self.class_mu 中缺少某些类别，我们需要处理，但通常不会发生
        # 为简单起见，我们假设 total_classes 足够大，并使用 self.class_mu 中的类别
        
        # 构建 NCM 中心 Tensor
        ncm_centers = []
        valid_classes = []
        for cls in classes:
            ncm_centers.append(self.class_mu[cls])
            valid_classes.append(cls)
        
        if not ncm_centers:
             return np.zeros((query_features.shape[0], topk), dtype=np.int64)

        ncm_centers_t = torch.from_numpy(np.stack(ncm_centers)).float().to(device) # [C, D]
        query_t = torch.from_numpy(query_features).float().to(device) # [N, D]
        
        # 归一化
        query_t = torch.nn.functional.normalize(query_t, p=2, dim=1)
        ncm_centers_t = torch.nn.functional.normalize(ncm_centers_t, p=2, dim=1)
        
        # 计算 NCM 距离 (1 - cosine)
        # [N, D] @ [D, C] -> [N, C]
        sim_ncm = torch.mm(query_t, ncm_centers_t.t())
        dist_ncm = 1.0 - sim_ncm # [N, C]
        
        # 计算子簇距离
        # 我们需要构建一个大的原型矩阵，并记录每个原型属于哪个类
        all_protos = []
        proto_labels = []
        
        for cls in classes:
            clusters = self.class_clusters.get(cls, [])
            if clusters:
                cls_protos = np.stack([c.center for c in clusters])
                all_protos.append(cls_protos)
                proto_labels.extend([cls] * len(clusters))
            else:
                # 如果某类没有子簇（罕见），暂时用 NCM 中心代替作为子簇
                all_protos.append(self.class_mu[cls][np.newaxis, :])
                proto_labels.append(cls)
                
        if all_protos:
            all_protos_np = np.concatenate(all_protos, axis=0)
            all_protos_t = torch.from_numpy(all_protos_np).float().to(device) # [TotalP, D]
            all_protos_t = torch.nn.functional.normalize(all_protos_t, p=2, dim=1)
            proto_labels_t = torch.tensor(proto_labels, device=device) # [TotalP]
            
            # 计算所有子簇的距离
            # [N, D] @ [D, TotalP] -> [N, TotalP]
            sim_proto = torch.mm(query_t, all_protos_t.t())
            dist_proto_all = 1.0 - sim_proto # [N, TotalP]
            
            # 计算每类的最小子簇距离
            # dist_sub: [N, C]
            dist_sub = torch.full_like(dist_ncm, float('inf'))
            
            # 优化：避免逐类循环可能较慢，但比 Python 循环快。
            # 由于 C 通常不大 (100-1000)，循环是可接受的。
            # 如果 C 很大，可以使用 scatter_reduce_ (需要 torch_scatter) 或特定的 reshape技巧
            
            # 这里使用简单的循环，因为 C 在 CIFAR/ImageNetR 中是 100/200，完全没问题
            for i, cls in enumerate(valid_classes):
                # 找到属于该类的原型索引
                mask = (proto_labels_t == cls)
                if mask.any():
                    # [N, num_protos_of_cls] -> min -> [N]
                    min_d, _ = dist_proto_all[:, mask].min(dim=1)
                    dist_sub[:, i] = min_d
                else:
                    # 如果没有子簇，退化为 NCM 距离
                    dist_sub[:, i] = dist_ncm[:, i]
        else:
            dist_sub = dist_ncm
            
        # 融合分数
        # score = alpha * d_ncm + (1 - alpha) * d_sub
        final_scores = self.alpha * dist_ncm + (1.0 - self.alpha) * dist_sub
        
        # 拒识 (tau_reject) - 将分数过高的置为 inf (或者排在后面)
        # 注意：topk 是找最小距离。如果分数 > tau_reject，我们希望它不被选中，
        # 但如果所有类都 > tau_reject，我们通常还是得选一个最接近的，或者返回 "unknown"。
        # 现有逻辑是填补 remaining。
        # 这里为了保持 batch 操作，我们直接对 final_scores 排序
        
        # [N, C] -> sort -> [N, C]
        # values, indices = torch.topk(final_scores, k=min(topk, len(valid_classes)), dim=1, largest=False)
        
        # 转换回原始类别ID
        # indices 是在 valid_classes 中的索引
        # valid_classes_t = torch.tensor(valid_classes, device=device)
        # top_preds = valid_classes_t[indices]
        
        # 这里的逻辑稍微有点复杂，因为原始代码有 tau_reject 的过滤逻辑。
        # 为了 GPU 加速，我们可以先全部排序，然后 CPU 后处理拒识，或者忽略拒识（因为通常必须输出类别）。
        # 如果严格遵循原始逻辑：
        
        # 将分数转回 CPU 处理 (N 较小，C 较小)
        # 或者直接在 GPU 上做 topk，忽略 tau_reject 的截断特性（只影响顺序吗？不，原逻辑会截断）
        # 原逻辑：top = [c for c in sorted_cls if scores[c] <= tau_reject][:topk]
        # 如果我们不需要严格的“少于 topk 个预测”，可以直接返回 topk。
        # 只有在 open-set 评估或者需要 reject 选项时 tau_reject 才关键。
        # 标准 accuracy 评估不需要 reject。我们这里假设 standard closed-set evaluation。
        
        # 直接 TopK
        k = min(topk, len(valid_classes))
        _, indices = torch.topk(final_scores, k=k, dim=1, largest=False)
        indices = indices.cpu().numpy()
        
        valid_classes_np = np.array(valid_classes)
        top_preds = valid_classes_np[indices] # [N, k]
        
        # 如果 k < topk，补齐
        if k < topk:
            # 这种情况只在总类别数 < topk 时发生
            padding = np.zeros((query_features.shape[0], topk - k), dtype=np.int64)
            # 用第一个有效类补齐，或者 0
            if len(valid_classes) > 0:
                padding[:] = valid_classes[0] 
            top_preds = np.concatenate([top_preds, padding], axis=1)
            
        return top_preds.astype(np.int64)

    # ------------------------------------------------------------------ #
    # 特征漂移对齐支持
    # ------------------------------------------------------------------ #
    def get_class_prototypes_info(self, cls: int, k: int = 5) -> Tuple[np.ndarray, List[int]]:
        """
        获取指定类别中重要性最高的 Top-K 原型中心及其权重（样本计数）
        返回: (centers, counts)
            centers: [K, D]
            counts: [K]
        """
        if cls not in self.class_clusters or len(self.class_clusters[cls]) == 0:
            return np.zeros((0, 0)), []
            
        clusters = self.class_clusters[cls]
        # 按 count 降序排列
        sorted_clusters = sorted(clusters, key=lambda c: c.count, reverse=True)
        top_clusters = sorted_clusters[:k]
        
        centers = np.stack([c.center for c in top_clusters], axis=0)
        counts = [c.count for c in top_clusters]
        
        return centers, counts

    def apply_rigid_transform(
        self, cls: int, R: Optional[np.ndarray], mu_old: np.ndarray, mu_new: np.ndarray, proxy_scale: float = 1.0
    ) -> None:
        """
        对指定类别的所有原型应用刚性变换：
        1. 恢复尺度: W_raw = W_norm * proxy_scale
        2. 变换: W_new_raw = (W_raw - mu_old) @ R + mu_new
           (如果 R is None，则只做平移对齐: W_raw - mean(W_raw) + mu_new)
        3. 归一化: W_new = normalize(W_new_raw)
        
        R: [D, D] 旋转矩阵 (可选)
        mu_old: [D] 旧中心 (锚点中心)
        mu_new: [D] 新中心 (锚点中心)
        proxy_scale: float 旧特征的平均模长
        """
        if cls not in self.class_clusters:
            return
            
        clusters = self.class_clusters[cls]
        if len(clusters) == 0:
            return
        
        # 批量处理 [M, D]
        centers = np.stack([c.center for c in clusters], axis=0)
        
        # 1. 恢复尺度
        centers_restored = centers * proxy_scale
        
        # 2. 变换逻辑
        if R is not None:
            # 有旋转矩阵：(W - mu_old) @ R + mu_new
            # 存在过拟合风险，如果 Anchors 太少
            centers_centered = centers_restored - mu_old
            centers_rotated = np.dot(centers_centered, R)
            centers_new_raw = centers_rotated + mu_new
        else:
            # 无旋转矩阵：仅平移对齐 (Translation Only)
            # 强制将 Cluster 中心对齐到 mu_new
            # 这种方式最稳健，避免了 R 的估计误差，也避免了 mu_old 与 centers 的不匹配
            centers_self_mean = centers_restored.mean(axis=0)
            centers_centered = centers_restored - centers_self_mean
            centers_new_raw = centers_centered + mu_new
        
        # 3. 归一化并更新回对象
        for i, c in enumerate(clusters):
            c.center = _normalize(centers_new_raw[i])
            
        # 同时更新 NCM 的类中心
        if cls in self.class_mu:
            # NCM 只有一个点，直接平移到 mu_new 即可
            # 即使有 R，因为 old_mu ~ mu_old，旋转后也差不多
            # 为了一致性，直接设为 mu_new (归一化)
            # 但为了保留 NCM 可能存在的微小偏差信息（如果有的话），还是走一遍流程
            if R is not None:
                old_mu = self.class_mu[cls]
                old_mu_restored = old_mu * proxy_scale
                new_mu_global = np.dot(old_mu_restored - mu_old, R) + mu_new
            else:
                # 仅平移：直接用 mu_new
                # 因为 NCM 的 old_mu 理论上就应该跟着 mu_new 走
                new_mu_global = mu_new
                
            self.class_mu[cls] = _normalize(new_mu_global)

    # ------------------------------------------------------------------ #
    # 工具函数
    # ------------------------------------------------------------------ #
    def prototypes_per_class(self) -> Dict[int, int]:
        return {c: len(v) for c, v in self.class_clusters.items() if len(v) > 0}
