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

from typing import Dict, List, Optional, Set, Tuple
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
    max_degree_for_removal: int = 1,
) -> List[_Cluster]:
    """
    在簇中心上应用简化版 SOINN 自组织机制
    
    参数:
        cluster_centers: 簇中心列表 [M, D]
        cluster_counts: 每个簇的样本计数 [M]
        ad: 边最大年龄（越小，边越容易被删除，节点越容易被删除）
        lam: 每 lam 次迭代删除孤立节点（在当前实现中，改为每轮迭代结束后删除）
        threshold_scale: 阈值缩放因子（影响插入判断，当前实现中不插入新节点）
        max_iterations: 最大迭代轮数（越多，边老化机会越多，节点越容易被删除）
        max_degree_for_removal: 删除节点的最大度阈值（度 <= max_degree_for_removal 的节点会被删除）
                              默认1（孤立节点或尾部节点），可以设为2使其更激进
    
    返回:
        (clusters, edges): 经过自组织调整后的簇列表和边信息
        - clusters: List[_Cluster]
        - edges: Dict[int, Set[int]] 节点索引到邻居集合的映射
    """
    if len(cluster_centers) == 0:
        return [], {}
    if len(cluster_centers) == 1:
        return [_Cluster(cluster_centers[0], cluster_counts[0])], {}
    
    # 初始化：所有簇中心作为节点
    nodes = [_normalize(c.copy()) for c in cluster_centers]
    win_counts = cluster_counts.copy()
    edges = defaultdict(dict)
    
    # 初始化边：为每个节点找到最近的 1 个邻居建立连接
    # 只连接最近邻居可以减少初始连接数，使边老化后节点更容易被删除，删除更平滑
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
                # 只连接最近的邻居（1条边），使删除更平滑
                sorted_idx = np.argsort(dists)
                nearest_idx = sorted_idx[0]
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
            
            # 5) 边老化（只对 s1 的边进行老化，不要双向更新，避免边被快速删除）
            neighbors_of_s1 = list(edges[s1].keys())
            for nbr in neighbors_of_s1:
                # 只对从 s1 出发的边进行老化
                edges[s1][nbr] += 1
                # 同步更新 nbr 中指向 s1 的边的年龄（保持一致性）
                if s1 in edges[nbr]:
                    edges[nbr][s1] = edges[s1][nbr]  # 同步年龄，而不是 +1
                
                # 若某条边年龄 > ad，删除该边（双向删除）
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
        
        # 7) 在每个迭代轮次结束后，删除孤立节点（度 <= max_degree_for_removal 的节点）
        # 这样可以在每轮迭代结束后清理一次，避免在迭代过程中频繁删除导致不稳定
        if len(nodes) > 2:  # 只有节点数大于2时才考虑删除
            to_remove = []
            for i in range(len(nodes)):
                # 删除度 <= max_degree_for_removal 的节点（孤立节点或尾部节点）
                if len(edges[i]) <= max_degree_for_removal:
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
                
                # 删除节点后，检查是否有节点因为邻居被删除而变成孤立节点（度=0）
                # 为这些节点重新建立连接，确保图的连通性
                if len(nodes) >= 2:
                    for i in range(len(nodes)):
                        if len(edges[i]) == 0:  # 度=0的节点（因为所有邻居都被删除了）
                            # 为这个节点找到最近的邻居并建立连接
                            dists = []
                            indices_new = []
                            for j in range(len(nodes)):
                                if j != i:
                                    dists.append(_cosine_distance(nodes[i], nodes[j]))
                                    indices_new.append(j)
                            if len(dists) > 0:
                                nearest_idx = np.argmin(dists)
                                j = indices_new[nearest_idx]
                                edges[i][j] = 0
                                edges[j][i] = 0
    
    # 最终清理孤立节点
    to_remove = []
    for i in range(len(nodes)):
        if len(edges[i]) == 0:
            to_remove.append(i)
    
    if len(to_remove) > 0 and len(nodes) - len(to_remove) >= 1:
        keep_mask = [i not in to_remove for i in range(len(nodes))]
        nodes = [nodes[i] for i in range(len(nodes)) if keep_mask[i]]
        win_counts = [win_counts[i] for i in range(len(win_counts)) if keep_mask[i]]
    
    # 转换边信息：将节点索引映射到最终的边集合
    # 由于可能删除了孤立节点，需要保留最终的边关系
    final_edges = {}
    for i in range(len(nodes)):
        final_edges[i] = set()
        # edges 字典中可能包含已删除节点的引用，所以只保留最终存在的节点索引
        # 由于我们在删除时重建了索引，edges 应该与最终节点对应
    
    # 重建边的索引映射：由于可能删除了节点，需要映射旧索引到新索引
    # 为了简化，我们直接使用最终的 edges 字典（删除后重建的）
    # 但要注意，最终的 edges 已经是基于最终节点列表的索引
    
    # 由于在删除节点时我们已经重建了 old_to_new 映射，edges 已经更新为新索引
    # 所以直接使用 edges 即可，但需要过滤掉不存在的节点
    final_edges = {}
    for i in range(len(nodes)):
        final_edges[i] = set()
        if i in edges:
            for j in edges[i]:
                if j < len(nodes):  # 确保 j 是有效索引
                    final_edges[i].add(j)
    
    # 转换为 _Cluster 对象
    result = []
    for node, count in zip(nodes, win_counts):
        result.append(_Cluster(node, count))
    
    return result, final_edges


def _compress_class_worker(args):
    """
    Worker function for parallel compression
    args: (cls, feats, target_k, tau_merge, linkage_method, distance_metric, max_prototypes, 
           use_soinn_refinement, soinn_ad, soinn_lam, soinn_threshold_scale, soinn_max_iter,
           soinn_max_degree_for_removal)
    
    Returns:
        (cls, clusters, hierarchical_count, final_count, soinn_edges)
        - cls: 类别编号
        - clusters: 最终的聚类结果
        - hierarchical_count: 层次聚类后的聚类点数量
        - final_count: SOINN 精炼后的最终聚类点数量
        - soinn_edges: Dict[int, Set[int]] 节点索引到邻居集合的映射（仅在 use_soinn_refinement=True 时有效）
    """
    # 在多进程环境中，强制使用非 GUI 后端，避免 tkinter 相关错误
    try:
        import matplotlib
        matplotlib.use('Agg', force=True)  # Agg 是纯后端，不需要 GUI
    except ImportError:
        pass  # 如果没有安装 matplotlib，跳过
    
    (cls, feats, target_k, tau_merge, linkage_method, distance_metric, max_prototypes,
     use_soinn_refinement, soinn_ad, soinn_lam, soinn_threshold_scale, soinn_max_iter,
     soinn_max_degree_for_removal) = args
    
    # 归一化
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    
    # 1. 层次聚类得到初始簇中心
    clusters = _hierarchical_cluster(feats, target_k, linkage_method, distance_metric)
    hierarchical_count = len(clusters)  # 记录层次聚类后的数量
    
    # 2. 在簇中心上应用 SOINN 自组织机制（如果启用）
    if use_soinn_refinement and len(clusters) > 1:
        cluster_centers = [c.center for c in clusters]
        cluster_counts = [c.count for c in clusters]
        
        # 应用简化版 SOINN 自组织
        clusters, soinn_edges = _simplified_soinn_on_clusters(
            cluster_centers,
            cluster_counts,
            ad=soinn_ad,
            lam=soinn_lam,
            threshold_scale=soinn_threshold_scale,
            max_iterations=soinn_max_iter,
            max_degree_for_removal=soinn_max_degree_for_removal,
        )
    else:
        # 否则使用简单的距离阈值合并
        clusters = _merge_close_clusters(clusters, tau_merge)
        soinn_edges = {}  # 没有 SOINN 精炼时没有边信息
    
    # 按样本计数排序，保持确定性
    # 注意：排序会改变节点顺序，需要更新边的索引映射
    if soinn_edges:
        # 创建排序前的索引到排序后的索引的映射
        sorted_indices = sorted(range(len(clusters)), key=lambda i: clusters[i].count, reverse=True)
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_indices)}
        
        # 重新排序 clusters
        clusters = [clusters[i] for i in sorted_indices]
        
        # 更新边的索引映射
        new_edges = {}
        for old_i, neighbors in soinn_edges.items():
            if old_i in old_to_new:
                new_i = old_to_new[old_i]
                new_edges[new_i] = {old_to_new[j] for j in neighbors if j in old_to_new}
        soinn_edges = new_edges
    else:
        clusters = sorted(clusters, key=lambda c: c.count, reverse=True)
    
    if max_prototypes is not None:
        clusters = clusters[: max_prototypes]
        # 更新边索引：如果截断了 clusters，需要更新 soinn_edges
        if soinn_edges and len(clusters) < len(soinn_edges):
            new_edges = {}
            for i in range(len(clusters)):
                if i in soinn_edges:
                    new_edges[i] = {j for j in soinn_edges[i] if j < len(clusters)}
            soinn_edges = new_edges
    
    final_count = len(clusters)  # 记录最终数量
        
    return cls, clusters, hierarchical_count, final_count, soinn_edges


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
        soinn_max_degree_for_removal: int = 1,
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
        self.soinn_max_degree_for_removal = int(soinn_max_degree_for_removal)

        # 类中心（NCM）与样本计数
        self.class_mu: Dict[int, np.ndarray] = {}  # 标准模式：存储完整特征的原型
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

            # 更新完整特征的class_mu（标准模式）
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
                self.soinn_threshold_scale, self.soinn_max_iter, self.soinn_max_degree_for_removal
            ))

        # 并行处理
        if tasks:
            logging.info(f"[HC-SOINN] Compressing {len(tasks)} classes using multiprocessing...")
            # 使用 ProcessPoolExecutor 进行并行处理
            # max_workers 默认是 CPU 核心数
            with ProcessPoolExecutor() as executor:
                results = list(executor.map(_compress_class_worker, tasks))
            
            for cls, clusters, hierarchical_count, final_count, soinn_edges in results:
                self.class_clusters[cls] = clusters
                if not hasattr(self, 'class_edges'):
                    self.class_edges: Dict[int, Dict[int, Set[int]]] = {}
                self.class_edges[cls] = soinn_edges  # 存储边信息
                self.buffers[cls] = []  # 清空缓冲
                
                # 记录层次聚类后的数量和最终数量，方便对比
                if self.use_soinn_refinement:
                    logging.info(
                        f"[HC-SOINN] class {cls}: hierarchical_clusters={hierarchical_count} -> "
                        f"soinn_refined={final_count} (reduction: {hierarchical_count - final_count})"
                    )
                else:
                    logging.info(
                        f"[HC-SOINN] class {cls}: hierarchical_clusters={hierarchical_count} -> "
                        f"merged={final_count} (reduction: {hierarchical_count - final_count})"
                    )
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
        query_dim = query_features.shape[1]
        
        # 标准模式：使用完整的特征
        classes = sorted(self.class_mu.keys())
        if len(classes) == 0:
            fallback = np.arange(min(topk, max(total_classes, 1)), dtype=np.int64)
            return np.tile(fallback, (query_features.shape[0], 1))
        
        # 构建 NCM 中心 Tensor
        ncm_centers = []
        valid_classes = []
        for cls in classes:
            cls_mu = self.class_mu[cls]
            # 只使用维度匹配的类别
            if cls_mu.shape[0] == query_dim:
                ncm_centers.append(cls_mu)
                valid_classes.append(cls)
            else:
                logging.debug(
                    f"Skipping class {cls} in NCM prediction: "
                    f"class_mu dim={cls_mu.shape[0]}, query dim={query_dim}"
                )
        
        if not ncm_centers:
            logging.warning(
                f"No valid NCM centers found for query dim={query_dim}. "
                f"Available class_mu dims: {[self.class_mu[cls].shape[0] for cls in classes]}"
            )
            return np.zeros((query_features.shape[0], topk), dtype=np.int64)

        ncm_centers_t = torch.from_numpy(np.stack(ncm_centers)).float().to(device)
        query_t = torch.from_numpy(query_features).float().to(device)
        
        query_t = torch.nn.functional.normalize(query_t, p=2, dim=1)
        ncm_centers_t = torch.nn.functional.normalize(ncm_centers_t, p=2, dim=1)
        
        sim_ncm = torch.mm(query_t, ncm_centers_t.t())
        dist_ncm = 1.0 - sim_ncm  # [N, C]
        
        # 计算子簇距离
        all_protos = []
        proto_labels = []
        
        for cls in valid_classes:
            clusters = self.class_clusters.get(cls, [])
            if clusters:
                cluster_dims = [c.center.shape[0] for c in clusters]
                if all(dim == query_dim for dim in cluster_dims):
                    cls_protos = np.stack([c.center for c in clusters])
                    all_protos.append(cls_protos)
                    proto_labels.extend([cls] * len(clusters))
            else:
                if cls in self.class_mu and self.class_mu[cls].shape[0] == query_dim:
                    all_protos.append(self.class_mu[cls][np.newaxis, :])
                    proto_labels.append(cls)
        
        if all_protos:
            all_protos_np = np.concatenate(all_protos, axis=0)
            all_protos_t = torch.from_numpy(all_protos_np).float().to(device)
            all_protos_t = torch.nn.functional.normalize(all_protos_t, p=2, dim=1)
            proto_labels_t = torch.tensor(proto_labels, device=device)
            
            sim_proto = torch.mm(query_t, all_protos_t.t())
            dist_proto_all = 1.0 - sim_proto
            
            dist_sub = torch.full_like(dist_ncm, float('inf'))
            for i, cls in enumerate(valid_classes):
                mask = (proto_labels_t == cls)
                if mask.any():
                    min_d, _ = dist_proto_all[:, mask].min(dim=1)
                    dist_sub[:, i] = min_d
                else:
                    dist_sub[:, i] = dist_ncm[:, i]
        else:
            dist_sub = dist_ncm
        
        # 融合分数
        # score = alpha * d_ncm + (1 - alpha) * d_sub
        final_scores = self.alpha * dist_ncm + (1.0 - self.alpha) * dist_sub
        
        # 直接 TopK
        k = min(topk, len(valid_classes))
        _, indices = torch.topk(final_scores, k=k, dim=1, largest=False)
        indices = indices.cpu().numpy()
        
        # 转换回原始类别ID
        valid_classes_t = np.array(valid_classes)
        top_preds = valid_classes_t[indices]  # [N, topk]
        
        return top_preds

    # ------------------------------------------------------------------ #
    # 特征漂移对齐支持
    # ------------------------------------------------------------------ #
    def get_class_prototypes_info(self, cls: int, k: int = 5) -> Tuple[np.ndarray, List[int]]:
        """
        ========================================================================
        STAR 辅助函数：获取指定类别的 Top-K 原型节点信息
        ========================================================================
        
        【功能】
        返回指定类别中重要性最高的 Top-K 个 SOINN 节点中心，用于锚点选择。
        
        【重要性定义】
        - 使用节点的 count 属性（样本计数）作为重要性指标
        - count 越大，说明该节点代表的数据越多，是更重要的"骨架点"
        
        【用途】
        在锚点选择时，我们选择 Top-K 节点对应的最近邻样本作为锚点。
        这样可以保证锚点位于数据流形的"关节"位置，而不是边缘或低密度区域。
        
        【参数】
        - cls: 类别 ID
        - k: 返回的 Top-K 节点数量（默认 5）
        
        【返回】
        - centers: [K, D] Top-K 节点中心（已归一化的单位向量）
        - counts: [K] 每个节点的重要性（样本计数）
        
        【示例】
        假设某类有 20 个节点，count 分别为 [100, 95, 80, 60, 50, ...]
        返回 Top-5: centers=[前5个节点中心], counts=[100, 95, 80, 60, 50]
        """
        if cls not in self.class_clusters or len(self.class_clusters[cls]) == 0:
            return np.zeros((0, 0)), []
            
        clusters = self.class_clusters[cls]
        # 按 count 降序排列（重要性从高到低）
        sorted_clusters = sorted(clusters, key=lambda c: c.count, reverse=True)
        top_clusters = sorted_clusters[:k]  # 取前 k 个
        
        # 提取节点中心和计数
        centers = np.stack([c.center for c in top_clusters], axis=0)  # [K, D]
        counts = [c.count for c in top_clusters]  # [K]
        
        return centers, counts

    def apply_rigid_transform(
        self, cls: int, R: Optional[np.ndarray], mu_old: np.ndarray, mu_new: np.ndarray, scale: float = 1.0, base_scale: float = 1.0
    ) -> None:
        """
        ========================================================================
        STAR 核心函数：应用相似变换到 SOINN 节点 (Apply Similarity Transformation)
        ========================================================================
                    ncm_centers = []
                    valid_adapter_classes = []
                    for cls in adapter_classes:
                        # 优先使用per-adapter的原型
                        if (cls in self.class_mu_per_adapter and 
                            adapter_idx in self.class_mu_per_adapter[cls] and
                            self.class_mu_per_adapter[cls][adapter_idx] is not None):
                            ncm_centers.append(self.class_mu_per_adapter[cls][adapter_idx])
                            valid_adapter_classes.append(cls)
                        elif cls in self.class_mu:
                            # Fallback：使用完整特征的原型，提取对应的adapter段
                            # 注意：根据CL-LoRA的对角NCM逻辑，每个类别只应该使用"属于它的adapter"的原型
                            # 这个fallback主要用于向后兼容，正常情况下应该使用class_mu_per_adapter
                            cls_mu = self.class_mu[cls]
                            if cls_mu.shape[0] == out_dim:
                                # 已经是单个adapter的特征
                                ncm_centers.append(cls_mu)
                                valid_adapter_classes.append(cls)
                            elif cls_mu.shape[0] >= (adapter_idx + 1) * out_dim:
                                # 提取对应的adapter段
                                ncm_centers.append(cls_mu[adapter_idx * out_dim:(adapter_idx + 1) * out_dim])
                                valid_adapter_classes.append(cls)
                        # 其他情况跳过（如果类别在某个adapter上没有原型，就不使用该adapter的特征来预测）
                    
                    if not ncm_centers:
                        adapter_logits = torch.zeros((query_segment.shape[0], end_cls - start_cls), device=device)
                    else:
                        # 计算当前 adapter 的相似度
                        ncm_centers_t = torch.from_numpy(np.stack(ncm_centers)).float().to(device)
                        query_t = torch.from_numpy(query_segment).float().to(device)
                        
                        query_t = torch.nn.functional.normalize(query_t, p=2, dim=1)
                        ncm_centers_t = torch.nn.functional.normalize(ncm_centers_t, p=2, dim=1)
                        
                        # 计算相似度 [N, out_dim] @ [out_dim, C] -> [N, C]
                        sim = torch.mm(query_t, ncm_centers_t.t())
                        
                        # 构建完整的 logits（包含所有 start_cls 到 end_cls 的类别）
                        adapter_logits = torch.full((query_segment.shape[0], end_cls - start_cls), 
                                                   float('-inf'), device=device)
                        for i, cls in enumerate(valid_adapter_classes):
                            if i < len(ncm_centers) and start_cls <= cls < end_cls:
                                adapter_logits[:, cls - start_cls] = sim[:, i]
                
                all_logits.append(adapter_logits)
            
            # 合并所有 adapter 的 logits
            dist_ncm = 1.0 - torch.cat(all_logits, dim=1)  # [N, total_classes]
            valid_classes = list(range(total_classes))
            
        else:
            # 标准模式：使用完整的特征
            classes = sorted(self.class_mu.keys())
            if len(classes) == 0:
                fallback = np.arange(min(topk, max(total_classes, 1)), dtype=np.int64)
                return np.tile(fallback, (query_features.shape[0], 1))
            
            # 构建 NCM 中心 Tensor
            ncm_centers = []
            valid_classes = []
            for cls in classes:
                cls_mu = self.class_mu[cls]
                # 只使用维度匹配的类别
                if cls_mu.shape[0] == query_dim:
                    ncm_centers.append(cls_mu)
                    valid_classes.append(cls)
                else:
                    logging.debug(
                        f"Skipping class {cls} in NCM prediction: "
                        f"class_mu dim={cls_mu.shape[0]}, query dim={query_dim}"
                    )
            
            if not ncm_centers:
                logging.warning(
                    f"No valid NCM centers found for query dim={query_dim}. "
                    f"Available class_mu dims: {[self.class_mu[cls].shape[0] for cls in classes]}"
                )
                return np.zeros((query_features.shape[0], topk), dtype=np.int64)

            ncm_centers_t = torch.from_numpy(np.stack(ncm_centers)).float().to(device)
            query_t = torch.from_numpy(query_features).float().to(device)
            
            query_t = torch.nn.functional.normalize(query_t, p=2, dim=1)
            ncm_centers_t = torch.nn.functional.normalize(ncm_centers_t, p=2, dim=1)
            
            sim_ncm = torch.mm(query_t, ncm_centers_t.t())
            dist_ncm = 1.0 - sim_ncm  # [N, C]
        
        # 计算子簇距离
        if is_cllora_mode:
            # CL-LoRA 模式：按 adapter 分段计算子簇距离
            dist_sub_list = []
            for adapter_idx in range(num_adapters):
                query_segment = query_features[:, adapter_idx * out_dim:(adapter_idx + 1) * out_dim]
                
                if adapter_idx == 0:
                    start_cls = 0
                    end_cls = init_cls
                else:
                    start_cls = init_cls + (adapter_idx - 1) * inc
                    end_cls = start_cls + inc
                
                adapter_classes = [cls for cls in sorted(self.class_clusters.keys()) 
                                 if start_cls <= cls < end_cls]
                
                all_protos = []
                proto_labels = []
                
                for cls in adapter_classes:
                    clusters = self.class_clusters.get(cls, [])
                    if clusters:
                        for c in clusters:
                            if c.center.shape[0] == out_dim:
                                all_protos.append(c.center)
                                proto_labels.append(cls)
                            elif c.center.shape[0] == query_dim:
                                all_protos.append(c.center[adapter_idx * out_dim:(adapter_idx + 1) * out_dim])
                                proto_labels.append(cls)
                
                if all_protos:
                    all_protos_np = np.stack(all_protos)
                    all_protos_t = torch.from_numpy(all_protos_np).float().to(device)
                    query_t_seg = torch.from_numpy(query_segment).float().to(device)
                    
                    all_protos_t = torch.nn.functional.normalize(all_protos_t, p=2, dim=1)
                    query_t_seg = torch.nn.functional.normalize(query_t_seg, p=2, dim=1)
                    
                    sim_proto = torch.mm(query_t_seg, all_protos_t.t())
                    dist_proto_all = 1.0 - sim_proto
                    proto_labels_t = torch.tensor(proto_labels, device=device)
                    
                    # 计算每类的最小子簇距离
                    adapter_dist_sub = torch.full((query_segment.shape[0], end_cls - start_cls), float('inf'), device=device)
                    for cls in adapter_classes:
                        mask = (proto_labels_t == cls)
                        if mask.any():
                            min_d, _ = dist_proto_all[:, mask].min(dim=1)
                            adapter_dist_sub[:, cls - start_cls] = min_d
                        else:
                            # 如果没有子簇，使用 NCM 距离（需要从 dist_ncm 中提取）
                            if cls < dist_ncm.shape[1]:
                                adapter_dist_sub[:, cls - start_cls] = dist_ncm[:, cls]
                    
                    dist_sub_list.append(adapter_dist_sub)
                else:
                    # 没有子簇，使用 NCM 距离
                    adapter_dist_sub = dist_ncm[:, start_cls:end_cls].clone()
                    dist_sub_list.append(adapter_dist_sub)
            
            dist_sub = torch.cat(dist_sub_list, dim=1)  # [N, total_classes]
        else:
            # 标准模式
            all_protos = []
            proto_labels = []
            
            for cls in valid_classes:
                clusters = self.class_clusters.get(cls, [])
                if clusters:
                    cluster_dims = [c.center.shape[0] for c in clusters]
                    if all(dim == query_dim for dim in cluster_dims):
                        cls_protos = np.stack([c.center for c in clusters])
                        all_protos.append(cls_protos)
                        proto_labels.extend([cls] * len(clusters))
                else:
                    if cls in self.class_mu and self.class_mu[cls].shape[0] == query_dim:
                        all_protos.append(self.class_mu[cls][np.newaxis, :])
                        proto_labels.append(cls)
            
            if all_protos:
                all_protos_np = np.concatenate(all_protos, axis=0)
                all_protos_t = torch.from_numpy(all_protos_np).float().to(device)
                all_protos_t = torch.nn.functional.normalize(all_protos_t, p=2, dim=1)
                proto_labels_t = torch.tensor(proto_labels, device=device)
                
                sim_proto = torch.mm(query_t, all_protos_t.t())
                dist_proto_all = 1.0 - sim_proto
                
                dist_sub = torch.full_like(dist_ncm, float('inf'))
                for i, cls in enumerate(valid_classes):
                    mask = (proto_labels_t == cls)
                    if mask.any():
                        min_d, _ = dist_proto_all[:, mask].min(dim=1)
                        dist_sub[:, i] = min_d
                    else:
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
        # """
        # ========================================================================
        # STAR 辅助函数：获取指定类别的 Top-K 原型节点信息
        # ========================================================================
        
        # 【功能】
        # 返回指定类别中重要性最高的 Top-K 个 SOINN 节点中心，用于锚点选择。
        
        # 【重要性定义】
        # - 使用节点的 count 属性（样本计数）作为重要性指标
        # - count 越大，说明该节点代表的数据越多，是更重要的"骨架点"
        
        # 【用途】
        # 在锚点选择时，我们选择 Top-K 节点对应的最近邻样本作为锚点。
        # 这样可以保证锚点位于数据流形的"关节"位置，而不是边缘或低密度区域。
        
        # 【参数】
        # - cls: 类别 ID
        # - k: 返回的 Top-K 节点数量（默认 5）
        
        # 【返回】
        # - centers: [K, D] Top-K 节点中心（已归一化的单位向量）
        # - counts: [K] 每个节点的重要性（样本计数）
        
        # 【示例】
        # 假设某类有 20 个节点，count 分别为 [100, 95, 80, 60, 50, ...]
        # 返回 Top-5: centers=[前5个节点中心], counts=[100, 95, 80, 60, 50]
        # """
        if cls not in self.class_clusters or len(self.class_clusters[cls]) == 0:
            return np.zeros((0, 0)), []
            
        clusters = self.class_clusters[cls]
        # 按 count 降序排列（重要性从高到低）
        sorted_clusters = sorted(clusters, key=lambda c: c.count, reverse=True)
        top_clusters = sorted_clusters[:k]  # 取前 k 个
        
        # 提取节点中心和计数
        centers = np.stack([c.center for c in top_clusters], axis=0)  # [K, D]
        counts = [c.count for c in top_clusters]  # [K]
        
        return centers, counts

    def apply_rigid_transform(
        self, cls: int, R: Optional[np.ndarray], mu_old: np.ndarray, mu_new: np.ndarray, scale: float = 1.0, base_scale: float = 1.0
    ) -> None:
        """
        ========================================================================
        STAR 核心函数：应用相似变换到 SOINN 节点 (Apply Similarity Transformation)
        ========================================================================
        
        【功能】
        将指定类别的所有 SOINN 节点从旧特征空间变换到新特征空间。
        使用相似变换 (Similarity Transformation)：y = s · R · x + t
        
        【数学公式】
        完整变换：W_new = normalize(s · (W_old · base_scale - μ_old) @ R + μ_new)
        
        其中：
            - W_old: [M, D] 旧 SOINN 节点（归一化的单位向量）
            - base_scale: 基准尺度，用于恢复归一化节点的物理尺度
            - μ_old: [D] 旧锚点的中心
            - R: [D, D] 旋转矩阵（正交矩阵）
            - s: 缩放因子（处理 Prompt 导致的幅值变化）
            - μ_new: [D] 新锚点的中心
            - W_new: [M, D] 新 SOINN 节点（归一化的单位向量）
        
        【变换步骤详解】
        Step 1: 恢复尺度 (Scale Restoration)
            W_raw = W_old · base_scale
            - 目的：将归一化的节点恢复到原始特征空间的物理尺度
            - base_scale 是旧特征的平均模长
        
        Step 2: 去中心化 (Centering)
            W_centered = W_raw - μ_old
            - 目的：将节点相对于旧锚点中心进行对齐
        
        Step 3: 旋转 (Rotation)
            W_rotated = W_centered @ R
            - 目的：应用 Procrustes 计算出的旋转矩阵
        
        Step 4: 缩放 (Scaling)
            W_scaled = s · W_rotated
            - 目的：处理特征空间的尺度变化（如 Prompt 导致的幅值变化）
        
        Step 5: 平移 (Translation)
            W_new_raw = W_scaled + μ_new
            - 目的：将节点平移到新锚点的中心
        
        Step 6: 归一化 (Normalization)
            W_new = normalize(W_new_raw)
            - 目的：将节点投影回单位超球面（HC-SOINN 要求）
        
        【参数】
        - cls: 类别 ID
        - R: [D, D] 旋转矩阵（从 Procrustes 计算，None 表示只做平移）
        - mu_old: [D] 旧锚点的中心
        - mu_new: [D] 新锚点的中心
        - scale: 最优缩放因子（从 Procrustes 计算，默认 1.0）
        - base_scale: 基准尺度（旧特征的平均模长，默认 1.0）
        
        【输出】
        - 更新 self.class_clusters[cls] 中所有节点的 center 属性
        - 更新 self.class_mu[cls] (NCM 类中心)
        """
        if cls not in self.class_clusters:
            return
            
        clusters = self.class_clusters[cls]
        if len(clusters) == 0:
            return
        
        # ========== Step 1: 批量提取节点中心 ==========
        # centers: [M, D] 该类所有 SOINN 节点中心（已归一化的单位向量）
        centers = np.stack([c.center for c in clusters], axis=0)
        
        # ========== Step 2: 恢复尺度 ==========
        # 将归一化的节点恢复到原始特征空间的物理尺度
        # 原因：SOINN 节点是归一化的（模长=1），但特征漂移发生在原始欧氏空间
        # base_scale 是旧特征的平均模长，用于恢复尺度
        centers_restored = centers * base_scale  # [M, D]
        
        # ========== Step 3: 应用相似变换 ==========
        if R is not None:
            # ===== 完整相似变换：s · (W - μ_old) @ R + μ_new =====
            
            # 3.1 去中心化：相对于旧锚点中心
            centers_centered = centers_restored - mu_old  # [M, D]
            
            # 3.2 旋转：应用 Procrustes 计算出的旋转矩阵
            centers_rotated = np.dot(centers_centered, R)  # [M, D]
            
            # 3.3 缩放：处理特征空间的尺度变化
            centers_scaled = centers_rotated * scale  # [M, D]
            
            # 3.4 平移：移动到新锚点的中心
            centers_new_raw = centers_scaled + mu_new  # [M, D]
        else:
            # ===== Fallback: 仅平移对齐（当 R 为 None 时）=====
            # 这种情况通常不会发生（因为 STAR 总是计算 R），但保留作为安全机制
            # 策略：只保留拓扑结构（相对位置），强制对齐到新中心
            centers_self_mean = centers_restored.mean(axis=0)  # [D] 节点自身的中心
            centers_centered = centers_restored - centers_self_mean  # 去中心化
            centers_new_raw = centers_centered + mu_new  # 平移到新中心
        
        # ========== Step 4: 归一化并更新节点 ==========
        # 将变换后的节点投影回单位超球面（HC-SOINN 要求节点是归一化的）
        for i, c in enumerate(clusters):
            c.center = _normalize(centers_new_raw[i])  # 更新节点中心
            
        # ========== Step 5: 同时更新 NCM 类中心 ==========
        # NCM 的类中心也需要应用相同的变换，保持一致性
        if cls in self.class_mu:
            if R is not None:
                # 对 NCM 应用相同的相似变换
                old_mu = self.class_mu[cls]  # [D] 旧的 NCM 中心（归一化）
                old_mu_restored = old_mu * base_scale  # 恢复尺度
                new_mu_global = scale * np.dot(old_mu_restored - mu_old, R) + mu_new  # 相似变换
            else:
                # Fallback: 仅平移
                new_mu_global = mu_new
                
            # 归一化并更新 NCM 中心
            self.class_mu[cls] = _normalize(new_mu_global)

    # ------------------------------------------------------------------ #
    # 工具函数
    # ------------------------------------------------------------------ #
    def prototypes_per_class(self) -> Dict[int, int]:
        return {c: len(v) for c, v in self.class_clusters.items() if len(v) > 0}
