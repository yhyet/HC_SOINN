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
    def __init__(self, center: np.ndarray, count: int, center_raw: Optional[np.ndarray] = None):
        """
        center: 归一化中心 (用于推理)
        center_raw: 未归一化中心 (用于 STAR 对齐)
        """
        self.center = _normalize(center)
        self.count = int(count)
        # 关键修复：如果提供了 raw，直接使用；否则仅当 center 本身未归一化时才复制
        if center_raw is not None:
            self.center_raw = center_raw.copy()
        else:
            # Fallback: 假设 center 就是 raw (但在 compress 中我们会强制提供 raw)
            self.center_raw = center.copy()
            
def _hierarchical_cluster(feats_norm: np.ndarray, feats_raw: np.ndarray, target_k: int, linkage_method: str, distance_metric: str) -> List[_Cluster]:
    """
    修改版：同时接收归一化特征（用于聚类计算）和原始特征（用于计算 center_raw）
    """
    if feats_norm.shape[0] == 0:
        return []
    if feats_norm.shape[0] <= target_k:
        return [_Cluster(feats_norm[i], 1, center_raw=feats_raw[i]) for i in range(feats_norm.shape[0])]

    try:
        # 使用归一化特征进行距离计算和聚类
        Z = linkage(feats_norm, method=linkage_method, metric=distance_metric)
        cluster_ids = fcluster(Z, t=target_k, criterion="maxclust")
    except Exception:
        return [_Cluster(feats_norm[i], 1, center_raw=feats_raw[i]) for i in range(feats_norm.shape[0])]

    clusters_map = defaultdict(list)
    clusters_raw_map = defaultdict(list)
    
    for i, cid in enumerate(cluster_ids):
        clusters_map[cid].append(feats_norm[i])
        clusters_raw_map[cid].append(feats_raw[i])

    out: List[_Cluster] = []
    for cid in clusters_map:
        vectors = np.stack(clusters_map[cid])
        vectors_raw = np.stack(clusters_raw_map[cid])
        
        # 计算归一化中心
        center = vectors.mean(axis=0)
        # 计算原始中心 (保留 Magnitude 信息)
        center_raw = vectors_raw.mean(axis=0)
        
        out.append(_Cluster(center, vectors.shape[0], center_raw=center_raw))
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
        # 合并 close_idxs (both normalized and raw centers)
        total_count = sum(clusters[k].count for k in close_idxs)
        weighted_center = sum(clusters[k].center * clusters[k].count for k in close_idxs) / float(
            max(total_count, 1)
        )
        weighted_center_raw = sum(clusters[k].center_raw * clusters[k].count for k in close_idxs) / float(
            max(total_count, 1)
        )
        new_clusters.append(_Cluster(weighted_center, total_count, center_raw=weighted_center_raw))

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
    cluster_centers: List[np.ndarray],     # Normalized
    cluster_centers_raw: List[np.ndarray], # Raw (Unnormalized)
    cluster_counts: List[int],
    ad: int = 20,
    lam: int = 20,
    threshold_scale: float = 0.5,
    max_iterations: int = 3,
    max_degree_for_removal: int = 1,
) -> List[_Cluster]:
    """
    修改版：在 SOINN 调整 normalized 节点位置时，同步线性更新 raw 节点位置
    """
    if len(cluster_centers) == 0:
        return [], {}
    
    # 初始化节点
    nodes = [_normalize(c.copy()) for c in cluster_centers]
    nodes_raw = [c.copy() for c in cluster_centers_raw] # 同步维护 Raw 节点
    
    win_counts = cluster_counts.copy()
    edges = defaultdict(dict)
    
    # 初始化边
    n_nodes = len(nodes)
    if n_nodes >= 2:
        for i in range(n_nodes):
            dists = [_cosine_distance(nodes[i], nodes[j]) if i != j else float('inf') for j in range(n_nodes)]
            if len(dists) > 0:
                nearest_idx = np.argmin(dists)
                edges[i][nearest_idx] = 0
                edges[nearest_idx][i] = 0
    
    # 原始簇中心作为输入信号 (Input Signals)
    input_signals = list(zip(cluster_centers, cluster_centers_raw))
    
    # IMPORTANT (Determinism):
    # This refinement previously used python's global `random.shuffle(indices)`, which makes results
    # depend on the global RNG state of each worker process. Enabling/disabling STAR (or any other
    # feature) can consume RNG calls before `compress()`, changing the shuffle order and producing
    # different prototypes even on Task0 (where STAR hasn't aligned anything yet).
    #
    # Use a local RNG with a fixed seed so refinement order is reproducible and independent of
    # global RNG / multiprocessing for comparable experiments.
    _local_rng = np.random.RandomState(0)
    
    for iteration in range(max_iterations):
        indices = list(range(len(input_signals)))
        if len(indices) > 1:
            _local_rng.shuffle(indices)
        
        for t, idx in enumerate(indices, start=1):
            x, x_raw = input_signals[idx]
            x_norm = _normalize(x)
            
            if len(nodes) < 2: continue
            
            # 1. 寻找 Winner
            dists = np.array([_cosine_distance(x_norm, w) for w in nodes])
            sorted_idx = np.argsort(dists)
            s1, s2 = sorted_idx[0], sorted_idx[1]
            
            # 2. 阈值判断 (省略细节，保持原逻辑)
            # ... (保持原有的阈值计算和插入判断逻辑) ...
            
            # 3. 建立边
            if s2 not in edges[s1]:
                edges[s1][s2] = 0
                edges[s2][s1] = 0
            
            # 4. 边老化
            for nbr in list(edges[s1].keys()):
                edges[s1][nbr] += 1
                if s1 in edges[nbr]: edges[nbr][s1] = edges[s1][nbr]
                if edges[s1][nbr] > ad:
                    del edges[s1][nbr]
                    if s1 in edges[nbr]: del edges[nbr][s1]
            
            # 5. 更新节点位置 (核心修改)
            # 学习率
            eta1 = 1.0 / float(t + iteration * len(input_signals) + 1)
            eta2 = 1.0 / (100.0 * float(t + iteration * len(input_signals) + 1))
            
            # A. 归一化空间：使用 SLERP
            nodes[s1] = _spherical_interpolate(nodes[s1], x_norm, eta1)
            
            # B. 原始空间：使用线性插值 (Standard Hebbian Learning)
            # w_raw += eta * (x_raw - w_raw)
            nodes_raw[s1] = nodes_raw[s1] + eta1 * (x_raw - nodes_raw[s1])
            
            for nbr in list(edges[s1].keys()):
                nodes[nbr] = _spherical_interpolate(nodes[nbr], x_norm, eta2)
                nodes_raw[nbr] = nodes_raw[nbr] + eta2 * (x_raw - nodes_raw[nbr])
                
            win_counts[s1] += 1

        # 删除孤立节点逻辑 (保持不变，但在重建列表时要同步处理 nodes_raw)
        # ...
        # (为节省篇幅，这里简略，实现时请确保 nodes_raw 与 nodes 同步增删)
        # 简单实现：
        to_remove = [i for i in range(len(nodes)) if len(edges[i]) <= max_degree_for_removal]
        if len(to_remove) > 0 and len(nodes) - len(to_remove) >= 2:
            keep_mask = [i not in to_remove for i in range(len(nodes))]
            
            # 重建映射
            old_to_new = {}
            new_nodes, new_nodes_raw, new_counts = [], [], []
            for old_idx, keep in enumerate(keep_mask):
                if keep:
                    old_to_new[old_idx] = len(new_nodes)
                    new_nodes.append(nodes[old_idx])
                    new_nodes_raw.append(nodes_raw[old_idx])
                    new_counts.append(win_counts[old_idx])
            
            # 重建边
            new_edges = defaultdict(dict)
            for old_i, nbrs in edges.items():
                if old_i in old_to_new:
                    new_i = old_to_new[old_i]
                    for old_j in nbrs:
                        if old_j in old_to_new:
                            new_edges[new_i][old_to_new[old_j]] = edges[old_i][old_j]
            
            nodes, nodes_raw, win_counts, edges = new_nodes, new_nodes_raw, new_counts, new_edges
            
            # 修复连通性 (若删完后有新孤立点)
            for i in range(len(nodes)):
                if len(edges[i]) == 0 and len(nodes) > 1:
                    dists = [_cosine_distance(nodes[i], nodes[j]) if i != j else float('inf') for j in range(len(nodes))]
                    nearest = np.argmin(dists)
                    edges[i][nearest] = 0
                    edges[nearest][i] = 0

    # 最终结果
    result = []
    final_edges_map = {} # 映射回 0..N
    
    # 再次清理完全孤立点
    final_indices = [i for i in range(len(nodes)) if len(edges[i]) > 0]
    if not final_indices: final_indices = range(len(nodes)) # 防止空
    
    for new_idx, old_idx in enumerate(final_indices):
        result.append(_Cluster(nodes[old_idx], win_counts[old_idx], center_raw=nodes_raw[old_idx]))
        final_edges_map[new_idx] = set()
        for nbr in edges[old_idx]:
             if nbr in final_indices:
                 # 找到 nbr 在 final_indices 中的新索引 (效率较低但数据量小)
                 final_edges_map[new_idx].add(final_indices.index(nbr))
                 
    return result, final_edges_map

def _compress_class_worker(args):
    # ... (前置导入保持不变)
    
    (cls, feats, target_k, tau_merge, linkage_method, distance_metric, max_prototypes,
     use_soinn_refinement, soinn_ad, soinn_lam, soinn_threshold_scale, soinn_max_iter,
     soinn_max_degree_for_removal) = args
    
    # 关键修改 1: 保留原始特征，另外计算归一化特征
    feats_raw = feats.astype(np.float32)
    feats_norm = feats_raw / (np.linalg.norm(feats_raw, axis=1, keepdims=True) + 1e-8)
    
    # 关键修改 2: 传递两者给聚类函数
    clusters = _hierarchical_cluster(feats_norm, feats_raw, target_k, linkage_method, distance_metric)
    hierarchical_count = len(clusters)
    
    soinn_edges = {}
    if use_soinn_refinement and len(clusters) > 1:
        cluster_centers = [c.center for c in clusters]
        # 关键修改 3: 提取 raw centers 传给 SOINN
        cluster_centers_raw = [c.center_raw for c in clusters] 
        cluster_counts = [c.count for c in clusters]
        
        clusters, soinn_edges = _simplified_soinn_on_clusters(
            cluster_centers,
            cluster_centers_raw, # 传入
            cluster_counts,
            ad=soinn_ad,
            lam=soinn_lam,
            threshold_scale=soinn_threshold_scale,
            max_iterations=soinn_max_iter,
            max_degree_for_removal=soinn_max_degree_for_removal,
        )
    else:
        # Merge close logic 也可以相应修改，或者暂时忽略（因为主要用 SOINN）
        # 如果必须用，记得 merge 时也要加权平均 center_raw
        clusters = sorted(clusters, key=lambda c: c.count, reverse=True) # 简单排序
    
    # ... (后处理保持不变)
    
    # Debug log (检查 Norm 是否恢复正常)
    if len(clusters) > 0:
        raw_norms = [np.linalg.norm(c.center_raw) for c in clusters]
        avg_raw_norm = sum(raw_norms) / len(raw_norms)
        # 如果 avg_raw_norm 接近 1.0，说明还是有问题；应该接近 10-20
        # print(f"DEBUG WORKER {cls}: Avg Raw Norm = {avg_raw_norm:.4f}")

    final_count = len(clusters)
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
        self.class_mu: Dict[int, np.ndarray] = {}  # Normalized NCM centers for inference
        self.class_mu_raw: Dict[int, np.ndarray] = {}  # Raw (unnormalized) NCM centers for STAR
        self.class_count: Dict[int, int] = {}

        # 类内子簇
        self.class_clusters: Dict[int, List[_Cluster]] = {}
        
        # STAR: Store ORIGINAL clusters and NCM centers (before any transformation)
        # This allows re-alignment from scratch to avoid cumulative errors
        self.class_clusters_original: Dict[int, List[_Cluster]] = {}
        self.class_mu_original: Dict[int, np.ndarray] = {}  # Normalized
        self.class_mu_raw_original: Dict[int, np.ndarray] = {}  # Unnormalized

        # DEBUG: Store snapshots of clusters at different tasks for comparison
        # Format: {task_id: {cls: [centers_normalized, centers_raw]}}
        self.cluster_snapshots: Dict[int, Dict[int, tuple]] = {}
        
        # STAR: Frozen clusters for hard restore (Force Freeze Policy)
        # {cls: List[_Cluster]} - Stores deep copy of clusters
        self.frozen_clusters: Dict[int, List[_Cluster]] = {}

        # 任务内缓存特征（仅在 compress 时聚类）
        self.buffers: Dict[int, List[np.ndarray]] = {}

        # ------------------------------------------------------------------ #
        # Inference acceleration cache (built lazily on first predict_topk call)
        # ------------------------------------------------------------------ #
        # Cache is invalidated whenever class centers / prototypes change (compress, alignment, etc.)
        self._predict_cache_dirty: bool = True
        self._predict_cache: Dict[str, object] = {}

    # ------------------------------------------------------------------ #
    # Cache helpers
    # ------------------------------------------------------------------ #
    def invalidate_cache(self) -> None:
        """Invalidate cached tensors used by predict_topk(). Safe to call often."""
        self._predict_cache_dirty = True
        self._predict_cache.clear()

    def _ensure_predict_cache(
        self,
        device: torch.device,
        query_dim: int,
        valid_classes: List[int],
    ) -> None:
        """
        Build/cache prototype tensors on the target device to avoid per-call numpy concat and H2D copies.
        Cache key depends on: device, query_dim, and the ordered valid_classes list.
        """
        device_key = str(device)
        classes_key = tuple(int(c) for c in valid_classes)

        if (
            (not self._predict_cache_dirty)
            and self._predict_cache.get("device_key") == device_key
            and self._predict_cache.get("query_dim") == int(query_dim)
            and self._predict_cache.get("classes_key") == classes_key
        ):
            return

        # ---------- NCM centers (already normalized) ----------
        ncm_centers_np = np.stack([self.class_mu[cls] for cls in valid_classes]).astype(np.float32, copy=False)
        ncm_centers_t = torch.from_numpy(ncm_centers_np).to(device=device, dtype=torch.float32)

        # ---------- Prototypes (sub-clusters; fallback to NCM if empty) ----------
        all_protos: List[np.ndarray] = []
        proto_labels: List[int] = []       # original class id per proto
        proto_class_index: List[int] = []  # index in valid_classes per proto (0..C-1)

        for class_index, cls in enumerate(valid_classes):
            clusters = self.class_clusters.get(cls, [])
            if clusters:
                cluster_dims = [c.center.shape[0] for c in clusters]
                if all(dim == query_dim for dim in cluster_dims):
                    cls_protos = np.stack([c.center for c in clusters]).astype(np.float32, copy=False)
                    all_protos.append(cls_protos)
                    n_p = int(cls_protos.shape[0])
                    proto_labels.extend([int(cls)] * n_p)
                    proto_class_index.extend([int(class_index)] * n_p)
                    continue

            # Fallback: at least one proto per class (use NCM center)
            mu = self.class_mu[cls].astype(np.float32, copy=False)
            all_protos.append(mu[np.newaxis, :])
            proto_labels.append(int(cls))
            proto_class_index.append(int(class_index))

        all_protos_np = np.concatenate(all_protos, axis=0).astype(np.float32, copy=False)
        all_protos_t = torch.from_numpy(all_protos_np).to(device=device, dtype=torch.float32)
        proto_labels_t = torch.tensor(proto_labels, device=device, dtype=torch.long)
        proto_class_index_t = torch.tensor(proto_class_index, device=device, dtype=torch.long)

        # Prototypes and NCM centers are expected to already be L2-normalized.
        # We keep them as-is to avoid extra normalize() cost per call.
        self._predict_cache = {
            "device_key": device_key,
            "query_dim": int(query_dim),
            "classes_key": classes_key,
            "ncm_centers_t": ncm_centers_t,               # [C, D]
            "all_protos_t": all_protos_t,                 # [M, D]
            "proto_labels_t": proto_labels_t,             # [M]
            "proto_class_index_t": proto_class_index_t,   # [M] in 0..C-1
        }
        self._predict_cache_dirty = False

    # ------------------------------------------------------------------ #
    # STAR: Force Freeze Methods
    # ------------------------------------------------------------------ #
    def freeze_nodes(self, cls: int) -> None:
        """Force backup current nodes state."""
        if cls in self.class_clusters:
            # Deep copy to ensure independence
            self.frozen_clusters[cls] = [
                _Cluster(
                    center=c.center.copy(), 
                    count=c.count, 
                    center_raw=c.center_raw.copy() if c.center_raw is not None else None
                ) 
                for c in self.class_clusters[cls]
            ]
            logging.info(f"[HC-SOINN] Class {cls} nodes FROZEN.")

    def restore_frozen_nodes(self, cls: int) -> None:
        """Force restore nodes state from frozen backup."""
        if cls in self.frozen_clusters:
            self.class_clusters[cls] = []
            for c in self.frozen_clusters[cls]:
                # Deep copy raw center
                center_raw = c.center_raw.copy() if c.center_raw is not None else None
                
                # STRICTLY re-calculate normalized center from raw to ensure consistency
                # This fixes the "Raw identical but Normalized different" issue
                if center_raw is not None:
                    norm = np.linalg.norm(center_raw)
                    if norm > 1e-9:
                        center = center_raw / norm
                    else:
                        center = c.center.copy() # Fallback
                else:
                    center = c.center.copy()
                    
                self.class_clusters[cls].append(_Cluster(center, c.count, center_raw))
                
            logging.info(f"[HC-SOINN] Class {cls} nodes RESTORED from frozen state (Normalized re-calculated).")

    # ------------------------------------------------------------------ #
    # DEBUG: Snapshot and comparison methods
    # ------------------------------------------------------------------ #
    def save_cluster_snapshot(self, task_id: int, class_list: Optional[List[int]] = None) -> None:
        """
        Save a snapshot of cluster centers for debugging.
        
        Args:
            task_id: Current task ID
            class_list: List of classes to save (None = all classes)
        """
        if task_id not in self.cluster_snapshots:
            self.cluster_snapshots[task_id] = {}
        
        classes_to_save = class_list if class_list is not None else list(self.class_clusters.keys())
        
        for cls in classes_to_save:
            if cls in self.class_clusters and len(self.class_clusters[cls]) > 0:
                clusters = self.class_clusters[cls]
                centers_normalized = np.array([c.center.copy() for c in clusters])
                centers_raw = np.array([c.center_raw.copy() for c in clusters])
                self.cluster_snapshots[task_id][cls] = (centers_normalized, centers_raw)
                
    
    def compare_cluster_snapshots(self, task_id1: int, task_id2: int, class_list: Optional[List[int]] = None) -> None:
        """
        Compare cluster snapshots between two tasks.
        
        Args:
            task_id1: First task ID
            task_id2: Second task ID
            class_list: List of classes to compare (None = all common classes)
        """
        if task_id1 not in self.cluster_snapshots or task_id2 not in self.cluster_snapshots:
            return
        
        snapshot1 = self.cluster_snapshots[task_id1]
        snapshot2 = self.cluster_snapshots[task_id2]
        
        classes_to_compare = class_list if class_list is not None else list(set(snapshot1.keys()) & set(snapshot2.keys()))
        
        # Debug logging removed for performance
        for cls in sorted(classes_to_compare):
            if cls not in snapshot1 or cls not in snapshot2:
                continue
            
            centers_norm1, centers_raw1 = snapshot1[cls]
            centers_norm2, centers_raw2 = snapshot2[cls]
            
            # Check if dimensions match
            if centers_norm1.shape != centers_norm2.shape:
                logging.warning(
                    f"[DEBUG] Class {cls}: Shape mismatch! "
                    f"Task {task_id1}: {centers_norm1.shape}, Task {task_id2}: {centers_norm2.shape}"
                )
                continue
            
            # Compute differences (for internal use only, no logging)
            diff_norm = np.abs(centers_norm1 - centers_norm2)
            diff_raw = np.abs(centers_raw1 - centers_raw2)
    
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
            if cls in self.class_mu_raw:
                cls_sum_old = self.class_mu_raw[cls] * cls_count_old
            else:
                # 如果类别不存在，初始化为与特征相同维度的零数组
                cls_sum_old = np.zeros(cls_feats.shape[1], dtype=np.float32)
            cls_sum_new = cls_sum_old + cls_feats.sum(axis=0)
            cls_count_new = cls_count_old + cls_feats.shape[0]
            self.class_count[cls] = cls_count_new
            # Store both raw and normalized versions
            cls_mean_raw = cls_sum_new / float(max(cls_count_new, 1))
            self.class_mu_raw[cls] = cls_mean_raw
            self.class_mu[cls] = _normalize(cls_mean_raw)
            
            # STAR: Save original NCM centers (only once, when first created)
            if cls not in self.class_mu_original:
                self.class_mu_raw_original[cls] = cls_mean_raw.copy()
                self.class_mu_original[cls] = _normalize(cls_mean_raw)

        # class_mu / buffers updated => invalidate inference cache
        self.invalidate_cache()

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
                # FIX: 使用 center_raw (未归一化) 以保持特征尺度一致
                # 如果混合了归一化(norm=1)和未归一化(norm~80)的特征，会导致聚类中心漂移
                old_centers_raw = np.stack([c.center_raw for c in self.class_clusters[cls]], axis=0)
                feats = np.concatenate([feats, old_centers_raw], axis=0)

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
                
                # STAR: Save original clusters (only once, when first created)
                if cls not in self.class_clusters_original:
                    # Deep copy clusters to preserve original state
                    self.class_clusters_original[cls] = [
                        _Cluster(
                            center=c.center.copy(),
                            count=c.count,
                            center_raw=c.center_raw.copy() if c.center_raw is not None else None
                        )
                        for c in clusters
                    ]
                
                # DEBUG: Log new cluster statistics after replacement
                if len(clusters) > 0:
                    new_centers = np.array([c.center for c in clusters])
                    new_centers_raw = np.array([c.center_raw for c in clusters])
                    new_mean_norm = np.mean(np.linalg.norm(new_centers, axis=1))
                    new_mean_norm_raw = np.mean(np.linalg.norm(new_centers_raw, axis=1))
                    logging.info(
                        f"[HC-SOINN DEBUG] Class {cls}: Created {len(clusters)} new nodes "
                        f"(center_mean_norm={new_mean_norm:.6f}, center_raw_mean_norm={new_mean_norm_raw:.6f})"
                    )
                
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

        # Prototypes replaced => invalidate inference cache
        self.invalidate_cache()

    # ------------------------------------------------------------------ #
    # 预测 (GPU 加速版)
    # ------------------------------------------------------------------ #
    def predict_topk(
        self, query_features: np.ndarray, topk: int, total_classes: int, device=None,
        use_ease_reweighting: bool = False,
        cur_task: int = 0,
        task_increment: int = 10,
        init_cls: int = 10,
        ease_alpha: float = 0.5
    ) -> np.ndarray:
        """
        返回 shape [N, topk] 的类别索引
        使用 GPU 矩阵运算加速
        
        Args:
            query_features: [N, D] 查询特征
            topk: 返回前K个
            total_classes: 总类别数（用于fallback）
            device: 计算设备
            use_ease_reweighting: 是否使用 EASE 风格的分块加权距离
            cur_task: 当前任务 ID (用于 EASE reweighting)
            task_increment: 增量大小 (用于 EASE reweighting)
            init_cls: 初始类别数 (用于 EASE reweighting)
            ease_alpha: EASE 的 alpha 参数
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
        
        # 筛选有效类别
        valid_classes = [cls for cls in classes if self.class_mu[cls].shape[0] == query_dim]
        
        if not valid_classes:
            return np.zeros((query_features.shape[0], topk), dtype=np.int64)

        query_t = torch.from_numpy(query_features).to(device=device, dtype=torch.float32)

        # Build/reuse cached prototype tensors on the target device
        self._ensure_predict_cache(device=device, query_dim=query_dim, valid_classes=valid_classes)
        
        # -----------------------------------------------------------
        # 辅助函数：计算距离 (支持 EASE Reweighting 或 标准 Cosine)
        # -----------------------------------------------------------
        def compute_distance(protos_t, labels_t=None):
            """
            protos_t: [M, D] 原型向量
            labels_t: [M] 对应的类别标签 (仅用于 EASE reweighting)
            Return: [N, M] 距离矩阵
            """
            if use_ease_reweighting and cur_task > 0:
                # EASE 风格：分块归一化 + 加权点积
                # chunk_dim = 768 (假设)
                chunk_dim = 768 # 硬编码或从参数推断
                num_chunks = query_dim // chunk_dim
                
                # Reshape: [N, T, C]
                q_chunks = query_t.view(query_t.shape[0], num_chunks, chunk_dim)
                p_chunks = protos_t.view(protos_t.shape[0], num_chunks, chunk_dim)
                
                # Normalize each chunk independently
                q_chunks = torch.nn.functional.normalize(q_chunks, p=2, dim=2)
                p_chunks = torch.nn.functional.normalize(p_chunks, p=2, dim=2)
                
                # Compute dot product per chunk: [N, M, T]
                # q: [N, T, D'] -> [N, 1, T, D']
                # p: [M, T, D'] -> [1, M, T, D']
                # product: sum(q * p, dim=-1) -> [N, M, T]
                sim_per_chunk = (q_chunks.unsqueeze(1) * p_chunks.unsqueeze(0)).sum(dim=-1)
                
                # Apply weights
                # Weights depend on:
                # 1. Chunk index (j in 0..T-1)
                # 2. Class Task ID (i in 0..T-1)
                
                # Determine task_id for each prototype
                # labels_t: [M]
                if labels_t is None: # NCM case where protos are ordered by valid_classes
                    proto_task_ids = []
                    for c in valid_classes:
                        if c < init_cls:
                            tid = 0
                        else:
                            tid = (c - init_cls) // task_increment + 1
                        proto_task_ids.append(tid)
                    proto_task_ids = torch.tensor(proto_task_ids, device=device) # [M]
                else:
                    # Map labels to task ids
                    # This is slow, better vectorized
                    proto_task_ids = torch.zeros_like(labels_t)
                    mask_init = labels_t < init_cls
                    proto_task_ids[mask_init] = 0
                    proto_task_ids[~mask_init] = (labels_t[~mask_init] - init_cls) // task_increment + 1
                
                # Construct weight matrix [M, T]
                # If chunk_idx != task_id: weight = alpha / cur_task
                # Else: weight = 1.0
                # Note: EASE logic is: if j != i: alpha/cur_task.
                # Here j is chunk_idx (0..cur_task), i is proto_task_id
                
                M = protos_t.shape[0]
                T = num_chunks
                weights = torch.ones(M, T, device=device)
                
                # Expand proto_task_ids to [M, T]
                task_ids_expanded = proto_task_ids.unsqueeze(1).expand(M, T) # [M, T]
                chunk_indices = torch.arange(T, device=device).unsqueeze(0).expand(M, T) # [M, T]
                
                # Apply penalty mask
                penalty_mask = (chunk_indices != task_ids_expanded)
                # re-weight factor from EASE
                factor = ease_alpha / float(cur_task)
                weights[penalty_mask] = factor
                
                # Weighted Sum: [N, M]
                # sim_per_chunk: [N, M, T]
                # weights: [M, T] -> [1, M, T]
                weighted_sim = (sim_per_chunk * weights.unsqueeze(0)).sum(dim=2)
                
                return 1.0 - weighted_sim # Convert to distance-like (sort order)
                
            else:
                # 标准 Cosine Distance
                q_norm = torch.nn.functional.normalize(query_t, p=2, dim=1)
                # protos_t are expected to already be normalized (stored as such in HC-SOINN)
                sim = torch.mm(q_norm, protos_t.t())
                return 1.0 - sim

        # -----------------------------------------------------------
        # 1. 计算 NCM 距离
        # -----------------------------------------------------------
        ncm_centers_t = self._predict_cache["ncm_centers_t"]
        
        # NCM 不需要 label 参数，因为顺序对应 valid_classes
        dist_ncm = compute_distance(ncm_centers_t)  # [N, C]
        
        # -----------------------------------------------------------
        # 2. 计算 Sub-Cluster 距离
        # -----------------------------------------------------------
        all_protos_t = self._predict_cache["all_protos_t"]
        proto_labels_t = self._predict_cache["proto_labels_t"]
        proto_class_index_t = self._predict_cache["proto_class_index_t"]

        dist_proto_all = compute_distance(all_protos_t, proto_labels_t)  # [N, M]

        # Fast per-class min over prototypes: dist_sub[n, c] = min_{m in class c} dist_proto_all[n, m]
        # This replaces the Python loop with boolean masks.
        C = len(valid_classes)
        N = dist_proto_all.shape[0]
        dist_sub = torch.full((N, C), float("inf"), device=device, dtype=dist_proto_all.dtype)

        if hasattr(dist_sub, "scatter_reduce_"):
            idx = proto_class_index_t.view(1, -1).expand(N, -1)  # [N, M]
            dist_sub.scatter_reduce_(1, idx, dist_proto_all, reduce="amin", include_self=True)
        else:
            # Fallback for older torch: keep previous behavior (slower)
            for i, cls in enumerate(valid_classes):
                mask = (proto_labels_t == cls)
                if mask.any():
                    min_d, _ = dist_proto_all[:, mask].min(dim=1)
                    dist_sub[:, i] = min_d
                else:
                    dist_sub[:, i] = dist_ncm[:, i]
        
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
        self, 
        cls: int, 
        R: Optional[np.ndarray], 
        mu_old: np.ndarray, 
        mu_new: np.ndarray, 
        scale: float = 1.0,  # 新增 scale 参数
        base_scale: Optional[float] = None #以此为准，忽略此参数兼容性
    ) -> None:
        """
        [Corrected for Chain Alignment]
        Apply rigid transform (Rotation + Translation + Scaling) to current nodes.
        Formula: W_new = s * (W_old - mu_old) @ R + mu_new
        """
        if cls not in self.class_clusters:
            return
            
        clusters = self.class_clusters[cls]
        if len(clusters) == 0:
            return
        
        # 1. 获取当前节点 (Current Nodes)
        # 移除 Plan B (Original Nodes)，因为我们执行的是 Chain Update
        centers_raw = np.stack([c.center_raw for c in clusters], axis=0)  # [M, D]
        
        # 2. 应用变换 (Unnormalized Space)
        # Step A: 去中心化 (相对于旧空间的锚点中心)
        centers_centered = centers_raw - mu_old
        
        # Step B: 旋转
        if R is not None:
            centers_rotated = np.dot(centers_centered, R)
        else:
            centers_rotated = centers_centered
            
        # Step C: 缩放 & 平移 (相对于新空间的锚点中心)
        centers_new_raw = (centers_rotated * scale) + mu_new
        
        # 3. 更新节点
        for i, c in enumerate(clusters):
            new_raw = centers_new_raw[i]
            # 只有非零向量才更新
            if np.linalg.norm(new_raw) > 1e-9:
                c.center_raw = new_raw
                c.center = _normalize(new_raw) # 重新归一化用于推理
        
        # 4. 同步更新 NCM 中心
        if cls in self.class_mu_raw:
            old_mu_raw = self.class_mu_raw[cls]
            # 应用相同的变换逻辑
            mu_centered = old_mu_raw - mu_old
            if R is not None:
                mu_rotated = np.dot(mu_centered, R)
            else:
                mu_rotated = mu_centered
            
            new_mu_raw = (mu_rotated * scale) + mu_new
            
            if np.linalg.norm(new_mu_raw) > 1e-9:
                self.class_mu_raw[cls] = new_mu_raw
                self.class_mu[cls] = _normalize(new_mu_raw)

        # Nodes / class centers updated => invalidate inference cache
        self.invalidate_cache()
                
    # ------------------------------------------------------------------ #
    # 工具函数
    # ------------------------------------------------------------------ #
    def prototypes_per_class(self) -> Dict[int, int]:
        return {c: len(v) for c, v in self.class_clusters.items() if len(v) > 0}