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
        # 预先归一化的特征库（用于加速推理，仅当 metric='cosine' 时使用）
        self.bank_feats_norm = None  # numpy array shape (M, D)，已归一化
        # 使用频率统计：记录每个节点被使用的次数
        # 格式：{class_id: np.ndarray([Ni,])}，每个元素对应该类中第i个节点的使用次数
        self.usage_counts: Dict[int, np.ndarray] = {}

    def clear(self) -> None:
        """清空整个特征库。"""
        self.class_to_features.clear()
        self.bank_feats_norm = None
        self.usage_counts.clear()

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
            # 初始化使用频率统计（如果该类还没有）
            if cls not in self.usage_counts:
                # 计算该类当前的总节点数
                total_nodes = sum(chunk.shape[0] for chunk in self.class_to_features[cls])
                self.usage_counts[cls] = np.zeros(total_nodes, dtype=np.int64)
            else:
                # 扩展使用频率数组以匹配新的节点数
                total_nodes = sum(chunk.shape[0] for chunk in self.class_to_features[cls])
                old_size = len(self.usage_counts[cls])
                if total_nodes > old_size:
                    # 扩展数组，新增的节点使用次数为0
                    new_counts = np.zeros(total_nodes, dtype=np.int64)
                    new_counts[:old_size] = self.usage_counts[cls]
                    self.usage_counts[cls] = new_counts

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
        bank_feats = np.concatenate(all_features)
        bank_labels = np.concatenate(all_labels)
        
        # 如果使用 cosine 距离，预先归一化特征库以加速推理
        if self.metric == "cosine":
            norms = np.linalg.norm(bank_feats, axis=1, keepdims=True)
            self.bank_feats_norm = bank_feats / (norms + 1e-8)
        else:
            self.bank_feats_norm = None
        
        return bank_feats, bank_labels

    def _update_usage_counts(self, used_indices: np.ndarray, bank_labels: np.ndarray) -> None:
        """
        更新使用频率统计。
        
        参数:
        - used_indices: [N, K] 或 [N, M] 的索引数组，表示每个查询使用的节点索引（全局索引）
        - bank_labels: [M] 特征库的标签数组
        """
        if len(self.usage_counts) == 0:
            return
        
        # 计算每个类别在bank_labels中的起始位置
        class_start_indices = {}
        current_idx = 0
        for cls in sorted(self.class_to_features.keys()):
            class_start_indices[cls] = current_idx
            # 计算该类在bank_labels中的节点数
            class_mask = bank_labels == cls
            class_size = np.sum(class_mask)
            current_idx += class_size
        
        # 统计每个节点被使用的次数
        for query_indices in used_indices:
            for global_idx in query_indices:
                if global_idx < len(bank_labels):
                    cls = bank_labels[global_idx]
                    if cls in self.usage_counts and cls in class_start_indices:
                        # 计算在类内的相对索引
                        class_start = class_start_indices[cls]
                        # 找到该类在bank_labels中的所有索引
                        class_mask = bank_labels == cls
                        class_global_indices = np.where(class_mask)[0]
                        if global_idx in class_global_indices:
                            local_idx = np.where(class_global_indices == global_idx)[0][0]
                            if local_idx < len(self.usage_counts[cls]):
                                self.usage_counts[cls][local_idx] += 1

    def predict_topk(self, query_features: np.ndarray, topk: int, total_classes: int, device: Optional[torch.device] = None, track_usage: bool = False) -> np.ndarray:
        """
        使用特征库对查询特征进行 KNN 预测，返回 top-k 类别索引。
        优化版本：使用矩阵运算和 GPU 加速。

        参数:
        - query_features: [N, D] 查询特征矩阵。
        - topk: 返回前 k 个类别。
        - total_classes: 当前已学习的类别总数（用于在样本不足时补齐不重复的类别索引）。
        - device: 可选的 torch.device，用于 GPU 加速。如果为 None，使用 CPU。

        细节:
        - 若 use_all_samples=True 或未设置 k_neighbors，则用“全量样本投票”；
        - 否则仅用前 K 个最近邻样本投票（K 至少会覆盖 topk）。
        - 同一类别的票数相同（平票）时，用最小距离更近者优先。
        """
        bank_feats, bank_labels = self._gather_bank()
        if bank_feats is None or bank_feats.shape[0] == 0:
            # 特征库为空时应该抛出错误，而不是返回占位预测
            raise ValueError(
                "KNN feature bank is empty. Please ensure features are added to the bank "
                "before calling predict_topk (e.g., call add_from_loader or rebuild_from_loader)."
            )

        query_features = np.asarray(query_features)
        N = query_features.shape[0]  # 查询样本数
        M = bank_feats.shape[0]  # 特征库样本数
        
        # 决定使用 CPU 还是 GPU
        use_gpu = device is not None and torch.cuda.is_available() and self.metric in ["euclidean", "cosine"]
        
        # # 记录使用的设备
        # if use_gpu:
        #     logging.info(f"KNN inference using GPU: {device} (query_samples={N}, bank_samples={M}, metric={self.metric})")
        # else:
        #     device_str = str(device) if device is not None else "CPU"
        #     reason = "CUDA not available" if device is not None and not torch.cuda.is_available() else \
        #              f"metric '{self.metric}' not GPU-accelerated" if device is not None and self.metric not in ["euclidean", "cosine"] else \
        #              "device not specified"
        #     logging.info(f"KNN inference using CPU (query_samples={N}, bank_samples={M}, metric={self.metric}, reason={reason})")
        
        if use_gpu:
            # GPU 加速版本
            query_t = torch.from_numpy(query_features).float().to(device)  # [N, D]
            bank_labels_t = torch.from_numpy(bank_labels).long().to(device)  # [M]
            
            if self.metric == "cosine":
                # 使用预先归一化的特征库
                if self.bank_feats_norm is None:
                    # 如果还没有归一化，现在归一化
                    norms = np.linalg.norm(bank_feats, axis=1, keepdims=True)
                    bank_feats_norm = bank_feats / (norms + 1e-8)
                else:
                    bank_feats_norm = self.bank_feats_norm
                
                bank_feats_t = torch.from_numpy(bank_feats_norm).float().to(device)  # [M, D]
                # 归一化查询特征
                query_norm_t = torch.nn.functional.normalize(query_t, p=2, dim=1)  # [N, D]
                # 批量计算余弦相似度：[N, M] = [N, D] @ [D, M]
                cosine_sims = torch.mm(query_norm_t, bank_feats_t.t())  # [N, M]
                # 转换为余弦距离
                dists = 1.0 - cosine_sims  # [N, M]
            else:  # euclidean
                bank_feats_t = torch.from_numpy(bank_feats).float().to(device)  # [M, D]
                # 批量计算欧氏距离：[N, M]
                # dists[i, j] = ||query[i] - bank[j]||^2
                dists = torch.cdist(query_t, bank_feats_t, p=2)  # [N, M]
            
            # 根据 use_all_samples 决定使用全量样本还是 K-NN
            if self.use_all_samples or self.k_neighbors is None:
                # 全量样本投票：使用所有样本
                nn_labels = bank_labels_t.unsqueeze(0).expand(N, -1)  # [N, M]
                nn_dists = dists  # [N, M]
                weights = 1.0 / (dists + 1e-8)  # [N, M]
            else:
                # 仅取前 K 个最近邻投票
                K = max(self.k_neighbors, topk)
                k_dists, k_indices = torch.topk(dists, k=min(K, M), dim=1, largest=False)  # [N, K]
                nn_labels = bank_labels_t[k_indices]  # [N, K]
                nn_dists = k_dists  # [N, K]
                weights = 1.0 / (k_dists + 1e-8)  # [N, K]
            
            # 批量处理：将数据移到 CPU 进行投票计算
            nn_labels_cpu = nn_labels.cpu().numpy()  # [N, M] or [N, K]
            nn_dists_cpu = nn_dists.cpu().numpy()  # [N, M] or [N, K]
            weights_cpu = weights.cpu().numpy()  # [N, M] or [N, K]
            
            # 记录使用的节点索引（用于使用频率统计）
            used_indices_list = []
            
            # 对每个查询样本进行投票
            preds = []
            for i in range(N):
                # 统计每个类别的加权票数与最小距离
                votes: Dict[int, float] = {}
                min_dist: Dict[int, float] = {}
                
                # 记录使用的节点索引
                if track_usage:
                    if self.use_all_samples or self.k_neighbors is None:
                        # 全量样本投票：使用所有节点
                        used_indices = np.arange(M)
                    else:
                        # K-NN投票：使用k_indices
                        used_indices = k_indices[i].cpu().numpy()
                    used_indices_list.append(used_indices)
                
                for lbl, dist, weight in zip(nn_labels_cpu[i], nn_dists_cpu[i], weights_cpu[i]):
                    votes[lbl] = votes.get(lbl, 0.0) + weight
                    if lbl not in min_dist:
                        min_dist[lbl] = dist
                    else:
                        min_dist[lbl] = min(min_dist[lbl], dist)
                
                # 排序：先按加权票数降序，再按最小距离升序
                sorted_classes = sorted(votes.keys(), key=lambda c: (-votes[c], min_dist[c]))
                top = sorted_classes[:topk]
                
                # 补齐
                if len(top) < topk:
                    remaining = [c for c in range(total_classes) if c not in top]
                    top.extend(remaining[: topk - len(top)])
                
                preds.append(top)
            
            # 更新使用频率统计
            if track_usage and len(used_indices_list) > 0:
                used_indices_array = np.array(used_indices_list)
                self._update_usage_counts(used_indices_array, bank_labels)
            
            return np.array(preds, dtype=np.int64)
        else:
            # CPU 版本（使用矩阵运算优化）
            if self.metric == "cosine":
                # 使用预先归一化的特征库
                if self.bank_feats_norm is None:
                    # 如果还没有归一化，现在归一化
                    norms = np.linalg.norm(bank_feats, axis=1, keepdims=True)
                    bank_feats_norm = bank_feats / (norms + 1e-8)
                else:
                    bank_feats_norm = self.bank_feats_norm
                
                # 归一化查询特征
                query_norms = np.linalg.norm(query_features, axis=1, keepdims=True)
                query_norm = query_features / (query_norms + 1e-8)  # [N, D]
                
                # 批量计算余弦相似度：[N, M] = [N, D] @ [D, M]
                cosine_sims = np.dot(query_norm, bank_feats_norm.T)  # [N, M]
                # 转换为余弦距离
                dists = 1.0 - cosine_sims  # [N, M]
            else:  # euclidean or other metrics
                # 使用 cdist 批量计算距离（对于非 cosine 距离，cdist 已经优化）
                dists = cdist(query_features, bank_feats, metric=self.metric)  # [N, M]
            
            # 根据 use_all_samples 决定使用全量样本还是 K-NN
            if self.use_all_samples or self.k_neighbors is None:
                # 全量样本投票：使用所有样本
                nn_labels = bank_labels  # [M]
                nn_dists = dists  # [N, M]
                weights = 1.0 / (dists + 1e-8)  # [N, M]
                
                # 记录使用的节点索引（用于使用频率统计）
                used_indices_list = []
                
                # 对每个查询样本进行投票
                preds = []
                for i in range(N):
                    votes: Dict[int, float] = {}
                    min_dist: Dict[int, float] = {}
                    
                    # 记录使用的节点索引
                    if track_usage:
                        # 全量样本投票：使用所有节点
                        used_indices = np.arange(M)
                        used_indices_list.append(used_indices)
                    
                    for lbl, dist, weight in zip(nn_labels, nn_dists[i], weights[i]):
                        votes[lbl] = votes.get(lbl, 0.0) + weight
                        if lbl not in min_dist:
                            min_dist[lbl] = dist
                        else:
                            min_dist[lbl] = min(min_dist[lbl], dist)
                    
                    sorted_classes = sorted(votes.keys(), key=lambda c: (-votes[c], min_dist[c]))
                    top = sorted_classes[:topk]
                    
                    if len(top) < topk:
                        remaining = [c for c in range(total_classes) if c not in top]
                        top.extend(remaining[: topk - len(top)])
                    
                    preds.append(top)
                
                # 更新使用频率统计
                if track_usage and len(used_indices_list) > 0:
                    used_indices_array = np.array(used_indices_list)
                    self._update_usage_counts(used_indices_array, bank_labels)
                
                return np.array(preds, dtype=np.int64)
            else:
                # 仅取前 K 个最近邻投票
                K = max(self.k_neighbors, topk)
                # 使用 argpartition 优化（比 argsort 更快）
                k_indices = np.argpartition(dists, K-1, axis=1)[:, :K]  # [N, K]
                # 对每个查询的 k 个最近邻进行排序
                for i in range(N):
                    k_indices[i] = k_indices[i][np.argsort(dists[i, k_indices[i]])]
                
                k_labels = bank_labels[k_indices]  # [N, K]
                k_dists = np.take_along_axis(dists, k_indices, axis=1)  # [N, K]
                weights = 1.0 / (k_dists + 1e-8)  # [N, K]
                
                # 记录使用的节点索引（用于使用频率统计）
                used_indices_list = []
                
                # 对每个查询样本进行投票
                preds = []
                for i in range(N):
                    votes: Dict[int, float] = {}
                    min_dist: Dict[int, float] = {}
                    
                    # 记录使用的节点索引
                    if track_usage:
                        used_indices = k_indices[i]
                        used_indices_list.append(used_indices)
                    
                    for lbl, dist, weight in zip(k_labels[i], k_dists[i], weights[i]):
                        votes[lbl] = votes.get(lbl, 0.0) + weight
                        if lbl not in min_dist:
                            min_dist[lbl] = dist
                        else:
                            min_dist[lbl] = min(min_dist[lbl], dist)
                    
                    sorted_classes = sorted(votes.keys(), key=lambda c: (-votes[c], min_dist[c]))
                    top = sorted_classes[:topk]
                    
                    if len(top) < topk:
                        remaining = [c for c in range(total_classes) if c not in top]
                        top.extend(remaining[: topk - len(top)])
                    
                    preds.append(top)
                
                # 更新使用频率统计
                if track_usage and len(used_indices_list) > 0:
                    used_indices_array = np.array(used_indices_list)
                    self._update_usage_counts(used_indices_array, bank_labels)
                
                return np.array(preds, dtype=np.int64)


