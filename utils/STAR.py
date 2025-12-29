# """
# STAR (Structure-Topology Alignment via Residuals) 算法实现

# 核心哲学：适应而非对抗
# - 在基于预训练模型的持续学习中，由于 Backbone 的调整，特征空间会发生漂移
# - STAR 不通过蒸馏限制 Backbone，而是通过几何变换让分类器主动对齐新空间
# - 实现"结构追随"而非"结构对抗"

# 关键设计：
# 1. Plan B (原始骨架还原)：始终从类别最初产生时记录的原始节点出发进行变换
# 2. 全量拓扑映射：选择所有 SOINN 节点和 NCM 中心作为锚点
# 3. 链式覆盖：每个任务结束后更新参考特征，确保对齐链的连续性
# """

# import numpy as np
# import logging
# from typing import Dict, List, Optional, Tuple, Callable, Set, Any
# from torch.utils.data import DataLoader
# import torch


# class STARAligner:
#     """
#     STAR 对齐器：计算特征空间刚性变换并应用到 HC-SOINN 分类器
    
#     核心功能：
#     1. 计算正交 Procrustes 变换（旋转 + 平移）
#     2. 选择锚点样本（全量拓扑映射）
#     3. 对齐旧类别的 SOINN 节点
#     4. 管理锚点存储和链式更新
#     """
    
#     def __init__(
#         self,
#         hc_soinn: Any,  # HCSOINNClassifier
#         feature_extractor: Callable[[torch.Tensor], torch.Tensor],
#         device: torch.device,
#         use_full_task_rehearsal: bool = False,
#     ):
#         """
#         初始化 STAR 对齐器
        
#         Args:
#             hc_soinn: HC-SOINN 分类器实例
#             feature_extractor: 特征提取函数，输入图像 Tensor，输出特征 Tensor
#             device: 计算设备
#             use_full_task_rehearsal: 是否使用全量任务回放模式（保存所有样本而非锚点）
#         """
#         self.hc_soinn = hc_soinn
#         self.feature_extractor = feature_extractor
#         self.device = device
#         self.use_full_task_rehearsal = use_full_task_rehearsal
        
#         # 锚点存储：Key = class_id, Value = {
#         #     'images': Tensor [K, C, H, W],  # 锚点图片（持久化）
#         #     'feats_ref': np.ndarray [K, D]  # 参考特征（当前模型下的特征，用于链式对齐）
#         # }
#         self.anchor_store: Dict[int, Dict[str, Any]] = {}
        
#         logging.info("[STAR] Initialized (Plan B: Re-alignment from Original Nodes)")
#         if self.use_full_task_rehearsal:
#             logging.info("[STAR] Mode: Full Task Rehearsal (saving all samples)")
#         else:
#             logging.info("[STAR] Mode: Anchor-based (saving SOINN nodes + NCM centers)")
    
#     def compute_rigid_transform(
#         self, 
#         feats_old: np.ndarray, 
#         feats_new: np.ndarray
#     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#         """
#         计算从旧特征空间到新特征空间的刚性变换（正交 Procrustes 问题）
        
#         数学公式：
#             min ||X @ R - Y||_F^2
#             其中 X = feats_old - mu_old (去中心化)
#                  Y = feats_new - mu_new (去中心化)
#                  R 是正交旋转矩阵
        
#         变换公式：
#             feats_new = (feats_old - mu_old) @ R + mu_new
        
#         Args:
#             feats_old: 旧锚点特征 [M, D] (由上一个任务结束时保存)
#             feats_new: 相同锚点在新模型下的特征 [M, D] (由当前任务开始时提取)
        
#         Returns:
#             R: [D, D] 旋转矩阵（正交矩阵，det(R) = 1）
#             mu_old: [D] 旧特征空间的中心（未归一化）
#             mu_new: [D] 新特征空间的中心（未归一化）
        
#         Raises:
#             ValueError: 如果特征维度不匹配或样本数不足
#         """
#         feats_old = np.asarray(feats_old, dtype=np.float32)
#         feats_new = np.asarray(feats_new, dtype=np.float32)
        
#         if feats_old.shape != feats_new.shape:
#             raise ValueError(
#                 f"Feature shape mismatch: feats_old {feats_old.shape} != feats_new {feats_new.shape}"
#             )
        
#         if feats_old.shape[0] < 2:
#             raise ValueError(
#                 f"Insufficient samples for Procrustes: need at least 2, got {feats_old.shape[0]}"
#             )
        
#         # 1. 计算均值（在未归一化空间）
#         mu_old = feats_old.mean(axis=0)  # [D]
#         mu_new = feats_new.mean(axis=0)  # [D]
        
#         # 2. 去中心化
#         X = feats_old - mu_old  # [M, D]
#         Y = feats_new - mu_new  # [M, D]
        
#         # 3. 计算旋转矩阵 R（使用 SVD）
#         # 目标: 最小化 ||X @ R - Y||_F^2
#         # 解: R = U @ V^T, 其中 M = X^T @ Y = U @ S @ V^T
#         M = np.dot(X.T, Y)  # [D, D]
#         U, S, Vt = np.linalg.svd(M, full_matrices=False)
#         R = np.dot(U, Vt)  # [D, D]
        
#         # 4. 确保 R 是旋转矩阵（行列式需为 1，而非反射）
#         if np.linalg.det(R) < 0:
#             # 如果行列式为负，说明是反射而非旋转，需要修正
#             Vt[-1, :] *= -1
#             R = np.dot(U, Vt)
        
#         # 验证：R 应该是正交矩阵
#         R_Rt = np.dot(R, R.T)
#         identity = np.eye(R.shape[0], dtype=R.dtype)
#         if not np.allclose(R_Rt, identity, atol=1e-6):
#             logging.warning(
#                 f"[STAR] Rotation matrix is not orthogonal: max deviation = "
#                 f"{np.abs(R_Rt - identity).max():.2e}"
#             )
        
#         # 计算对齐误差（用于调试）
#         X_aligned = np.dot(X, R)  # [M, D]
#         error_before = np.linalg.norm(Y - X, 'fro') / np.sqrt(X.shape[0])
#         error_after = np.linalg.norm(Y - X_aligned, 'fro') / np.sqrt(X.shape[0])
        
#         logging.debug(
#             f"[STAR] Procrustes alignment: error_before={error_before:.6f}, "
#             f"error_after={error_after:.6f}, reduction={1 - error_after/error_before:.2%}"
#         )
        
#         return R, mu_old, mu_new
    
#     def select_anchors_for_current_task(
#         self,
#         dataset,
#         batch_size: int = 128,
#         num_workers: int = 4,
#         current_task_classes: Optional[Set[int]] = None,
#     ) -> None:
#         """
#         为当前任务的新类别选择并保存锚点（全量拓扑映射）
        
#         策略：
#         1. 确定靶心：获取该类所有 SOINN 节点的 center_raw 和 NCM 中心的 class_mu_raw
#         2. 最近邻匹配：在训练数据中找到与这些靶心余弦距离最近的原始训练样本
#         3. 持久化：保存这些样本的图像（用于重提取）和当前模型下的特征（作为 feats_ref）
        
#         关键点：
#         - 锚点选择在 compress() 之后进行（此时已有 SOINN 节点）
#         - 保存的图像用于后续任务重新提取特征
#         - 保存的特征作为当前任务的参考（feats_ref），用于链式对齐
        
#         Args:
#             dataset: 当前任务的训练数据集
#             batch_size: 批处理大小
#             num_workers: 数据加载器的工作进程数
#             current_task_classes: 当前任务的类别集合（如果为 None，则从 dataset 推断）
#         """
#         loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
#         # 1. 提取所有样本特征和图像
#         all_feats = []
#         all_imgs = []
#         all_targets = []
        
#         hc_soinn = self.hc_soinn
        
#         with torch.no_grad():
#             for _, inputs, targets in loader:
#                 inputs = inputs.to(self.device)
#                 feats = self.feature_extractor(inputs)
                
#                 if isinstance(feats, torch.Tensor):
#                     feats = feats.detach().cpu().numpy()
#                 elif isinstance(feats, tuple):
#                     feats = feats[0].detach().cpu().numpy()
                
#                 all_feats.append(feats)
#                 all_imgs.append(inputs.cpu())
#                 all_targets.append(targets.numpy())
        
#         if len(all_feats) == 0:
#             logging.warning("[STAR] No features extracted for anchor selection")
#             return
        
#         all_feats = np.concatenate(all_feats, axis=0)  # [N, D]
#         all_imgs = torch.cat(all_imgs, dim=0)  # [N, C, H, W]
#         all_targets = np.concatenate(all_targets, axis=0)  # [N]
        
#         # 2. 确定当前任务的类别
#         if current_task_classes is None:
#             current_task_classes = set(np.unique(all_targets))
#         else:
#             current_task_classes = set(current_task_classes)
        
#         # 3. 为每个类别选择锚点
#         for cls in current_task_classes:
#             if cls in self.anchor_store:
#                 logging.warning(f"[STAR] Class {cls} already has anchors, skipping selection")
#                 continue
            
#             cls_mask = (all_targets == cls)
#             cls_feats = all_feats[cls_mask]  # [N_cls, D]
#             cls_imgs = all_imgs[cls_mask]  # [N_cls, C, H, W]
            
#             if cls_feats.shape[0] == 0:
#                 logging.warning(f"[STAR] No samples found for class {cls}, skipping anchor selection")
#                 continue
            
#             # 3.1 确定靶心：获取所有 SOINN 节点和 NCM 中心
#             target_points = []
            
#             # SOINN 节点（使用 center_raw，未归一化）
#             if cls in hc_soinn.class_clusters and len(hc_soinn.class_clusters[cls]) > 0:
#                 for cluster in hc_soinn.class_clusters[cls]:
#                     if hasattr(cluster, 'center_raw') and cluster.center_raw is not None:
#                         target_points.append(cluster.center_raw)
#                     else:
#                         # Fallback: 使用归一化的 center（需要反归一化，但这里简化处理）
#                         logging.warning(
#                             f"[STAR] Class {cls}: Cluster missing center_raw, using normalized center"
#                         )
#                         target_points.append(cluster.center)
            
#             # NCM 中心（使用 class_mu_raw，未归一化）
#             if cls in hc_soinn.class_mu_raw:
#                 target_points.append(hc_soinn.class_mu_raw[cls])
            
#             if len(target_points) == 0:
#                 logging.warning(
#                     f"[STAR] Class {cls}: No SOINN nodes or NCM center found, "
#                     f"using all samples as anchors"
#                 )
#                 # Fallback: 使用所有样本
#                 anchors_feats = cls_feats
#                 anchors_imgs = cls_imgs
#             else:
#                 # 3.2 最近邻匹配：在归一化特征上找最近邻（使用余弦相似度）
#                 target_points = np.array(target_points)  # [M, D]
                
#                 # 归一化靶心和样本特征（用于余弦距离计算）
#                 target_norm = target_points / (np.linalg.norm(target_points, axis=1, keepdims=True) + 1e-8)
#                 cls_feats_norm = cls_feats / (np.linalg.norm(cls_feats, axis=1, keepdims=True) + 1e-8)
                
#                 # 计算相似度矩阵 [M, N_cls]
#                 sim_matrix = np.dot(target_norm, cls_feats_norm.T)
                
#                 # 为每个靶心找到最近邻样本
#                 nearest_indices = np.argmax(sim_matrix, axis=1)  # [M]
#                 unique_indices = np.unique(nearest_indices)
                
#                 anchors_feats = cls_feats[unique_indices]  # [K, D]
#                 anchors_imgs = cls_imgs[unique_indices]  # [K, C, H, W]
            
#             # 3.3 持久化：保存锚点图像和特征
#             self.anchor_store[cls] = {
#                 'images': anchors_imgs,  # Tensor [K, C, H, W]
#                 'feats_ref': anchors_feats.copy(),  # np.ndarray [K, D] (当前模型下的特征)
#             }
            
#             logging.info(
#                 f"[STAR] Class {cls}: Selected {len(anchors_feats)} anchors "
#                 f"(from {len(target_points)} target points, {cls_feats.shape[0]} samples)"
#             )
    
#     def align_old_classes(
#         self,
#         cur_task: int,
#         current_task_classes: Optional[Set[int]] = None,
#     ) -> None:
#         """
#         对齐所有旧类别的 SOINN 节点（链式对齐）
        
#         流程：
#         1. 对于每个旧类别（不在 current_task_classes 中）：
#            a. 加载锚点图像
#            b. 用当前模型重新提取特征（feats_new）
#            c. 获取参考特征（feats_old = anchor_store[cls]['feats_ref']）
#            d. 计算变换矩阵 (R, mu_old, mu_new)
#            e. 调用 hc_soinn.apply_rigid_transform 更新节点位置
#         2. 链式覆盖：将 feats_ref 更新为 feats_new（用于下一个任务）
        
#         关键点：
#         - Plan B：apply_rigid_transform 会从 class_clusters_original 开始变换
#         - 链式覆盖：确保对齐链的连续性（F_0 -> F_1 -> F_2 -> ...）
        
#         Args:
#             cur_task: 当前任务编号
#             current_task_classes: 当前任务的新类别集合（这些类别不需要对齐）
#         """
#         if current_task_classes is None:
#             current_task_classes = set()
        
#         hc_soinn = self.hc_soinn
#         aligned_count = 0
#         skipped_count = 0
        
#         for cls, anchor_data in list(self.anchor_store.items()):
#             # 跳过当前任务的新类别
#             if cls in current_task_classes:
#                 continue
            
#             # 检查类别是否存在于 HC-SOINN 中
#             if cls not in hc_soinn.class_clusters or len(hc_soinn.class_clusters[cls]) == 0:
#                 logging.warning(
#                     f"[STAR] Class {cls}: No clusters found in HC-SOINN, skipping alignment"
#                 )
#                 skipped_count += 1
#                 continue
            
#             try:
#                 # 1. 加载锚点图像
#                 anchor_imgs = anchor_data['images']  # Tensor [K, C, H, W]
#                 feats_old = anchor_data['feats_ref']  # np.ndarray [K, D] (参考特征)
                
#                 # 2. 用当前模型重新提取特征
#                 anchor_imgs = anchor_imgs.to(self.device)
#                 with torch.no_grad():
#                     feats_new = self.feature_extractor(anchor_imgs)
#                     if isinstance(feats_new, torch.Tensor):
#                         feats_new = feats_new.detach().cpu().numpy()
#                     elif isinstance(feats_new, tuple):
#                         feats_new = feats_new[0].detach().cpu().numpy()
                
#                 # 3. 维度检查
#                 if feats_old.shape != feats_new.shape:
#                     logging.error(
#                         f"[STAR] Class {cls}: Feature shape mismatch: "
#                         f"feats_old {feats_old.shape} != feats_new {feats_new.shape}, skipping"
#                     )
#                     skipped_count += 1
#                     continue
                
#                 # 4. 计算变换矩阵
#                 R, mu_old, mu_new = self.compute_rigid_transform(feats_old, feats_new)
                
#                 # 5. 应用变换到 HC-SOINN 节点（Plan B：从原始节点开始）
#                 hc_soinn.apply_rigid_transform(cls, R, mu_old, mu_new)
                
#                 # 6. 链式覆盖：更新参考特征为当前特征（用于下一个任务）
#                 self.anchor_store[cls]['feats_ref'] = feats_new.copy()
                
#                 aligned_count += 1
#                 logging.debug(
#                     f"[STAR] Task {cur_task}: Aligned class {cls} "
#                     f"({len(anchor_imgs)} anchors, {len(hc_soinn.class_clusters[cls])} nodes)"
#                 )
                
#             except Exception as e:
#                 logging.error(
#                     f"[STAR] Class {cls}: Alignment failed: {e}",
#                     exc_info=True
#                 )
#                 skipped_count += 1
        
#         if aligned_count > 0:
#             logging.info(
#                 f"[STAR] Task {cur_task}: Aligned {aligned_count} classes, "
#                 f"skipped {skipped_count} classes"
#             )
#         elif skipped_count > 0:
#             logging.warning(
#                 f"[STAR] Task {cur_task}: No classes aligned, {skipped_count} classes skipped"
#             )
    
#     def get_anchor_info(self, cls: int) -> Optional[Dict[str, Any]]:
#         """
#         获取指定类别的锚点信息（用于调试）
        
#         Args:
#             cls: 类别 ID
        
#         Returns:
#             锚点信息字典，如果类别不存在则返回 None
#         """
#         if cls not in self.anchor_store:
#             return None
        
#         anchor_data = self.anchor_store[cls]
#         return {
#             'num_anchors': len(anchor_data['images']),
#             'feat_dim': anchor_data['feats_ref'].shape[1],
#             'image_shape': anchor_data['images'].shape[1:],
#         }
    
#     def clear_anchors(self, class_list: Optional[List[int]] = None) -> None:
#         """
#         清除指定类别的锚点（用于调试或重置）
        
#         Args:
#             class_list: 要清除的类别列表，如果为 None 则清除所有
#         """
#         if class_list is None:
#             self.anchor_store.clear()
#             logging.info("[STAR] Cleared all anchors")
#         else:
#             for cls in class_list:
#                 if cls in self.anchor_store:
#                     del self.anchor_store[cls]
#             logging.info(f"[STAR] Cleared anchors for classes: {class_list}")

"""
STAR (Structure-Topology Alignment via Residuals) 算法实现

修复版本说明：
1. 增加了 Scale (缩放因子) 的计算，以应对特征空间的伸缩。
2. 实现了严格的"链式对齐"：Task T-1 -> Task T。
3. 配合修改后的 hc_soinn_classifier 使用，移除了与 Plan B 的逻辑冲突。
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Callable, Set, Any
from torch.utils.data import DataLoader
import torch


class STARAligner:
    """
    STAR 对齐器：计算特征空间刚性变换并应用到 HC-SOINN 分类器
    """
    
    def __init__(
        self,
        hc_soinn: Any,  # HCSOINNClassifier
        feature_extractor: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device,
        use_full_task_rehearsal: bool = False,
    ):
        self.hc_soinn = hc_soinn
        self.feature_extractor = feature_extractor
        self.device = device
        self.use_full_task_rehearsal = use_full_task_rehearsal
        
        # 锚点存储
        self.anchor_store: Dict[int, Dict[str, Any]] = {}
        
        logging.info("[STAR] Initialized (Mode: Chain Alignment with Scaling)")
    
    def compute_rigid_transform(
        self, 
        feats_old: np.ndarray, 
        feats_new: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        计算刚性变换 (旋转 + 平移 + 缩放)
        
        返回:
            R: 旋转矩阵 [D, D]
            mu_old: 旧中心 [D]
            mu_new: 新中心 [D]
            s: 缩放因子 (float)
        """
        feats_old = np.asarray(feats_old, dtype=np.float32)
        feats_new = np.asarray(feats_new, dtype=np.float32)
        
        if feats_old.shape != feats_new.shape:
            raise ValueError(f"Feature shape mismatch: {feats_old.shape} != {feats_new.shape}")
        
        if feats_old.shape[0] < 2:
            # 样本太少无法计算，返回单位变换
            D = feats_old.shape[1]
            return np.eye(D, dtype=np.float32), np.zeros(D), np.zeros(D), 1.0
        
        # 1. 计算均值
        mu_old = feats_old.mean(axis=0)
        mu_new = feats_new.mean(axis=0)
        
        # 2. 去中心化
        X = feats_old - mu_old
        Y = feats_new - mu_new
        
        # 3. 计算旋转 R (SVD)
        # min || s * X @ R - Y ||
        M = np.dot(X.T, Y)
        U, S, Vt = np.linalg.svd(M)
        R = np.dot(U, Vt)
        
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(U, Vt)
        
        # 4. 计算缩放因子 s
        var_old = np.sum(np.square(X))
        trace_S = np.sum(S)
        if var_old < 1e-8:
            s = 1.0
        else:
            s = trace_S / var_old
            
        # ================= [新增 Debug 信息] =================
        # 计算旧中心和新中心的欧氏距离
        shift_dist = np.linalg.norm(mu_new - mu_old)
        # 计算特征的平均模长 (用于归一化漂移量)
        avg_norm = np.mean(np.linalg.norm(feats_old, axis=1))
        # 计算旋转角度 (弧度)
        trace_R = np.trace(R)
        theta = np.arccos(np.clip((trace_R - (feats_old.shape[1] - 2)) / 2, -1, 1)) # 简化的估算

        logging.info(f"[STAR DEBUG] Drift Analysis:")
        logging.info(f"  > Shift (Mean Move): {shift_dist:.6f} (Avg Norm: {avg_norm:.2f})")
        logging.info(f"  > Scale Change: {s:.6f}")
        logging.info(f"  > Rotation Angle: {theta:.6f}")
        
        # 如果漂移非常小，STAR 就不会有效果
        if shift_dist < 0.1 and abs(s - 1.0) < 0.01:
             logging.warning("[STAR WARNING] Feature drift is NEGLIGIBLE! STAR will have no effect.")
        # ====================================================
        
        return R, mu_old, mu_new, s
    
    def select_anchors_for_current_task(
        self,
        dataset,
        batch_size: int = 128,
        num_workers: int = 4,
        current_task_classes: Optional[Set[int]] = None,
    ) -> None:
        """为当前任务选择锚点并保存参考特征"""
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        all_feats = []
        all_imgs = []
        all_targets = []
        
        with torch.no_grad():
            for _, inputs, targets in loader:
                inputs = inputs.to(self.device)
                feats = self.feature_extractor(inputs)
                if isinstance(feats, tuple): feats = feats[0]
                
                all_feats.append(feats.detach().cpu().numpy())
                all_imgs.append(inputs.cpu())
                all_targets.append(targets.numpy())
        
        if not all_feats: return
        
        all_feats = np.concatenate(all_feats, axis=0)
        all_imgs = torch.cat(all_imgs, dim=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        if current_task_classes is None:
            current_task_classes = set(np.unique(all_targets))
        
        for cls in current_task_classes:
            if cls in self.anchor_store: continue
            
            mask = (all_targets == cls)
            cls_feats = all_feats[mask]
            cls_imgs = all_imgs[mask]
            
            if len(cls_feats) == 0: continue
            
            # --- 拓扑锚点选择逻辑 ---
            # 1. 收集靶心 (SOINN 节点 + NCM)
            targets = []
            if cls in self.hc_soinn.class_clusters:
                for c in self.hc_soinn.class_clusters[cls]:
                    # 优先使用 raw center，如果没有则用 center
                    targets.append(c.center_raw if c.center_raw is not None else c.center)
            
            if cls in self.hc_soinn.class_mu_raw:
                targets.append(self.hc_soinn.class_mu_raw[cls])
            
            if not targets:
                # Fallback: 全量保存
                sel_feats = cls_feats
                sel_imgs = cls_imgs
            else:
                targets = np.array(targets)
                # 归一化用于计算余弦相似度
                t_norm = targets / (np.linalg.norm(targets, axis=1, keepdims=True) + 1e-8)
                f_norm = cls_feats / (np.linalg.norm(cls_feats, axis=1, keepdims=True) + 1e-8)
                
                sims = np.dot(t_norm, f_norm.T)
                indices = np.unique(np.argmax(sims, axis=1))
                
                sel_feats = cls_feats[indices]
                sel_imgs = cls_imgs[indices]

            self.anchor_store[cls] = {
                'images': sel_imgs,
                'feats_ref': sel_feats.copy() # 保存当前任务空间的特征作为参考
            }
            logging.info(f"[STAR] Class {cls}: Saved {len(sel_feats)} anchors.")

    def align_old_classes(
        self,
        cur_task: int,
        current_task_classes: Optional[Set[int]] = None,
    ) -> None:
        """执行链式对齐: Task T-1 -> Task T"""
        if current_task_classes is None: current_task_classes = set()
        
        aligned_cnt = 0
        
        for cls, data in self.anchor_store.items():
            if cls in current_task_classes: continue
            if cls not in self.hc_soinn.class_clusters: continue
            
            try:
                # 1. 准备数据
                imgs = data['images'].to(self.device)
                feats_old = data['feats_ref']  # T-1 时刻的特征
                
                # 2. 提取当前 (Task T) 特征
                with torch.no_grad():
                    feats_new = self.feature_extractor(imgs)
                    if isinstance(feats_new, tuple): feats_new = feats_new[0]
                    feats_new = feats_new.detach().cpu().numpy()
                
                # 3. 计算变换 (含 Scaling)
                R, mu_old, mu_new, s = self.compute_rigid_transform(feats_old, feats_new)
                
                # 4. 应用变换 (传入 scale)
                # 注意：这里会修改 HC-SOINN 内部节点的 center_raw
                self.hc_soinn.apply_rigid_transform(cls, R, mu_old, mu_new, scale=s)
                
                # 5. 关键：链式更新
                # 将参考特征更新为当前特征，以便 Task T+1 时使用
                self.anchor_store[cls]['feats_ref'] = feats_new.copy()
                
                aligned_cnt += 1
                
            except Exception as e:
                logging.error(f"[STAR] Align failed for class {cls}: {e}")

        logging.info(f"[STAR] Task {cur_task}: Aligned {aligned_cnt} classes.")