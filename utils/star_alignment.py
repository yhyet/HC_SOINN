"""
STAR (Structure-Topology Alignment via Residuals) - 特征漂移对齐模块

STAR 是 HC-SOINN 的辅助插件，用于处理增量学习中的特征漂移问题。
通过相似变换 (Similarity Transformation) 将旧类别的 SOINN 节点从旧特征空间
对齐到新特征空间，保持拓扑结构不变。

【设计原则】
- 独立于具体的模型实现（CodaPrompt, SimpleCIL 等）
- 通过回调函数接口与外部模型交互
- 最小化外部依赖，只依赖 HC-SOINN 分类器
"""

import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, Callable, Optional, Tuple
from utils.hc_soinn_classifier import HCSOINNClassifier


class STARAlignment:
    """
    STAR 特征漂移对齐器
    
    【职责】
    1. 管理锚点（Anchor）的存储和选择
    2. 计算并应用相似变换（Similarity Transformation）
    3. 与 HC-SOINN 分类器协作，更新节点位置
    
    【使用方式】
    ```python
    star = STARAlignment(
        hc_soinn=hc_soinn_classifier,
        feature_extractor=feature_fn,  # 回调函数
        device=device
    )
    
    # 在 after_task 中调用
    star.align_old_classes(cur_task)  # 对齐旧类别
    star.select_anchors_for_current_task(dataset, batch_size, num_workers)  # 选择锚点
    ```
    """
    
    def __init__(
        self,
        hc_soinn: HCSOINNClassifier,
        feature_extractor: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device,
        use_full_task_rehearsal: bool = False,
    ):
        """
        初始化 STAR 对齐器
        
        参数:
            hc_soinn: HC-SOINN 分类器实例
            feature_extractor: 特征提取函数，输入 [B, C, H, W]，输出 [B, D]
            device: 计算设备
            use_full_task_rehearsal: 是否使用全量样本模式（默认 False）
                - False: 使用锚点模式（选择 SOINN 节点 + NCM 对应的最近邻图像）
                - True: 使用全量样本模式（保存整个 task 的训练集，用于验证性能上界）
        """
        self.hc_soinn = hc_soinn
        self.feature_extractor = feature_extractor
        self.device = device
        self.use_full_task_rehearsal = use_full_task_rehearsal
        
        # 锚点存储：{cls: {'images': tensor [K, C, H, W], 'feats': numpy [K, D]}}
        # 锚点模式：K 是去重后的锚点数量（所有 SOINN 节点 + NCM 点对应的最近邻图像，去重后）
        # 全量模式：K 是该类的所有训练样本数量
        self._class_anchors: Dict[int, Dict[str, torch.Tensor]] = {}
    
    def align_old_classes(self, cur_task: int) -> None:
        """
        ========================================================================
        STAR Step 1: 计算并应用特征漂移校正 (Drift Estimation & Alignment)
        ========================================================================
        
        对旧类别的 SOINN 节点进行相似变换，将其从旧模型空间对齐到新模型空间。
        
        参数:
            cur_task: 当前任务编号（用于判断是否为第一个任务）
        """
        # 边界检查：第一个任务或没有锚点时，无需对齐
        if cur_task == 0 or len(self._class_anchors) == 0:
            return

        if self.use_full_task_rehearsal:
            logging.info("[STAR] Starting drift estimation and alignment (FULL REHEARSAL mode)...")
        else:
            logging.info("[STAR] Starting drift estimation and alignment (anchor mode)...")

        aligned_count = 0
        # 遍历所有旧类别（所有已保存锚点的类别）
        for cls, data in self._class_anchors.items():
            # ========== Step 1.1: 获取旧特征 ==========
            feats_old = data["feats"]  # [K, D] 上一轮保存的特征
            # 锚点模式：K 是去重后的锚点数量
            # 全量模式：K 是该类的所有训练样本数量
            imgs = data["images"]      # [K, C, H, W] 图片（固定不变）
            
            # ========== Step 1.2: 计算新特征 ==========
            # 将锚点图片输入新模型，得到新特征 F_new
            with torch.no_grad():
                imgs = imgs.to(self.device)
                feats_new = self.feature_extractor(imgs)  # [K, D]
                if isinstance(feats_new, torch.Tensor):
                    feats_new = feats_new.cpu().numpy()
            
            # ========== Step 1.3: Similarity Transformation (Procrustes Analysis) ==========
            # 计算变换参数：R (旋转), s (缩放), t (平移)
            
            # 1.3.1 计算中心
            mu_old = feats_old.mean(axis=0)  # [D] 旧锚点的中心
            mu_new = feats_new.mean(axis=0)  # [D] 新锚点的中心
            
            # 1.3.2 去中心化（Procrustes 要求）
            X_old = feats_old - mu_old  # [K, D]
            X_new = feats_new - mu_new  # [K, D]
            
            # 1.3.3 计算旋转矩阵 R (Orthogonal Procrustes)
            M = np.dot(X_old.T, X_new)  # [D, D]
            U, S, Vt = np.linalg.svd(M, full_matrices=False)
            R = np.dot(U, Vt)  # [D, D] 正交旋转矩阵
            
            # 1.3.4 计算最优缩放因子 s
            trace_old = np.trace(np.dot(X_old.T, X_old))
            if trace_old > 1e-10:
                s = float(np.sum(S) / trace_old)
            else:
                # Fallback: 使用 Frobenius 范数比
                norm_old = np.linalg.norm(X_old, ord='fro')
                norm_new = np.linalg.norm(X_new, ord='fro')
                if norm_old > 1e-10:
                    s = float(norm_new / norm_old)
                else:
                    s = 1.0
            
            # 1.3.5 缩放因子合理性检查
            if s < 0.1 or s > 10.0:
                logging.warning(f"[STAR] Class {cls}: scale s={s:.4f} out of range, clamping to [0.1, 10.0]")
                s = np.clip(s, 0.1, 10.0)
            
            # 1.3.6 计算基准尺度 base_scale
            norms_old = np.linalg.norm(feats_old, axis=1)
            base_scale = float(np.mean(norms_old))
            if base_scale < 1e-6:
                base_scale = 1.0
            
            # ========== Step 1.4: 应用相似变换到 HC-SOINN 节点 ==========
            self.hc_soinn.apply_rigid_transform(cls, R, mu_old, mu_new, scale=s, base_scale=base_scale)
            
            # ========== Step 1.5: 更新缓存的特征 ==========
            self._class_anchors[cls]["feats"] = feats_new
            aligned_count += 1
            
            # 记录对齐信息
            if self.use_full_task_rehearsal:
                logging.info(f"[STAR] Class {cls}: aligned using {len(feats_old)} full samples "
                           f"(scale={s:.4f}, base_scale={base_scale:.2f})")
        
        logging.info(f"[STAR] Aligned {aligned_count} classes.")
    
    def select_anchors_for_current_task(
        self,
        dataset,
        batch_size: int,
        num_workers: int = 8,
    ) -> None:
        """
        ========================================================================
        STAR Step 3: 为当前任务选择并存储锚点 (Anchor Selection)
        ========================================================================
        
        为当前任务的每个新类别选择锚点，保存用于下一轮的漂移对齐。
        
        【两种模式】
        1. 锚点模式（use_full_task_rehearsal=False）：
           - 获取该类的所有 SOINN 节点（不限制数量）
           - 获取该类的 NCM 点（类中心）
           - 为每个节点/NCM 点找最近邻真实样本
           - 去重：如果多个节点选到同一图像，只保留一个
        
        2. 全量模式（use_full_task_rehearsal=True）：
           - 保存整个 task 的训练集（所有样本）
           - 用于验证 STAR 实现的性能上界
           - 如果全量样本性能还差，说明实现可能有 bug
        
        参数:
            dataset: 当前任务的训练数据集
            batch_size: 批大小
            num_workers: DataLoader 的工作进程数
        """
        if self.use_full_task_rehearsal:
            logging.info("[STAR] Full task rehearsal mode: saving all training samples...")
        else:
            logging.info("[STAR] Anchor mode: selecting anchors (all nodes + NCM)...")
        
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        # ========== Step 3.1: 提取所有样本的特征和图片 ==========
        # 注意：此时模型已固定（训练完成），提取的特征可以直接复用，既用于最近邻搜索，也作为最终保存的特征
        all_feats = []
        all_imgs = []
        all_targets = []
        
        with torch.no_grad():
            for _, inputs, targets in loader:
                inputs = inputs.to(self.device)
                feats = self.feature_extractor(inputs)
                if isinstance(feats, torch.Tensor):
                    feats = feats.cpu().numpy()
                all_feats.append(feats)
                all_imgs.append(inputs.cpu())
                all_targets.append(targets.cpu().numpy())
        
        if len(all_feats) == 0:
            return

        all_feats = np.concatenate(all_feats, axis=0)  # [N, D]
        all_imgs = torch.cat(all_imgs, dim=0)          # [N, C, H, W]
        all_targets = np.concatenate(all_targets, axis=0)  # [N]
        
        # ========== Step 3.2: 为每个新类别选择锚点 ==========
        current_classes = np.unique(all_targets)
        
        # 全量模式：直接保存所有样本
        if self.use_full_task_rehearsal:
            for cls in current_classes:
                cls_mask = (all_targets == cls)
                cls_imgs = all_imgs[cls_mask]  # [N_cls, C, H, W]
                cls_feats = all_feats[cls_mask]  # [N_cls, D]
                
                if len(cls_feats) == 0:
                    continue
                
                # 存储该类的所有样本
                self._class_anchors[cls] = {
                    "images": cls_imgs,  # [N_cls, C, H, W]
                    "feats": cls_feats   # [N_cls, D] 当前模型的特征
                }
                
                logging.info(f"[STAR] Class {cls}: saved {len(cls_feats)} full training samples")
            
            logging.info(f"[STAR] Full rehearsal stored for classes: {current_classes}")
            return
        
        # 锚点模式：选择 SOINN 节点 + NCM 对应的最近邻图像
        for cls in current_classes:
            # 3.2.1 获取该类的所有 SOINN 节点（不限制数量）
            if cls not in self.hc_soinn.class_clusters or len(self.hc_soinn.class_clusters[cls]) == 0:
                # 如果没有 SOINN 节点，尝试使用 NCM 点
                if cls not in self.hc_soinn.class_mu:
                    continue
                soinn_centers = []
            else:
                clusters = self.hc_soinn.class_clusters[cls]
                soinn_centers = [c.center for c in clusters]  # 所有 SOINN 节点中心（已归一化）
            
            # 3.2.2 获取该类的 NCM 点（类中心）
            ncm_center = None
            if cls in self.hc_soinn.class_mu:
                ncm_center = self.hc_soinn.class_mu[cls]  # [D] 已归一化
            
            # 3.2.3 合并所有目标点（SOINN 节点 + NCM 点）
            target_points = soinn_centers.copy()
            if ncm_center is not None:
                target_points.append(ncm_center)
            
            if len(target_points) == 0:
                continue
            
            # 3.2.4 获取该类所有训练样本
            cls_mask = (all_targets == cls)
            cls_feats = all_feats[cls_mask]        # [N_cls, D] 用于最近邻搜索
            cls_imgs_tensor = all_imgs[cls_mask]   # [N_cls, C, H, W] 图像
            cls_indices = np.where(cls_mask)[0]   # [N_cls] 原始索引（用于去重）
            
            if len(cls_feats) == 0:
                continue

            # 3.2.5 归一化样本特征（用于余弦相似度计算）
            cls_feats_norm = cls_feats / (np.linalg.norm(cls_feats, axis=1, keepdims=True) + 1e-8)
            
            # 3.2.6 为每个目标点（SOINN 节点 + NCM）找最近邻真实样本
            selected_indices = set()  # 用于去重
            selected_image_indices = []  # 保存去重后的图像索引（用于从 all_imgs 和 all_feats 中取出）
            
            for target_point in target_points:
                # 计算余弦相似度
                sims = np.dot(cls_feats_norm, target_point)  # [N_cls]
                best_idx = np.argmax(sims)  # 在 cls_feats 中的索引
                
                # 转换为原始 all_imgs/all_feats 中的索引
                original_idx = cls_indices[best_idx]
                
                # 去重：如果该图像已被选择，跳过
                if original_idx not in selected_indices:
                    selected_indices.add(original_idx)
                    selected_image_indices.append(original_idx)
            
            if len(selected_image_indices) == 0:
                continue
            
            # 3.2.7 从已提取的特征中取出对应的特征（复用，节省计算量）
            # 注意：此时模型已固定（训练完成），all_feats 就是当前模型的特征，可以直接复用
            selected_image_indices = np.array(selected_image_indices)
            selected_images = all_imgs[selected_image_indices]  # [K, C, H, W]
            selected_feats = all_feats[selected_image_indices]   # [K, D] 当前模型的特征（复用，节省计算量）
            
            # 3.2.8 存储锚点
            self._class_anchors[cls] = {
                "images": selected_images,  # [K, C, H, W]
                "feats": selected_feats     # [K, D] 当前模型的特征
            }
            
            logging.info(f"[STAR] Class {cls}: selected {len(selected_images)} anchors "
                        f"(from {len(target_points)} target points: {len(soinn_centers)} SOINN nodes + "
                        f"{1 if ncm_center is not None else 0} NCM point)")
        
        logging.info(f"[STAR] Anchors stored for classes: {current_classes}")
    
    def clear_anchors(self) -> None:
        """清空所有锚点（用于重置）"""
        self._class_anchors.clear()
    
    def get_anchor_count(self) -> int:
        """返回已保存的锚点类别数"""
        return len(self._class_anchors)

