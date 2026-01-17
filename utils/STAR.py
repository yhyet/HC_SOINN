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
    ) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray, float]:
        """
        计算刚性变换 (仅平移模式：旋转和缩放已禁用)
        
        注意：当前实现只保留平移，旋转 R 设置为 None，缩放 s 设置为 1.0
        
        返回:
            R: 旋转矩阵 [D, D] 或 None（平移模式下为 None）
            mu_old: 旧中心 [D]
            mu_new: 新中心 [D]
            s: 缩放因子 (float，平移模式下为 1.0)
        """
        feats_old = np.asarray(feats_old, dtype=np.float32)
        feats_new = np.asarray(feats_new, dtype=np.float32)
        
        if feats_old.shape != feats_new.shape:
            raise ValueError(f"Feature shape mismatch: {feats_old.shape} != {feats_new.shape}")
        
        if feats_old.shape[0] < 2:
            # 样本太少无法计算，返回平移变换（R=None, s=1.0）
            D = feats_old.shape[1]
            return None, np.zeros(D, dtype=np.float32), np.zeros(D, dtype=np.float32), 1.0
        
        # 1. 计算均值
        mu_old = feats_old.mean(axis=0)
        mu_new = feats_new.mean(axis=0)
        
        # 2. 去中心化
        X = feats_old - mu_old
        Y = feats_new - mu_new
        
        # 3. 计算旋转 R (SVD) - 已禁用，只保留平移
        # min || s * X @ R - Y ||
        # 注意：为了只保留平移，将 R 设置为 None，s 设置为 1.0
        M = np.dot(X.T, Y)
        U, S, Vt = np.linalg.svd(M)
        R_computed = np.dot(U, Vt)
        
        if np.linalg.det(R_computed) < 0:
            Vt[-1, :] *= -1
            R_computed = np.dot(U, Vt)
        
        # 4. 计算缩放因子 s - 已禁用，设置为 1.0
        var_old = np.sum(np.square(X))
        trace_S = np.sum(S)
        if var_old < 1e-8:
            s_computed = 1.0
        else:
            s_computed = trace_S / var_old
        
        # ========== 只保留平移：禁用旋转和缩放 ==========
        # 将 R 设置为 None（表示不使用旋转）
        R = None
        # 将 s 设置为 1.0（表示不使用缩放）
        s = 1.0
        
        # ================= [Debug 信息] =================
        # 计算旧中心和新中心的欧氏距离
        shift_dist = np.linalg.norm(mu_new - mu_old)
        # 计算特征的平均模长 (用于归一化漂移量)
        avg_norm = np.mean(np.linalg.norm(feats_old, axis=1))
        # 计算旋转角度 (弧度) - 仅用于调试信息
        if R_computed is not None:
            trace_R = np.trace(R_computed)
            theta = np.arccos(np.clip((trace_R - (feats_old.shape[1] - 2)) / 2, -1, 1))
        else:
            theta = 0.0

        logging.info(f"[STAR DEBUG] Drift Analysis (Translation Only Mode):")
        logging.info(f"  > Shift (Mean Move): {shift_dist:.6f} (Avg Norm: {avg_norm:.2f})")
        logging.info(f"  > Scale Change (computed but disabled): {s_computed:.6f}")
        logging.info(f"  > Rotation Angle (computed but disabled): {theta:.6f}")
        logging.info(f"  > Using: Translation only (R=None, s=1.0)")
        
        # 如果漂移非常小，STAR 就不会有效果
        if shift_dist < 0.1:
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
                    # Support for DualPrompt: Pass class_id to feature_extractor if it supports it
                    try:
                        feats_new = self.feature_extractor(imgs, class_id=cls)
                    except TypeError:
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