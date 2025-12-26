import torch
import numpy as np
import logging
from typing import Dict, Optional, Callable, Set, Any
from torch.utils.data import DataLoader

# 假设 HCSOINNClassifier 在 utils.hc_soinn_classifier 中
# 为了避免循环引用，这里只做类型提示
# from .hc_soinn_classifier import HCSOINNClassifier

class STARAlignment:
    """
    STAR (Structure-Topology Alignment via Residuals) - Refactored Version
    
    核心理念：
    1. 每次 Task 结束时，对所有旧类别进行特征对齐。
    2. 始终以【原始状态】（首次学习该类别时的状态）为基准，计算到【当前状态】的变换。
    3. 避免累积误差：T_current = Align(Feats_original -> Feats_current)
    4. 应用变换：Nodes_current = T_current(Nodes_original)
    """
    
    def __init__(
        self,
        hc_soinn: Any, # HCSOINNClassifier
        feature_extractor: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device,
        use_full_task_rehearsal: bool = False,
    ):
        self.hc_soinn = hc_soinn
        self.feature_extractor = feature_extractor
        self.device = device
        self.use_full_task_rehearsal = use_full_task_rehearsal
        
        # 核心存储：只存储类别首次出现时的原始锚点信息
        # Key: class_id
        # Value: {
        #    'images': Tensor [K, C, H, W],  # 原始图片（不变）
        #    'feats_orig': np.ndarray [K, D] # 原始特征（Task 0/Initial 提取，不变）
        # }
        self.anchor_store: Dict[int, Dict[str, Any]] = {}
        
        # Track aligned classes to implement "Align Once" policy
        self.aligned_classes: Set[int] = set()
        
        logging.info("[STAR] Initialized (Refactored Clean Version)")

    def select_anchors_for_current_task(
        self,
        dataset,
        batch_size: int = 128,
        num_workers: int = 4
    ) -> None:
        """
        为当前任务的新类别选择并保存原始锚点。
        这些锚点将作为未来所有对齐操作的基准（Source）。
        """
        # 精简日志：只在开始时输出一次
        if self.use_full_task_rehearsal:
            logging.info(f"[STAR] Selecting anchors: Full Rehearsal mode")
        # else: 锚点模式不需要额外日志
        
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        # 1. 提取所有样本特征
        all_feats = []
        all_imgs = []
        all_targets = []
        
        # self.feature_extractor is a function, not a model, so no .eval() method
        # self.feature_extractor.eval() 
        with torch.no_grad():
            for _, inputs, targets in loader:
                inputs = inputs.to(self.device)
                feats = self.feature_extractor(inputs)
                
                if isinstance(feats, torch.Tensor):
                    feats = feats.cpu().numpy()
                
                all_feats.append(feats)
                all_imgs.append(inputs.cpu())
                all_targets.append(targets.numpy())
                
        all_feats = np.concatenate(all_feats, axis=0) # [N, D]
        all_imgs = torch.cat(all_imgs, dim=0)         # [N, C, H, W]
        all_targets = np.concatenate(all_targets, axis=0)
        
        current_classes = np.unique(all_targets)
        
        for cls in current_classes:
            if cls in self.anchor_store:
                continue # 已经存在（理论上不应该发生）
                
            cls_mask = (all_targets == cls)
            cls_feats = all_feats[cls_mask]
            cls_imgs = all_imgs[cls_mask]
            
            # 选择锚点
            if self.use_full_task_rehearsal:
                # 全量模式：保存所有样本
                anchors_feats = cls_feats
                anchors_imgs = cls_imgs
            else:
                # 锚点模式：选择 SOINN 节点 + NCM 中心对应的最近邻
                # 1. 获取该类的 SOINN 节点和 NCM 中心（原始未归一化的）
                target_points = []
                
                # SOINN 节点
                if cls in self.hc_soinn.class_clusters:
                    # 注意：这里我们获取的是刚刚压缩完的节点，它们就是“原始节点”
                    target_points.extend([c.center_raw for c in self.hc_soinn.class_clusters[cls]])
                
                # NCM 中心
                if cls in self.hc_soinn.class_mu_raw:
                    target_points.append(self.hc_soinn.class_mu_raw[cls])
                    
                if not target_points:
                    logging.warning(f"[STAR] No SOINN nodes or NCM center found for class {cls}")
                    continue
                    
                target_points = np.array(target_points) # [M, D]
                
                # 2. 寻找最近邻
                # 在归一化特征上找最近邻（使用余弦相似度）
                tgt_norm = target_points / (np.linalg.norm(target_points, axis=1, keepdims=True) + 1e-8)
                src_norm = cls_feats / (np.linalg.norm(cls_feats, axis=1, keepdims=True) + 1e-8)
                
                sim_matrix = np.dot(tgt_norm, src_norm.T) # [M, N_cls]
                nearest_indices = np.argmax(sim_matrix, axis=1)
                
                unique_indices = np.unique(nearest_indices)
                anchors_feats = cls_feats[unique_indices]
                anchors_imgs = cls_imgs[unique_indices]

            # 保存到 anchor_store
            self.anchor_store[cls] = {
                'images': anchors_imgs,
                'feats_orig': anchors_feats # 原始特征，作为 Source
            }
        
        # 精简日志：只输出总结信息
        if current_classes.size > 0:
            total_anchors = sum(len(self.anchor_store[cls]['feats_orig']) for cls in current_classes if cls in self.anchor_store)
            logging.info(f"[STAR] Selected anchors for {len(current_classes)} classes (total: {total_anchors} anchors)")

    def align_old_classes(self, cur_task: int, current_task_classes: Optional[Set[int]] = None) -> None:
        """
        对所有旧类别进行对齐。
        
        冻结策略：
        - class 0-9: 在 Task 1 结束时冻结
        - class 10-19: 在 Task 2 结束时冻结
        - class 20-29: 在 Task 3 结束时冻结
        - 以此类推：class (k*10) 到 (k*10+9) 在 Task k 结束时冻结
        """
        frozen_classes = []
        restored_classes = []
        new_frozen_classes = []
        
        for cls, data in self.anchor_store.items():
            # 跳过当前任务的新类别（不需要对齐）
            if current_task_classes and cls in current_task_classes:
                continue
            
            # 计算该类应该在哪一个任务冻结
            # class 0-9 -> Task 1, class 10-19 -> Task 2, class 20-29 -> Task 3, ...
            freeze_task = (cls // 10) + 1
            
            # --- 按任务冻结策略 ---
            # 如果当前任务 >= 该类应该冻结的任务，则跳过对齐并恢复冻结状态
            if cur_task >= freeze_task:
                if cls not in self.aligned_classes:
                    # 首次冻结：标记为已对齐并冻结
                    if hasattr(self.hc_soinn, 'freeze_nodes'):
                        self.hc_soinn.freeze_nodes(cls)
                    self.aligned_classes.add(cls)
                    new_frozen_classes.append(cls)
                else:
                    # 已经冻结过：恢复冻结状态
                    if hasattr(self.hc_soinn, 'restore_frozen_nodes'):
                        self.hc_soinn.restore_frozen_nodes(cls)
                    restored_classes.append(cls)
                continue
            # -------------------------
                
            # 1. 准备数据
            imgs = data['images']          # [K, C, H, W]
            feats_orig = data['feats_orig'] # [K, D] (Source)
            
            # 2. 提取当前特征 (Target)
            # 使用当前模型提取锚点图片的特征
            with torch.no_grad():
                imgs = imgs.to(self.device)
                feats_curr = self.feature_extractor(imgs) # [K, D]
                if isinstance(feats_curr, torch.Tensor):
                    feats_curr = feats_curr.cpu().numpy()
            
            # 3. 计算变换 (Procrustes)
            # T: feats_orig -> feats_curr
            # Work in UNNORMALIZED space
            
            mu_orig = feats_orig.mean(axis=0)
            mu_curr = feats_curr.mean(axis=0)
            
            X_orig = feats_orig - mu_orig
            X_curr = feats_curr - mu_curr
            
            # SVD: M = X_orig^T @ X_curr
            M = np.dot(X_orig.T, X_curr)
            U, S, Vt = np.linalg.svd(M, full_matrices=False)
            R = np.dot(U, Vt)
            
            # 4. 评估误差
            X_aligned = np.dot(X_orig, R)
            diff_before = X_curr - X_orig
            diff_after = X_curr - X_aligned
            
            err_before = np.linalg.norm(diff_before) / np.sqrt(len(feats_orig))
            err_after = np.linalg.norm(diff_after) / np.sqrt(len(feats_orig))
            
            total_error_before += err_before
            total_error_after += err_after
            
            # 5. 应用变换到 HC-SOINN 节点
            # 计算该类应该在哪一个任务冻结
            freeze_task = (cls // 10) + 1
            
            # 如果当前任务正好是该类应该冻结的任务，则跳过对齐并直接冻结
            if cur_task == freeze_task:
                if hasattr(self.hc_soinn, 'freeze_nodes'):
                    self.hc_soinn.freeze_nodes(cls)
                self.aligned_classes.add(cls)
                new_frozen_classes.append(cls)
            elif hasattr(self.hc_soinn, 'apply_rigid_transform'):
                # 正常对齐流程（如果未来需要启用对齐）
                # 当前使用纯冻结模式，不执行对齐
                pass
            else:
                logging.error("[STAR] HC-SOINN does not have apply_rigid_transform method!")
        
        # 精简日志输出：只输出关键信息
        if new_frozen_classes:
            logging.info(f"[STAR] Task {cur_task}: Frozen {len(new_frozen_classes)} new classes: {new_frozen_classes[:5]}{'...' if len(new_frozen_classes) > 5 else ''}")
        if restored_classes:
            logging.info(f"[STAR] Task {cur_task}: Restored {len(restored_classes)} frozen classes")
