"""
Cluster Structure Analyzer - 簇结构分析模块 (修正版: 移除归一化)

修正说明：
1. 移除了 save_task1_samples 和 compute_procrustes_distances 中的 L2 Normalization。
2. _compute_procrustes_distance 增加了缩放因子 s 的计算，以支持带缩放的刚性对齐。
"""

import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, Callable, Optional, Tuple
from pathlib import Path
import csv


class ClusterStructureAnalyzer:
    """
    簇结构分析器
    """
    
    def __init__(
        self,
        feature_extractor: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device,
        args: Dict,
    ):
        self.feature_extractor = feature_extractor
        self.device = device
        self.args = args
        self._cluster_samples: Dict[int, Dict[str, torch.Tensor]] = {}
        self._procrustes_distances: Dict[int, Dict[int, float]] = {}
    
    def save_task1_samples(
        self,
        dataset_loader: Callable[[], torch.utils.data.Dataset],
        batch_size: int = 128,
        num_workers: int = 8,
    ) -> None:
        logging.info("[Cluster Structure Analysis] Saving Task 1 samples (Raw Features)...")
        
        init_cls = self.args.get("init_cls", 10)
        task1_dataset = dataset_loader()
        task1_loader = DataLoader(
            task1_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers
        )
        
        images_dict = {} 
        feats_dict = {} 
        
        with torch.no_grad():
            for _, (_, inputs, targets) in enumerate(task1_loader):
                inputs = inputs.to(self.device)
                targets = targets.cpu().numpy()
                
                features = self.feature_extractor(inputs)
                if isinstance(features, torch.Tensor):
                    features = features.cpu().numpy()
                
                for i in range(len(targets)):
                    cls = int(targets[i])
                    if cls not in images_dict:
                        images_dict[cls] = []
                        feats_dict[cls] = []
                    images_dict[cls].append(inputs[i].cpu())
                    feats_dict[cls].append(features[i])
        
        for cls in images_dict:
            images_tensor = torch.stack(images_dict[cls])
            feats_array = np.array(feats_dict[cls])
            
            # 【修正】直接保存原始特征，不进行归一化
            # feats_array_norm = feats_array / (np.linalg.norm(feats_array, axis=1, keepdims=True) + 1e-8)
            
            # 【关键修复】使用 copy() 确保 feats_task1 不会被意外修改
            # 如果直接赋值，如果某个地方修改了数组，会影响保存的值
            self._cluster_samples[cls] = {
                'images': images_tensor,
                'feats_task1': feats_array.copy()  # Raw features (deep copy to prevent accidental modification)
            }
            logging.info(f"[Cluster Structure Analysis] Saved {len(images_dict[cls])} samples for class {cls} (RAW)")
        
        logging.info(f"[Cluster Structure Analysis] Task 1 samples saved for {len(self._cluster_samples)} classes")
    
    def compute_procrustes_distances(self, cur_task: int) -> None:
        if len(self._cluster_samples) == 0:
            logging.warning("[Cluster Structure Analysis] No Task 1 samples found, skipping")
            return
        
        logging.info(f"[Cluster Structure Analysis] Computing Procrustes distances for Task {cur_task + 1}...")
        
        if cur_task not in self._procrustes_distances:
            self._procrustes_distances[cur_task] = {}
        
        distances_summary = []
        
        for cls, data in self._cluster_samples.items():
            images = data['images']
            feats_task1 = data['feats_task1']
            
            feats_current = []
            with torch.no_grad():
                batch_size = 32
                for i in range(0, len(images), batch_size):
                    batch_images = images[i:i+batch_size].to(self.device)
                    batch_feats = self.feature_extractor(batch_images)
                    if isinstance(batch_feats, torch.Tensor):
                        batch_feats = batch_feats.cpu().numpy()
                    feats_current.append(batch_feats)
            
            feats_current = np.concatenate(feats_current, axis=0)
            
            # 【调试】验证特征是否真的不同
            # 检查 feats_task1 和 feats_current 的第一个样本的第一个特征值
            if len(feats_task1) > 0 and len(feats_current) > 0:
                feat1_first_val = feats_task1[0, 0] if feats_task1.ndim > 1 else feats_task1[0]
                feat_curr_first_val = feats_current[0, 0] if feats_current.ndim > 1 else feats_current[0]
                feat_diff = np.abs(feat1_first_val - feat_curr_first_val)
                feat1_mean = np.mean(feats_task1)
                feat_curr_mean = np.mean(feats_current)
                feat_mean_diff = np.abs(feat1_mean - feat_curr_mean)
                
                # 检查是否完全相同（考虑浮点误差）
                are_identical = np.allclose(feats_task1, feats_current, rtol=1e-8, atol=1e-8)
                
                logging.warning(
                    f"[Cluster Structure Analysis DEBUG] Class {cls}: "
                    f"feats_task1[0,0]={feat1_first_val:.8f}, "
                    f"feats_current[0,0]={feat_curr_first_val:.8f}, "
                    f"diff={feat_diff:.8f}, "
                    f"mean_diff={feat_mean_diff:.8f}, "
                    f"are_identical={are_identical}"
                )
                
                if are_identical:
                    logging.error(
                        f"[Cluster Structure Analysis ERROR] Class {cls}: "
                        f"feats_task1 and feats_current are IDENTICAL! "
                        f"This indicates a bug: either feats_task1 was incorrectly updated, "
                        f"or feature_extractor is not using the current task's model weights."
                    )
            
            # 【修正】不进行归一化
            # feats_current_norm = feats_current / (np.linalg.norm(feats_current, axis=1, keepdims=True) + 1e-8)
            
            # 计算带 Scale 的 Procrustes 距离
            procrustes_dist, scale, angle_rad = self._compute_procrustes_distance(feats_task1, feats_current)
            self._procrustes_distances[cur_task][cls] = procrustes_dist
            
            distances_summary.append({
                'task': cur_task + 1,
                'class': cls,
                'procrustes_distance': procrustes_dist,
                'num_samples': len(feats_task1),
                'scale': scale,
                'angle': angle_rad
            })
            
            logging.info(
                f"[Cluster Structure Analysis] T{cur_task + 1}, C{cls}: "
                f"Dist={procrustes_dist:.4f}, Scale={scale:.4f}, Angle={angle_rad:.4f}"
            )
        
        if len(distances_summary) > 0:
            avg_distance = np.mean([d['procrustes_distance'] for d in distances_summary])
            avg_scale = np.mean([d['scale'] for d in distances_summary])
            logging.info(
                f"[Cluster Structure Analysis] Task {cur_task + 1} Avg Dist: {avg_distance:.4f}, Avg Scale: {avg_scale:.4f}"
            )
        
        self._save_procrustes_results()
    
    def _compute_procrustes_distance(self, X1: np.ndarray, X2: np.ndarray) -> Tuple[float, float, float]:
        """
        计算原始空间中的广义 Procrustes 距离 (包含 Scale)
        返回: (Distance, Scale, Rotation_Angle_Estimate)
        """
        if X1.shape != X2.shape:
            raise ValueError(f"Shape mismatch: X1 {X1.shape} vs X2 {X2.shape}")
        if X1.shape[0] < 2:
            return 0.0, 1.0, 0.0
        
        # 【调试】检查输入是否完全相同
        are_inputs_identical = np.allclose(X1, X2, rtol=1e-8, atol=1e-8)
        if are_inputs_identical:
            logging.warning(
                f"[Procrustes DEBUG] Input matrices X1 and X2 are IDENTICAL! "
                f"This will result in Dist=0, Scale=1, Angle=π (due to det(R) correction)."
            )
        
        # 1. 去中心化
        mu1 = X1.mean(axis=0)
        mu2 = X2.mean(axis=0)
        X1_centered = X1 - mu1
        X2_centered = X2 - mu2
        
        # 【调试】检查中心化后是否相同
        are_centered_identical = np.allclose(X1_centered, X2_centered, rtol=1e-8, atol=1e-8)
        if are_centered_identical and not are_inputs_identical:
            logging.warning(
                f"[Procrustes DEBUG] After centering, X1_centered and X2_centered are IDENTICAL! "
                f"This suggests the difference is only in the mean."
            )
        
        # 2. 计算旋转 R
        M = np.dot(X1_centered.T, X2_centered)
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        R = np.dot(U, Vt)
        det_R_before = np.linalg.det(R)
        if det_R_before < 0:
            Vt[-1, :] *= -1
            R = np.dot(U, Vt)
            det_R_after = np.linalg.det(R)
            # 【调试】如果触发了镜像修正，说明数据可能完全相同或高度对称
            if are_centered_identical or are_inputs_identical:
                logging.warning(
                    f"[Procrustes DEBUG] Mirror correction triggered: det(R)={det_R_before:.6f} -> {det_R_after:.6f}. "
                    f"This is expected when X1 and X2 are identical, resulting in Angle=π."
                )
            
        # 3. 计算缩放 s
        # s = trace(X2_centered.T @ X1_centered @ R) / trace(X1_centered.T @ X1_centered)
        # trace(A @ B) = sum(elementwise_multiplication)
        numerator = np.sum(S) # trace(S) comes from SVD
        denominator = np.sum(X1_centered ** 2)
        s = numerator / (denominator + 1e-8)
        
        # 4. 对齐: Y_aligned = s * (X1 - mu1) @ R + mu2
        # 我们比较 X2 和 对齐后的 X1
        # 在中心化空间比较即可: X2_c vs s * X1_c @ R
        X1_transformed = s * np.dot(X1_centered, R)
        residual = X2_centered - X1_transformed
        
        # 5. 计算距离 (归一化以便跨任务比较)
        # 通常除以 sqrt(N) 或 原始数据的范数
        norm_factor = np.sqrt(np.sum(X2_centered**2)) + 1e-8 # 相对于目标数据的 Scale 归一化
        # 或者使用绝对 MSE: np.sqrt(np.mean(residual**2))
        residual_norm = np.linalg.norm(residual, ord='fro')
        procrustes_dist = residual_norm / norm_factor
        
        # 【调试】如果距离为 0，输出详细信息
        if procrustes_dist < 1e-6:
            logging.warning(
                f"[Procrustes DEBUG] Procrustes distance is near zero: "
                f"residual_norm={residual_norm:.8f}, norm_factor={norm_factor:.8f}, "
                f"dist={procrustes_dist:.8f}. This suggests X1 and X2 are nearly identical."
            )
        
        # 6. 估算旋转角度 (参考)
        trace_R = np.trace(R)
        D = X1.shape[1]
        # cos(theta) = (Tr(R) - (D-2)) / 2  [Approx for high dim]
        cos_theta = (trace_R - (D - 2)) / 2.0
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        
        return float(procrustes_dist), float(s), float(angle)
    
    def _save_procrustes_results(self) -> None:
        if len(self._procrustes_distances) == 0: return
        
        init_cls = self.args.get("init_cls", 10)
        increment = self.args.get("increment", 10)
        dataset = self.args.get("dataset", "unknown")
        model_name = self.args.get("model_name", "unknown")
        
        log_dir = Path(f"logs/{model_name}/{dataset}/{init_cls}/{increment}")
        log_dir.mkdir(parents=True, exist_ok=True)
        csv_path = log_dir / "procrustes_distances.csv"
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Task', 'Class', 'Procrustes_Dist', 'Num_Samples'])
            for task, class_distances in sorted(self._procrustes_distances.items()):
                for cls, dist in sorted(class_distances.items()):
                    num_samples = len(self._cluster_samples[cls]['feats_task1'])
                    writer.writerow([task + 1, cls, f"{dist:.6f}", num_samples])