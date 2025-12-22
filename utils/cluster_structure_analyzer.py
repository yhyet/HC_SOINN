"""
Cluster Structure Analyzer - 簇结构分析模块

用于验证在增量学习过程中，特征漂移时样本点簇内部结构是否发生改变。
通过计算Procrustes距离来量化簇结构的稳定性。

【设计原则】
- 独立于具体的模型实现（CodaPrompt, SimpleCIL 等）
- 通过回调函数接口与外部模型和数据交互
- 最小化外部依赖，只依赖numpy和torch
- 可插拔设计，易于在不同方法上复用
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
    
    【职责】
    1. 在Task 1结束后保存训练样本（图像和特征）
    2. 在后续任务中计算与Task 1的Procrustes距离
    3. 记录并保存分析结果
    
    【使用方式】
    ```python
    analyzer = ClusterStructureAnalyzer(
        feature_extractor=feature_fn,  # 回调函数：输入图像，输出特征
        device=device,
        args=args  # 配置参数（用于确定保存路径）
    )
    
    # 在Task 1结束后
    analyzer.save_task1_samples(
        dataset_loader=lambda: data_manager.get_dataset(...),
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # 在后续任务中
    analyzer.compute_procrustes_distances(cur_task)
    ```
    """
    
    def __init__(
        self,
        feature_extractor: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device,
        args: Dict,
    ):
        """
        初始化簇结构分析器
        
        参数:
            feature_extractor: 特征提取函数，输入 [B, C, H, W]，输出 [B, D]
            device: 计算设备
            args: 配置参数字典（用于确定保存路径等）
        """
        self.feature_extractor = feature_extractor
        self.device = device
        self.args = args
        
        # 存储结构：{cls: {'images': tensor [N_cls, C, H, W], 'feats_task1': np.ndarray [N_cls, D]}}
        self._cluster_samples: Dict[int, Dict[str, torch.Tensor]] = {}
        
        # 存储Procrustes距离结果：{task: {cls: distance}}
        self._procrustes_distances: Dict[int, Dict[int, float]] = {}
    
    def save_task1_samples(
        self,
        dataset_loader: Callable[[], torch.utils.data.Dataset],
        batch_size: int = 128,
        num_workers: int = 8,
    ) -> None:
        """
        在Task 1结束后，保存所有训练样本（图像和特征）
        用于后续任务计算Procrustes距离
        
        参数:
            dataset_loader: 数据集加载函数，返回Task 1的训练数据集
            batch_size: 批处理大小
            num_workers: 数据加载器的工作进程数
        """
        logging.info("[Cluster Structure Analysis] Saving Task 1 samples for Procrustes distance calculation...")
        
        # 获取Task 1的训练数据（类别0到init_cls-1）
        init_cls = self.args.get("init_cls", 10)
        task1_dataset = dataset_loader()
        task1_loader = DataLoader(
            task1_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers
        )
        
        # 提取特征并保存图像
        images_dict = {}  # {cls: list of images}
        feats_dict = {}  # {cls: list of features}
        
        with torch.no_grad():
            for _, (_, inputs, targets) in enumerate(task1_loader):
                inputs = inputs.to(self.device)
                targets = targets.cpu().numpy()
                
                # 提取特征
                features = self.feature_extractor(inputs)
                if isinstance(features, torch.Tensor):
                    features = features.cpu().numpy()
                else:
                    features = features
                
                # 按类别分组
                for i in range(len(targets)):
                    cls = int(targets[i])
                    if cls not in images_dict:
                        images_dict[cls] = []
                        feats_dict[cls] = []
                    images_dict[cls].append(inputs[i].cpu())
                    feats_dict[cls].append(features[i])
        
        # 转换为tensor和numpy数组
        for cls in images_dict:
            images_tensor = torch.stack(images_dict[cls])  # [N_cls, C, H, W]
            feats_array = np.array(feats_dict[cls])  # [N_cls, D]
            
            self._cluster_samples[cls] = {
                'images': images_tensor,
                'feats_task1': feats_array
            }
            logging.info(f"[Cluster Structure Analysis] Saved {len(images_dict[cls])} samples for class {cls}")
        
        logging.info(f"[Cluster Structure Analysis] Task 1 samples saved for {len(self._cluster_samples)} classes")
    
    def compute_procrustes_distances(self, cur_task: int) -> None:
        """
        在后续任务中，计算当前任务与Task 1之间的Procrustes距离
        
        参数:
            cur_task: 当前任务编号（0-based，Task 1对应cur_task=0）
        """
        if len(self._cluster_samples) == 0:
            logging.warning("[Cluster Structure Analysis] No Task 1 samples found, skipping Procrustes distance calculation")
            return
        
        logging.info(f"[Cluster Structure Analysis] Computing Procrustes distances for Task {cur_task + 1}...")
        
        # 初始化当前任务的距离字典
        if cur_task not in self._procrustes_distances:
            self._procrustes_distances[cur_task] = {}
        
        distances_summary = []
        
        for cls, data in self._cluster_samples.items():
            images = data['images']  # [N_cls, C, H, W]
            feats_task1 = data['feats_task1']  # [N_cls, D]
            
            # 使用当前模型重新提取特征
            feats_current = []
            with torch.no_grad():
                # 分批处理，避免内存溢出
                batch_size = 32
                for i in range(0, len(images), batch_size):
                    batch_images = images[i:i+batch_size].to(self.device)
                    batch_feats = self.feature_extractor(batch_images)
                    if isinstance(batch_feats, torch.Tensor):
                        batch_feats = batch_feats.cpu().numpy()
                    feats_current.append(batch_feats)
            
            feats_current = np.concatenate(feats_current, axis=0)  # [N_cls, D]
            
            # 计算Procrustes距离
            procrustes_dist = self._compute_procrustes_distance(feats_task1, feats_current)
            self._procrustes_distances[cur_task][cls] = procrustes_dist
            
            distances_summary.append({
                'task': cur_task + 1,
                'class': cls,
                'procrustes_distance': procrustes_dist,
                'num_samples': len(feats_task1)
            })
            
            logging.info(
                f"[Cluster Structure Analysis] Task {cur_task + 1}, Class {cls}: "
                f"Procrustes distance = {procrustes_dist:.6f} (N={len(feats_task1)})"
            )
        
        # 计算平均距离
        if len(distances_summary) > 0:
            avg_distance = np.mean([d['procrustes_distance'] for d in distances_summary])
            logging.info(
                f"[Cluster Structure Analysis] Task {cur_task + 1} average Procrustes distance: {avg_distance:.6f}"
            )
        
        # 保存结果到CSV文件
        self._save_procrustes_results()
    
    def _compute_procrustes_distance(self, X1: np.ndarray, X2: np.ndarray) -> float:
        """
        计算两个点集之间的Procrustes距离
        
        参数:
            X1: 参考点集 [N, D]
            X2: 目标点集 [N, D]
        
        返回:
            Procrustes距离（归一化后的Frobenius范数）
        """
        if X1.shape != X2.shape:
            raise ValueError(f"Shape mismatch: X1 {X1.shape} vs X2 {X2.shape}")
        
        if X1.shape[0] < 2:
            # 样本数太少，无法计算有意义的Procrustes距离
            return 0.0
        
        # 去中心化
        mu1 = X1.mean(axis=0)
        mu2 = X2.mean(axis=0)
        X1_centered = X1 - mu1  # [N, D]
        X2_centered = X2 - mu2  # [N, D]
        
        # 计算旋转矩阵 R (Orthogonal Procrustes)
        M = np.dot(X1_centered.T, X2_centered)  # [D, D]
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        R = np.dot(U, Vt)  # [D, D] 正交旋转矩阵
        
        # 计算最优缩放因子 s
        trace_X1 = np.trace(np.dot(X1_centered.T, X1_centered))
        if trace_X1 > 1e-10:
            s = float(np.sum(S) / trace_X1)
        else:
            # Fallback: 使用 Frobenius 范数比
            norm_X1 = np.linalg.norm(X1_centered, ord='fro')
            norm_X2 = np.linalg.norm(X2_centered, ord='fro')
            if norm_X1 > 1e-10:
                s = float(norm_X2 / norm_X1)
            else:
                s = 1.0
        
        # 计算变换后的X1
        X1_transformed = s * np.dot(X1_centered, R)  # [N, D]
        
        # 计算Procrustes距离（Frobenius范数）
        diff = X2_centered - X1_transformed
        procrustes_dist = np.linalg.norm(diff, ord='fro')
        
        # 归一化：除以X2的Frobenius范数
        norm_X2 = np.linalg.norm(X2_centered, ord='fro')
        if norm_X2 > 1e-10:
            normalized_dist = procrustes_dist / norm_X2
        else:
            normalized_dist = 0.0
        
        return float(normalized_dist)
    
    def _save_procrustes_results(self) -> None:
        """
        将Procrustes距离结果保存到CSV文件
        """
        if len(self._procrustes_distances) == 0:
            return
        
        # 确定保存路径（与日志文件在同一目录）
        init_cls = self.args.get("init_cls", 10)
        increment = self.args.get("increment", 10)
        dataset = self.args.get("dataset", "unknown")
        model_name = self.args.get("model_name", "unknown")
        
        log_dir = Path(f"logs/{model_name}/{dataset}/{init_cls}/{increment}")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path = log_dir / "procrustes_distances.csv"
        
        # 写入CSV
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Task', 'Class', 'Procrustes_Distance', 'Num_Samples'])
            
            for task, class_distances in sorted(self._procrustes_distances.items()):
                for cls, dist in sorted(class_distances.items()):
                    num_samples = len(self._cluster_samples[cls]['feats_task1'])
                    writer.writerow([task + 1, cls, f"{dist:.6f}", num_samples])
        
        logging.info(f"[Cluster Structure Analysis] Results saved to {csv_path}")
    
    def get_procrustes_distances(self) -> Dict[int, Dict[int, float]]:
        """
        获取Procrustes距离结果
        
        返回:
            {task: {cls: distance}} 格式的字典
        """
        return self._procrustes_distances.copy()
    
    def clear_samples(self) -> None:
        """
        清空保存的样本（用于释放内存）
        """
        self._cluster_samples.clear()
        logging.info("[Cluster Structure Analysis] Samples cleared")



