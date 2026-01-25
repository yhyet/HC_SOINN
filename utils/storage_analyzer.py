"""
存储占用分析器：统计不同配置下的存储空间占用

用于对比：
1. CodaPrompt (基础)
2. CodaPrompt + HC-SOINN
3. CodaPrompt + HC-SOINN + STAR
"""

import numpy as np
import logging
from typing import Dict, Any, Optional
import sys


class StorageAnalyzer:
    """存储占用分析器"""
    
    def __init__(self):
        self.stats = {}
    
    def analyze_model_storage(self, network) -> Dict[str, Any]:
        """
        分析模型参数存储占用
        
        Returns:
            dict: {
                'backbone_params': int,  # 参数量
                'prompt_params': int,
                'fc_params': int,
                'total_params': int,
                'total_size_mb': float  # 总大小（MB，假设float32）
            }
        """
        import torch.nn as nn
        
        # 获取模型（如果是DataParallel，需要获取module）
        model = network
        if isinstance(model, nn.DataParallel):
            model = model.module
        
        backbone_params = sum(p.numel() for p in model.backbone.parameters())
        prompt_params = sum(p.numel() for p in model.prompt.parameters()) if model.prompt is not None else 0
        fc_params = sum(p.numel() for p in model.fc.parameters())
        
        total_params = backbone_params + prompt_params + fc_params
        # 假设每个参数是float32 (4 bytes)
        total_size_mb = total_params * 4 / (1024 * 1024)
        
        return {
            'backbone_params': backbone_params,
            'prompt_params': prompt_params,
            'fc_params': fc_params,
            'total_params': total_params,
            'total_size_mb': total_size_mb
        }
    
    def analyze_hc_soinn_storage(self, hc_soinn) -> Dict[str, Any]:
        """
        分析HC-SOINN存储占用
        
        Returns:
            dict: {
                'num_classes': int,
                'total_nodes': int,
                'nodes_per_class': dict,  # {cls: num_nodes}
                'ncm_centers': int,  # 类别数
                'feature_dim': int,
                'storage_breakdown': {
                    'nodes_center': float,  # MB
                    'nodes_center_raw': float,  # MB
                    'nodes_count': float,  # MB
                    'ncm_mu': float,  # MB
                    'ncm_mu_raw': float,  # MB
                    'original_backup': float,  # MB (class_clusters_original)
                },
                'total_size_mb': float
            }
        """
        if not hasattr(hc_soinn, 'class_clusters'):
            return {
                'num_classes': 0,
                'total_nodes': 0,
                'nodes_per_class': {},
                'ncm_centers': 0,
                'feature_dim': 0,
                'storage_breakdown': {},
                'total_size_mb': 0.0
            }
        
        num_classes = len(hc_soinn.class_clusters)
        total_nodes = 0
        nodes_per_class = {}
        feature_dim = 0
        
        # 统计节点
        for cls, clusters in hc_soinn.class_clusters.items():
            num_nodes = len(clusters)
            nodes_per_class[cls] = num_nodes
            total_nodes += num_nodes
            if num_nodes > 0 and feature_dim == 0:
                # 从第一个节点获取特征维度
                feature_dim = clusters[0].center.shape[0] if hasattr(clusters[0], 'center') else 0
        
        # 如果没有节点，尝试从NCM中心获取维度
        if feature_dim == 0 and len(hc_soinn.class_mu) > 0:
            first_mu = next(iter(hc_soinn.class_mu.values()))
            feature_dim = first_mu.shape[0] if isinstance(first_mu, np.ndarray) else 0
        
        # 计算存储占用（假设float32，4 bytes）
        bytes_per_float = 4
        
        # 节点存储
        nodes_center_size = total_nodes * feature_dim * bytes_per_float  # normalized centers
        nodes_center_raw_size = total_nodes * feature_dim * bytes_per_float  # raw centers
        nodes_count_size = total_nodes * 4  # int32 counts
        
        # NCM中心存储
        ncm_centers = len(hc_soinn.class_mu)
        ncm_mu_size = ncm_centers * feature_dim * bytes_per_float  # normalized
        ncm_mu_raw_size = ncm_centers * feature_dim * bytes_per_float  # raw
        
        # 原始节点备份（class_clusters_original）
        original_backup_size = 0
        if hasattr(hc_soinn, 'class_clusters_original'):
            original_nodes = sum(len(clusters) for clusters in hc_soinn.class_clusters_original.values())
            original_backup_size = original_nodes * feature_dim * bytes_per_float * 2  # center + center_raw
        
        total_size_bytes = (
            nodes_center_size +
            nodes_center_raw_size +
            nodes_count_size +
            ncm_mu_size +
            ncm_mu_raw_size +
            original_backup_size
        )
        total_size_mb = total_size_bytes / (1024 * 1024)
        
        return {
            'num_classes': num_classes,
            'total_nodes': total_nodes,
            'nodes_per_class': nodes_per_class,
            'ncm_centers': ncm_centers,
            'feature_dim': feature_dim,
            'storage_breakdown': {
                'nodes_center_mb': nodes_center_size / (1024 * 1024),
                'nodes_center_raw_mb': nodes_center_raw_size / (1024 * 1024),
                'nodes_count_mb': nodes_count_size / (1024 * 1024),
                'ncm_mu_mb': ncm_mu_size / (1024 * 1024),
                'ncm_mu_raw_mb': ncm_mu_raw_size / (1024 * 1024),
                'original_backup_mb': original_backup_size / (1024 * 1024),
            },
            'total_size_mb': total_size_mb
        }
    
    def analyze_star_storage(self, star_aligner, image_shape: tuple = (224, 224, 3)) -> Dict[str, Any]:
        """
        分析STAR存储占用
        
        Args:
            star_aligner: STARAligner实例
            image_shape: 图像形状 (H, W, C)，默认(224, 224, 3)
        
        Returns:
            dict: {
                'num_classes': int,
                'total_anchors': int,
                'anchors_per_class': dict,  # {cls: num_anchors}
                'feature_dim': int,
                'storage_breakdown': {
                    'images': float,  # MB
                    'feats_ref': float,  # MB
                    'ema_delta': float,  # MB
                    'centers_raw_ref': float,  # MB (如果存在)
                },
                'total_size_mb': float
            }
        """
        if not hasattr(star_aligner, 'anchor_store') or len(star_aligner.anchor_store) == 0:
            return {
                'num_classes': 0,
                'total_anchors': 0,
                'anchors_per_class': {},
                'feature_dim': 0,
                'storage_breakdown': {},
                'total_size_mb': 0.0
            }
        
        num_classes = len(star_aligner.anchor_store)
        total_anchors = 0
        anchors_per_class = {}
        feature_dim = 0
        
        # 统计锚点
        for cls, anchor_data in star_aligner.anchor_store.items():
            if 'images' in anchor_data:
                num_anchors = len(anchor_data['images'])
                anchors_per_class[cls] = num_anchors
                total_anchors += num_anchors
                
                # 获取特征维度
                if feature_dim == 0 and 'feats_ref' in anchor_data:
                    feats_ref = anchor_data['feats_ref']
                    if isinstance(feats_ref, np.ndarray) and len(feats_ref.shape) > 0:
                        feature_dim = feats_ref.shape[1] if len(feats_ref.shape) > 1 else feats_ref.shape[0]
        
        # 计算存储占用
        bytes_per_float = 4
        bytes_per_uint8 = 1
        
        # 图像存储（假设uint8）
        H, W, C = image_shape
        image_size_bytes = H * W * C * bytes_per_uint8
        images_total_size = total_anchors * image_size_bytes
        
        # 特征存储（float32）
        feats_ref_size = total_anchors * feature_dim * bytes_per_float if feature_dim > 0 else 0
        
        # EMA delta存储（float32）
        ema_delta_size = total_anchors * feature_dim * bytes_per_float if feature_dim > 0 else 0
        
        # centers_raw_ref存储（如果存在，float32）
        centers_raw_ref_size = 0
        for anchor_data in star_aligner.anchor_store.values():
            if 'centers_raw_ref' in anchor_data:
                centers_raw_ref = anchor_data['centers_raw_ref']
                if isinstance(centers_raw_ref, np.ndarray):
                    centers_raw_ref_size += centers_raw_ref.size * bytes_per_float
        
        total_size_bytes = (
            images_total_size +
            feats_ref_size +
            ema_delta_size +
            centers_raw_ref_size
        )
        total_size_mb = total_size_bytes / (1024 * 1024)
        
        return {
            'num_classes': num_classes,
            'total_anchors': total_anchors,
            'anchors_per_class': anchors_per_class,
            'feature_dim': feature_dim,
            'storage_breakdown': {
                'images_mb': images_total_size / (1024 * 1024),
                'feats_ref_mb': feats_ref_size / (1024 * 1024),
                'ema_delta_mb': ema_delta_size / (1024 * 1024),
                'centers_raw_ref_mb': centers_raw_ref_size / (1024 * 1024),
            },
            'total_size_mb': total_size_mb
        }
    
    def analyze_all(self, network, hc_soinn=None, star_aligner=None, 
                   image_shape: tuple = (224, 224, 3)) -> Dict[str, Any]:
        """
        综合分析所有组件的存储占用
        
        Returns:
            dict: 完整的存储分析报告
        """
        results = {
            'model': self.analyze_model_storage(network),
            'hc_soinn': None,
            'star': None,
            'summary': {}
        }
        
        if hc_soinn is not None:
            results['hc_soinn'] = self.analyze_hc_soinn_storage(hc_soinn)
        
        if star_aligner is not None:
            results['star'] = self.analyze_star_storage(star_aligner, image_shape)
        
        # 计算总存储
        total_size_mb = results['model']['total_size_mb']
        if results['hc_soinn']:
            total_size_mb += results['hc_soinn']['total_size_mb']
        if results['star']:
            total_size_mb += results['star']['total_size_mb']
        
        results['summary'] = {
            'total_storage_mb': total_size_mb,
            'model_percentage': (results['model']['total_size_mb'] / total_size_mb * 100) if total_size_mb > 0 else 0,
            'hc_soinn_percentage': (results['hc_soinn']['total_size_mb'] / total_size_mb * 100) if results['hc_soinn'] and total_size_mb > 0 else 0,
            'star_percentage': (results['star']['total_size_mb'] / total_size_mb * 100) if results['star'] and total_size_mb > 0 else 0,
        }
        
        return results
    
    def print_report(self, results: Dict[str, Any], task_id: int = None):
        """打印存储分析报告"""
        print("\n" + "="*80)
        if task_id is not None:
            print(f"存储占用分析报告 - Task {task_id}")
        else:
            print("存储占用分析报告")
        print("="*80)
        
        # 模型存储
        model = results['model']
        print(f"\n【模型参数】")
        print(f"  Backbone: {model['backbone_params']:,} 参数 ({model['backbone_params']*4/(1024**2):.2f} MB)")
        print(f"  Prompt:   {model['prompt_params']:,} 参数 ({model['prompt_params']*4/(1024**2):.2f} MB)")
        print(f"  FC:       {model['fc_params']:,} 参数 ({model['fc_params']*4/(1024**2):.2f} MB)")
        print(f"  总计:     {model['total_params']:,} 参数 ({model['total_size_mb']:.2f} MB)")
        
        # HC-SOINN存储
        if results['hc_soinn']:
            hc = results['hc_soinn']
            print(f"\n【HC-SOINN】")
            print(f"  类别数: {hc['num_classes']}")
            print(f"  总节点数: {hc['total_nodes']}")
            print(f"  平均每类节点数: {hc['total_nodes']/hc['num_classes']:.1f}" if hc['num_classes'] > 0 else "  平均每类节点数: 0")
            print(f"  特征维度: {hc['feature_dim']}")
            print(f"  存储明细:")
            for key, value in hc['storage_breakdown'].items():
                print(f"    - {key}: {value:.4f} MB")
            print(f"  总计: {hc['total_size_mb']:.2f} MB")
        
        # STAR存储
        if results['star']:
            star = results['star']
            print(f"\n【STAR】")
            print(f"  类别数: {star['num_classes']}")
            print(f"  总锚点数: {star['total_anchors']}")
            print(f"  平均每类锚点数: {star['total_anchors']/star['num_classes']:.1f}" if star['num_classes'] > 0 else "  平均每类锚点数: 0")
            print(f"  特征维度: {star['feature_dim']}")
            print(f"  存储明细:")
            for key, value in star['storage_breakdown'].items():
                print(f"    - {key}: {value:.4f} MB")
            print(f"  总计: {star['total_size_mb']:.2f} MB")
        
        # 总结
        summary = results['summary']
        print(f"\n【总计】")
        print(f"  总存储: {summary['total_storage_mb']:.2f} MB")
        print(f"  模型占比: {summary['model_percentage']:.1f}%")
        if results['hc_soinn']:
            print(f"  HC-SOINN占比: {summary['hc_soinn_percentage']:.1f}%")
        if results['star']:
            print(f"  STAR占比: {summary['star_percentage']:.1f}%")
        
        print("="*80 + "\n")
        
        # 同时写入日志
        logging.info("="*80)
        if task_id is not None:
            logging.info(f"存储占用分析报告 - Task {task_id}")
        else:
            logging.info("存储占用分析报告")
        logging.info("="*80)
        logging.info(f"模型参数: {model['total_size_mb']:.2f} MB")
        if results['hc_soinn']:
            logging.info(f"HC-SOINN: {results['hc_soinn']['total_size_mb']:.2f} MB ({results['hc_soinn']['total_nodes']} nodes)")
        if results['star']:
            logging.info(f"STAR: {results['star']['total_size_mb']:.2f} MB ({results['star']['total_anchors']} anchors)")
        logging.info(f"总计: {summary['total_storage_mb']:.2f} MB")

