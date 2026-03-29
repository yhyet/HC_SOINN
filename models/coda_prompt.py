import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.optim import Optimizer
import math
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import CodaPromptVitNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
from utils.knn_classifier import KNNClassifier
from utils.soinn_classifier import SOINNClassifier
from utils.hc_soinn_classifier import HCSOINNClassifier
from utils.cluster_structure_analyzer import ClusterStructureAnalyzer
from utils.STAR import STARAligner
from utils.storage_analyzer import StorageAnalyzer
import os
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
try:
    # import umap
    # not use
    UMAP_AVAILABLE = False
except ImportError:
    UMAP_AVAILABLE = False
    logging.warning("UMAP not available, falling back to t-SNE")

# tune the model at first session with vpt, and then conduct simple shot.
num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
    
        # KAC分类器支持
        self.use_kac = args.get("use_kac", False)
        if self.use_kac:
            logging.info("KAC classifier enabled")
            kac_config = args.get("kac_config", {})
            logging.info(f"KAC config: grid_min={kac_config.get('grid_min', -2.0)}, "
                        f"grid_max={kac_config.get('grid_max', 2.0)}, "
                        f"num_grids={kac_config.get('num_grids', 16)}")
        
        self._network = CodaPromptVitNet(args, True)

        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.args = args
        # KNN plugin
        self.use_knn = args.get("use_knn", False)
        self.knn_upperbound = args.get("knn_upperbound", False)  # KNN上限测试模式
        # 是否启用基于使用频率的渐进式剪枝（仅对当前 task 的类别执行）
        self.knn_prune_zero_usage = args.get("knn_prune_zero_usage", False)
        if self.use_knn:
            knn_metric = args.get("knn_metric", "euclidean")
            logging.info(f"Initializing KNNClassifier with metric: {knn_metric}")
            if self.knn_upperbound:
                logging.info("KNN Upperbound mode enabled: will rebuild KNN bank with all seen tasks at eval time")
            self.knn = KNNClassifier(
                metric=knn_metric,
                use_all_samples=args.get("knn_use_all", True),
                k_neighbors=args.get("knn_k", None),
            )
            logging.info(f"KNNClassifier initialized with metric: {self.knn.metric}")
        
        # SOINN / HC-SOINN plugins（互斥，优先 HC-SOINN）
        self.use_hc_soinn = args.get("use_hc_soinn", False)
        self.use_soinn = args.get("use_soinn", False) and not self.use_hc_soinn
        if self.use_hc_soinn:
            logging.info("Initializing HC-SOINNClassifier")
            self.hc_soinn = HCSOINNClassifier(
                max_prototypes_per_class=args.get("hcsoinn_max_proto_per_class", 20),
                alpha=args.get("hcsoinn_alpha", 0.5),
                tau_merge=args.get("hcsoinn_tau_merge", 0.2),
                tau_reject=args.get("hcsoinn_tau_reject", 2.0),
                linkage_method=args.get("hcsoinn_linkage", "average"),
                distance_metric=args.get("hcsoinn_distance", "cosine"),
                use_soinn_refinement=args.get("hcsoinn_use_soinn_refinement", True),
                soinn_ad=args.get("hcsoinn_soinn_ad", 20),
                soinn_lam=args.get("hcsoinn_soinn_lam", 20),
                soinn_threshold_scale=args.get("hcsoinn_soinn_threshold_scale", 0.5),
                soinn_max_iter=args.get("hcsoinn_soinn_max_iter", 3),
                soinn_max_degree_for_removal=args.get("hcsoinn_soinn_max_degree_for_removal", 1),
                coarse_topk=args.get("hcsoinn_coarse_topk", None),
            )
        elif self.use_soinn:
            # 处理 seed 参数：如果传入的是列表，取第一个元素
            seed = args.get("seed", None)
            if isinstance(seed, (list, tuple)) and len(seed) > 0:
                seed = seed[0]
            
            logging.info("Initializing SOINNClassifier")
            self.soinn = SOINNClassifier(
                ad=args.get("soinn_ad", 20),
                lam=args.get("soinn_lam", 20),
                max_nodes_per_class=args.get("soinn_max_nodes_per_class", None),
                seed=seed,
                threshold_scale=args.get("soinn_threshold_scale", 0.3),
                k_neighbors=args.get("soinn_k_neighbors", 3),
            )
            logging.info(f"SOINNClassifier initialized with k_neighbors={self.soinn.k_neighbors}")
        
        total_params = sum(p.numel() for p in self._network.parameters())
        logging.info(f'{total_params:,} total parameters.')
        
        # 计算可训练参数（支持KAC分类器）
        fc_params = sum(p.numel() for p in self._network.fc.parameters() if p.requires_grad)
        prompt_params = sum(p.numel() for p in self._network.prompt.parameters() if p.requires_grad)
        total_trainable_params = fc_params + prompt_params
        logging.info(f'{total_trainable_params:,} fc and prompt training parameters.')
        if self.use_kac:
            logging.info(f'  - FC (KAC) parameters: {fc_params:,}')
            logging.info(f'  - Prompt parameters: {prompt_params:,}')
        
        # 实验模式：只测试第一个任务的类别（用于分析特征漂移 vs 新类别干扰）
        self.test_only_first_task_classes = args.get("test_only_first_task_classes", False)
        if self.test_only_first_task_classes:
            self.init_cls = args.get("init_cls", 10)
        
        # 簇结构分析实验：验证特征漂移时簇内部结构是否改变
        self.analyze_cluster_structure_drift = args.get("analyze_cluster_structure_drift", False)
        self.cluster_analyzer = None  # 簇结构分析器（延迟初始化）
        
        if self.analyze_cluster_structure_drift:
            logging.info("簇结构分析实验启用：将计算Procrustes距离验证特征漂移时簇结构是否改变")
            # 定义特征提取函数（适配 CodaPrompt 的网络结构）
            def feature_extractor(x):
                if isinstance(self._network, nn.DataParallel):
                    feats = self._network.module.extract_vector(x)
                else:
                    feats = self._network.extract_vector(x)
                return feats
            
            self.cluster_analyzer = ClusterStructureAnalyzer(
                feature_extractor=feature_extractor,
                device=self._device,
                args=args
            )

        # STAR 特征漂移对齐配置
        self.use_feature_alignment = args.get("use_feature_alignment", False)
        self.use_full_task_rehearsal = args.get("use_full_task_rehearsal", False)
        self.star = None  # STAR 对齐器（延迟初始化）
        
        # 如果启用特征对齐且使用 HC-SOINN，初始化 STAR
        if self.use_feature_alignment and self.use_hc_soinn:
            # 定义特征提取函数（适配 CodaPrompt 的网络结构）
            def feature_extractor(x):
                if isinstance(self._network, nn.DataParallel):
                    feats = self._network.module(x, pen=True, train=False)
                else:
                    feats = self._network(x, pen=True, train=False)
                
                if isinstance(feats, tuple):
                    feats = feats[0]
                return feats
            
            self.star = STARAligner(
                hc_soinn=self.hc_soinn,
                feature_extractor=feature_extractor,
                device=self._device,
                use_full_task_rehearsal=self.use_full_task_rehearsal,
                star_mode=args.get("star_mode", "rigid"),
                star_lambda=args.get("star_lambda", 0.3),
            )
            if self.use_full_task_rehearsal:
                logging.info("STAR alignment initialized (FULL TASK REHEARSAL mode - for performance upper bound)")
            else:
                logging.info("STAR alignment initialized (anchor mode: all SOINN nodes + NCM points)")
        
        # NCM分类器支持
        self.use_ncm = args.get("use_ncm", True)  # 默认开启，保持向后兼容
        if not self.use_ncm:
            logging.info("NCM classifier disabled")
        # NCM分类器：初始化类均值存储
        self._class_means = None
        
        # 存储分析器（用于存储占用测试）
        self.enable_storage_analysis = args.get("enable_storage_analysis", False)
        if self.enable_storage_analysis:
            self.storage_analyzer = StorageAnalyzer()
            logging.info("Storage analysis enabled")
        # feature_dim 是 BaseLearner 的 @property，会从 self._network.feature_dim 获取
        # 不需要在这里设置，CodaPromptVitNet 中已经有 feature_dim 属性

    def after_task(self):
        """
        每个 task 结束后的处理流程（STAR 特征漂移对齐的核心入口）
        
        【Pipeline 流程说明】
        ┌─────────────────────────────────────────────────────────────────┐
        │ Task t 训练结束，Backbone 已更新为 f_t                          │
        └─────────────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────────────────────────────┐
        │ Step 1: 漂移对齐 (Drift Alignment)                              │
        │   - 对旧类别 (C_0, ..., C_{t-1}) 的 SOINN 节点进行刚性变换校正  │
        │   - 使用保存的锚点计算 Procrustes 变换 (R, mu_old, mu_new)      │
        │   - 将旧节点从 f_{t-1} 空间对齐到 f_t 空间（Plan B：从原始节点） │
        └─────────────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────────────────────────────┐
        │ Step 2: 压缩 (Compress)                                          │
        │   - 对当前任务 (C_t) 的训练数据生成 SOINN 节点                  │
        │   - 使用层次聚类压缩，生成有限数量的原型                        │
        └─────────────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────────────────────────────┐
        │ Step 3: 锚点选择 (Anchor Selection)                              │
        │   - 为当前任务 (C_t) 的每个类别选择锚点（全量拓扑映射）          │
        │   - 基于 SOINN 节点和 NCM 中心选择最近邻样本                     │
        │   - 保存锚点图片和特征，用于下一轮 (Task t+1) 的漂移对齐        │
        └─────────────────────────────────────────────────────────────────┘
        
        【关键设计】
        - 顺序很重要：必须先对齐旧类，再压缩新类，最后选锚点
        - Plan B：对齐操作始终从 class_clusters_original 开始，避免累积误差
        - 链式覆盖：对齐后更新参考特征，确保对齐链的连续性
        """
        # ========== Prepare current task class set ==========
        # Get the classes that belong to the current task (to exclude from alignment)
        current_task_classes = set(range(self._known_classes, self._total_classes))
        
        # ========== Step 1: 特征漂移对齐（针对旧类别）==========
        # 注意：对齐已在 _build_classifiers() 中提前执行（在评估前）
        # 这里不再重复对齐，只处理新类别的压缩和锚点选择
        # 对齐时机修复：在评估前对齐，确保评估时使用的是对齐后的节点
        
        # ========== Step 2: 为当前任务选择锚点（用于下一轮对齐）==========
        # 注意：HC-SOINN压缩已在_build_hc_soinn_bank中完成，这里不再重复压缩
        # 目的：为当前任务的每个类别选择锚点，保存用于下一轮漂移对齐
        # 使用 STAR 模块进行锚点选择（全量拓扑映射）
        if self.star is not None:
            # 获取当前任务的训练数据集
            dataset = self.data_manager.get_dataset(
                np.arange(self._known_classes, self._total_classes),
                source="train", mode="test"
            )
            self.star.select_anchors_for_current_task(
                dataset=dataset,
                batch_size=self.batch_size,
                num_workers=num_workers,
                current_task_classes=current_task_classes
            )

        # 4. 基于使用频率的渐进式剪枝（只剪当前 task 的类别，可通过开关控制）
        if getattr(self, "use_knn", False) and getattr(self, "knn_prune_zero_usage", False):
            try:
                if hasattr(self, "knn") and hasattr(self.knn, "prune_zero_usage"):
                    current_task_classes = list(range(self._known_classes, self._total_classes))
                    if len(current_task_classes) > 0:
                        logging.info(
                            f"[coda_prompt] Pruning KNN nodes with zero usage for current task classes: "
                            f"{current_task_classes} (task {self._cur_task})"
                        )
                        self.knn.prune_zero_usage(current_task_classes)
            except Exception as e:
                logging.error(f"Error during KNN prune_zero_usage in coda_prompt.after_task: {e}", exc_info=True)

        # 5. 簇结构分析实验：保存Task 1样本或计算Procrustes距离
        if self.cluster_analyzer is not None:
            if self._cur_task == 0:
                # Task 1结束后：保存所有训练样本
                init_cls = self.args.get("init_cls", 10)
                dataset_loader = lambda: self.data_manager.get_dataset(
                    np.arange(0, init_cls), source="train", mode="test"
                )
                self.cluster_analyzer.save_task1_samples(
                    dataset_loader=dataset_loader,
                    batch_size=self.batch_size,
                    num_workers=num_workers
                )
            else:
                # 后续任务：计算Procrustes距离
                self.cluster_analyzer.compute_procrustes_distances(self._cur_task)
        
        # 6. t-SNE 可视化（如果启用）
        if getattr(self, "use_hc_soinn", False) and self.args.get("visualize_tsne", False):
            try:
                init_cls = self.args.get("init_cls", 10)
                self._visualize_hc_soinn_tsne(target_classes=list(range(min(10, init_cls))))
            except Exception as e:
                logging.error(f"Error during t-SNE visualization in coda_prompt.after_task: {e}", exc_info=True)
        
        # 7. 更新已知类别数
        self._known_classes = self._total_classes
        
        # 8. 存储占用分析（如果启用）
        if self.enable_storage_analysis:
            image_shape = (224, 224, 3)  # 默认图像尺寸，可根据数据集调整
            # 根据数据集调整图像尺寸
            dataset_name = self.args.get("dataset", "").lower()
            if "cifar" in dataset_name:
                image_shape = (32, 32, 3)
            elif "imagenet" in dataset_name or "inr" in dataset_name:
                image_shape = (224, 224, 3)
            
            storage_results = self.storage_analyzer.analyze_all(
                network=self._network,
                hc_soinn=getattr(self, 'hc_soinn', None) if getattr(self, 'use_hc_soinn', False) else None,
                star_aligner=getattr(self, 'star', None) if getattr(self, 'use_feature_alignment', False) else None,
                image_shape=image_shape
            )
            self.storage_analyzer.print_report(storage_results, task_id=self._cur_task)

        # 8. 保存checkpoint（每个任务训练完后）
        if self.args.get("save_checkpoint", False):
            self.save_checkpoint()
        
        # 7. SOINN t-SNE 可视化（如果启用）
        if getattr(self, "use_soinn", False) and hasattr(self, "soinn"):
            self._visualize_soinn_tsne()

    def incremental_train(self, data_manager):
        self._cur_task += 1
        
        # 如果启用了自动加载所有checkpoint模式，跳过训练，只加载checkpoint
        if self.args.get("load_all_checkpoints", False):
            self._load_checkpoint_for_task()
            # 加载checkpoint后，checkpoint中已经保存了正确的状态（_cur_task, _known_classes, _total_classes）
            # 但需要确保_total_classes与当前任务划分一致（因为可能因为任务划分而不同）
            # 重新计算以确保一致性
            expected_total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
            if self._total_classes != expected_total_classes:
                logging.warning(
                    f"Total classes mismatch: checkpoint has {self._total_classes}, "
                    f"expected {expected_total_classes}. Using expected value."
                )
                self._total_classes = expected_total_classes
            logging.info("Learning on {}-{} (checkpoint loaded, skipping training)".format(self._known_classes, self._total_classes))
            
            # 准备测试数据
            if self.test_only_first_task_classes:
                test_classes = np.arange(0, self.init_cls)
            else:
                test_classes = np.arange(0, self._total_classes)
            test_dataset = data_manager.get_dataset(test_classes, source="test", mode="test")
            self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
            
            # 更新prompt的任务计数（如果需要）
            if self._cur_task > 0:
                try:
                    if self._network.module.prompt is not None:
                        self._network.module.prompt.process_task_count()
                except:
                    if self._network.prompt is not None:
                        self._network.prompt.process_task_count()
            
            # 准备测试数据（与正常训练流程一致）
            train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train")
            self.train_dataset = train_dataset
            self.data_manager = data_manager
            if self.test_only_first_task_classes:
                test_classes = np.arange(0, self.init_cls)
            else:
                test_classes = np.arange(0, self._total_classes)
            test_dataset = data_manager.get_dataset(test_classes, source="test", mode="test")
            self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
            
            # 统一构建所有分类器（与正常训练流程一致）
            self._build_classifiers()
            
            return  # 跳过训练部分
        
        if self._cur_task > 0:
            try:
                if self._network.module.prompt is not None:
                    self._network.module.prompt.process_task_count()
            except:
                if self._network.prompt is not None:
                    self._network.prompt.process_task_count()

        if not self.args.get("load_all_checkpoints", False):
            self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
            # self._network.update_fc(self._total_classes)
            logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train")
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
        if self.test_only_first_task_classes:
            test_classes = np.arange(0, self.init_cls)
            logging.info(f"实验模式：测试数据集只包含类别 {test_classes}")
        else:
            test_classes = np.arange(0, self._total_classes)
        test_dataset = data_manager.get_dataset(test_classes, source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        
        # 训练结束后，统一构建所有分类器（NCM、KNN等）
        # 无论是正常训练还是加载checkpoint，都在这里统一构建
        self._build_classifiers()

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)

        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)

        self.data_weighting()
        self._init_train(train_loader, test_loader, optimizer, scheduler)

        # 注意：分类器构建已移至 _build_classifiers() 方法，在训练结束后统一构建

    def _build_classifiers(self):
        """
        统一构建所有分类器（在训练结束后、测试开始前调用）
        包括：NCM分类器、KNN bank（普通模式或upperbound模式）
        无论是正常训练还是加载checkpoint，都在这里统一构建
        """
        
        # 确保模型在正确的设备上并处于eval模式
        self._network.to(self._device)
        self._network.eval()
        
        # ========== Step 0: STAR 特征漂移对齐（在评估前对齐旧类别）==========
        # 关键修复：在评估之前对齐旧类别，确保评估时使用的是对齐后的节点
        # 这解决了性能断崖式下跌的问题（Task 2 评估时使用的是未对齐的节点）
        if self.star is not None and self._cur_task > 0:
            # 准备当前任务的类别集合（用于排除）
            current_task_classes = set(range(self._known_classes, self._total_classes))
            logging.info(
                f"[STAR] Pre-evaluation alignment: Aligning old classes before evaluation "
                f"(task {self._cur_task}, current task classes: {current_task_classes})"
            )
            self.star.align_old_classes(
                cur_task=self._cur_task,
                current_task_classes=current_task_classes
            )
        
        # 1. 构建NCM分类器：计算所有已见过的任务的类均值（如果启用）
        if getattr(self, "use_ncm", True):
            self._build_ncm_classifier()
        
        # 2. 构建KNN bank
        if getattr(self, "use_knn", False):
            if getattr(self, "knn_upperbound", False):
                # Upperbound模式：使用所有已见过的任务重新提取特征
                self._build_knn_bank_upperbound()
            else:
                # 普通模式：使用所有已见过的任务重新提取特征（与upperbound一致，但会在评估时使用）
                self._build_knn_bank()
        
        # 3. 构建 HC-SOINN / SOINN bank（互斥）
        if getattr(self, "use_hc_soinn", False):
            self._build_hc_soinn_bank()
        elif getattr(self, "use_soinn", False):
            self._build_soinn_bank()
        
    
    def _build_ncm_classifier(self):
        """
        构建NCM分类器：累积存储机制
        - 如果之前有类均值（正常训练）：保留之前任务的类均值，只计算当前任务新类别的类均值
        - 如果没有类均值（首次任务或加载checkpoint后）：计算所有已见过的任务的类均值
        - 这样符合增量学习的累积存储要求
        - 实验模式：只构建第一个任务的类别（0-init_cls-1）
        """
        # 实验模式：只构建第一个任务的类别
        if self.test_only_first_task_classes:
            ncm_classes = self.init_cls
        else:
            ncm_classes = self._total_classes
        
        # 判断是否需要重建所有类均值（首次任务或加载checkpoint后）
        need_rebuild_all = (self._class_means is None) or (self._class_means.shape[0] == 0)
        
        if need_rebuild_all:
            # 首次任务或加载checkpoint后：需要计算所有已见过的任务的类均值
            if self.test_only_first_task_classes:
                logging.info(f"Building NCM classifier (实验模式): computing first task classes (0-{ncm_classes-1}) [first task or after checkpoint load]")
            else:
                logging.info(f"Building NCM classifier: computing all seen classes (0-{ncm_classes-1}) [first task or after checkpoint load]")
            all_train_dataset = self.data_manager.get_dataset(
                np.arange(0, ncm_classes), source="train", mode="test"
            )
            all_train_loader = DataLoader(
                all_train_dataset, 
                batch_size=self.batch_size, 
                shuffle=False, 
                drop_last=False, 
                num_workers=num_workers
            )
            
            # 初始化类均值数组
            self._class_means = np.zeros((ncm_classes, self.feature_dim))
            
            # 提取所有训练样本的特征
            embedding_list = []
            label_list = []
            
            with torch.no_grad():
                for _, (_, inputs, targets) in enumerate(all_train_loader):
                    inputs = inputs.to(self._device)
                    if isinstance(self._network, nn.DataParallel):
                        embeddings = self._network.module.extract_vector(inputs)
                    else:
                        embeddings = self._network.extract_vector(inputs)
                    embedding_list.append(embeddings.cpu())
                    label_list.append(targets.cpu())
            
            if len(embedding_list) > 0:
                embedding_list = torch.cat(embedding_list, dim=0)
                label_list = torch.cat(label_list, dim=0)
                
                # 计算所有已见过的类别的均值
                class_list = np.unique(label_list.numpy())
                for class_index in class_list:
                    data_index = (label_list == class_index).nonzero().squeeze(-1)
                    embedding = embedding_list[data_index]
                    proto = embedding.mean(0)
                    self._class_means[int(class_index), :] = proto.cpu().numpy()
                
                logging.info(f"NCM classifier built: computed class means for {len(class_list)} classes (total: {ncm_classes} classes)")
            else:
                logging.warning("No training data available for NCM classifier")
        else:
            # 正常训练：累积存储机制 - 保留之前的类均值，只计算当前任务新类别的类均值
            if self.test_only_first_task_classes:
                # 实验模式：不添加新类别，只保留第一个任务的类别
                logging.info(f"Building NCM classifier (实验模式): preserving first task class means (0-{ncm_classes-1}), skipping new classes")
                # 不需要更新类均值，直接返回
                return
            else:
                logging.info(f"Building NCM classifier: preserving previous class means, computing new classes ({self._known_classes}-{self._total_classes-1})")
            
            # 扩展类均值数组以容纳新类别，保留之前的类均值
            if self._class_means.shape[0] < self._total_classes:
                new_class_means = np.zeros((self._total_classes, self.feature_dim))
                new_class_means[:self._class_means.shape[0]] = self._class_means
                self._class_means = new_class_means
                logging.info(f"Extended NCM classifier: preserved {self._class_means.shape[0] - (self._total_classes - self._known_classes)} previous class means")
            
            # 只获取当前任务的训练数据（累积存储：过往数据已存储，只需添加当前任务）
            current_task_dataset = self.data_manager.get_dataset(
                np.arange(self._known_classes, self._total_classes), source="train", mode="test"
            )
            current_task_loader = DataLoader(
                current_task_dataset, 
                batch_size=self.batch_size, 
                shuffle=False, 
                drop_last=False, 
                num_workers=num_workers
            )
            
            # 提取当前任务训练样本的特征
            embedding_list = []
            label_list = []
            
            with torch.no_grad():
                for _, (_, inputs, targets) in enumerate(current_task_loader):
                    inputs = inputs.to(self._device)
                    if isinstance(self._network, nn.DataParallel):
                        embeddings = self._network.module.extract_vector(inputs)
                    else:
                        embeddings = self._network.extract_vector(inputs)
                    embedding_list.append(embeddings.cpu())
                    label_list.append(targets.cpu())
            
            if len(embedding_list) > 0:
                embedding_list = torch.cat(embedding_list, dim=0)
                label_list = torch.cat(label_list, dim=0)
                
                # 只计算当前任务新类别的均值（累积存储：过往类均值已保留）
                class_list = np.unique(label_list.numpy())
                for class_index in class_list:
                    data_index = (label_list == class_index).nonzero().squeeze(-1)
                    embedding = embedding_list[data_index]
                    proto = embedding.mean(0)
                    self._class_means[int(class_index), :] = proto.cpu().numpy()
                
                logging.info(f"NCM classifier updated: computed class means for {len(class_list)} new classes (total: {self._total_classes} classes)")
            else:
                logging.warning("No training data available for NCM classifier update")
        
        # 更新所有类别的NCM FC层权重
        if self.test_only_first_task_classes:
            ncm_classes = self.init_cls
        else:
            ncm_classes = self._total_classes
        
        if isinstance(self._network, nn.DataParallel):
            for class_index in range(ncm_classes):
                if np.any(self._class_means[class_index] != 0):  # 只更新有数据的类别
                    self._network.module.ncm_fc.weight.data[int(class_index), :] = torch.from_numpy(self._class_means[int(class_index), :]).float().to(self._device)
        else:
            for class_index in range(ncm_classes):
                if np.any(self._class_means[class_index] != 0):  # 只更新有数据的类别
                    self._network.ncm_fc.weight.data[int(class_index), :] = torch.from_numpy(self._class_means[int(class_index), :]).float().to(self._device)
    
    def _build_knn_bank(self):
        """
        构建KNN bank（普通模式）：累积存储机制
        - 如果KNN bank为空（首次任务或加载checkpoint后）：需要重建所有已见过的任务的features
        - 如果KNN bank不为空（正常训练）：保留之前任务的features，只添加当前任务新类别的features
        - 这样符合增量学习的累积存储要求
        """
        # 判断是否需要重建所有features（首次任务或加载checkpoint后）
        # 检查KNN bank是否为空（通过检查是否有任何类别的features）
        knn_bank_empty = (not hasattr(self.knn, 'class_to_features')) or (len(self.knn.class_to_features) == 0)
        
        if knn_bank_empty:
            # 首次任务或加载checkpoint后：需要计算所有已见过的任务的features
            all_train_dataset = self.data_manager.get_dataset(
                np.arange(0, self._total_classes), source="train", mode="test"
            )
            all_train_loader = DataLoader(
                all_train_dataset, 
                batch_size=self.batch_size, 
                shuffle=False, 
                drop_last=False, 
                num_workers=num_workers
            )
            
            # 定义特征提取函数
            def feature_fn(x):
                if isinstance(self._network, nn.DataParallel):
                    feats = self._network.module(x, pen=True, train=False)
                else:
                    feats = self._network(x, pen=True, train=False)
                
                if isinstance(feats, tuple):
                    feats = feats[0]
                
                if not isinstance(feats, torch.Tensor):
                    raise TypeError(f"Expected tensor, got {type(feats)}")
                
                if len(feats.shape) == 1:
                    feats = feats.reshape(1, -1)
                elif len(feats.shape) != 2:
                    raise ValueError(f"Expected 2D features [B, D], got shape {feats.shape}")
                
                return feats
            
            # 清空并重建所有features
            self.knn.clear()
            with torch.no_grad():
                self.knn.add_from_loader(all_train_loader, feature_fn, self._device)
            
            logging.info(f"KNN bank built: computed features for {self._total_classes} classes")
        else:
            # 正常训练：累积存储机制 - 保留之前的features，只添加当前任务新类别的features
            
            # 注意：不清空KNN bank，保留之前的features（累积存储）
            
            # 定义特征提取函数
            def feature_fn(x):
                if isinstance(self._network, nn.DataParallel):
                    feats = self._network.module(x, pen=True, train=False)
                else:
                    feats = self._network(x, pen=True, train=False)
                
                if isinstance(feats, tuple):
                    feats = feats[0]
                
                if not isinstance(feats, torch.Tensor):
                    raise TypeError(f"Expected tensor, got {type(feats)}")
                
                if len(feats.shape) == 1:
                    feats = feats.reshape(1, -1)
                elif len(feats.shape) != 2:
                    raise ValueError(f"Expected 2D features [B, D], got shape {feats.shape}")
                
                return feats
            
            # 只获取当前任务的训练数据（累积存储：过往features已存储在KNN bank中）
            current_task_dataset = self.data_manager.get_dataset(
                np.arange(self._known_classes, self._total_classes), source="train", mode="test"
            )
            current_task_loader = DataLoader(
                current_task_dataset, 
                batch_size=self.batch_size, 
                shuffle=False, 
                drop_last=False, 
                num_workers=num_workers
            )
            
            # 只添加当前任务的特征到KNN bank（累积存储：过往features已保留）
            with torch.no_grad():
                self.knn.add_from_loader(current_task_loader, feature_fn, self._device)
            
            logging.info(f"KNN bank updated: added features for classes {self._known_classes}-{self._total_classes-1} (previous features preserved)")
    
    def _build_knn_bank_upperbound(self):
        """
        构建KNN bank（Upperbound模式）：不需要存储机制
        - 每次评估时都会重新构建（在eval_task中），使用最新的模型特征
        - 不使用累积存储，每次都重新提取所有数据（当前+过往）的特征
        - 训练结束后不需要构建，因为评估时会重建
        """
        # Upperbound模式：不需要在训练结束后构建
        # 因为每次评估时都会重新构建，使用最新的模型特征
        # 这样可以避免特征漂移，测试KNN方法的上限
        logging.info("KNN upperbound mode: skipping build at training end (will be rebuilt at eval time with latest model)")
    
    def _build_knn_bank_upperbound_at_eval(self):
        """
        在评估时构建KNN bank（Upperbound模式）：
        不使用累积存储，每次都重新提取所有数据（当前+过往）的特征
        使用最新的模型特征，避免特征漂移
        """
        
        # 清空KNN bank（不使用累积存储）
        self.knn.clear()
        
        # 确保模型处于eval模式
        self._network.eval()
        
        # 定义特征提取函数
        def feature_fn(x):
            if isinstance(self._network, nn.DataParallel):
                feats = self._network.module(x, pen=True, train=False)
            else:
                feats = self._network(x, pen=True, train=False)
            
            if isinstance(feats, tuple):
                feats = feats[0]
            
            if not isinstance(feats, torch.Tensor):
                raise TypeError(f"Expected tensor, got {type(feats)}")
            
            if len(feats.shape) == 1:
                feats = feats.reshape(1, -1)
            elif len(feats.shape) != 2:
                raise ValueError(f"Expected 2D features [B, D], got shape {feats.shape}")
            
            return feats
        
        # 获取所有已见过的任务的训练数据（不使用累积存储，每次都重新提取）
        all_train_dataset = self.data_manager.get_dataset(
            np.arange(0, self._total_classes), source="train", mode="test"
        )
        all_train_loader = DataLoader(
            all_train_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            drop_last=False, 
            num_workers=num_workers
        )
        
        # 使用最新的模型重新提取所有训练样本的特征并添加到KNN bank
        with torch.no_grad():
            self.knn.add_from_loader(all_train_loader, feature_fn, self._device)
        
        logging.info(f"KNN bank rebuilt in upperbound mode with {self._total_classes} classes using latest model features")

    def _get_soinn_feature_fn(self):
        """
        获取SOINN特征提取函数（统一特征提取逻辑）
        """
        def feature_fn(x):
            if isinstance(self._network, nn.DataParallel):
                feats = self._network.module(x, pen=True, train=False)
            else:
                feats = self._network(x, pen=True, train=False)
            
            if isinstance(feats, tuple):
                feats = feats[0]
            
            if not isinstance(feats, torch.Tensor):
                raise TypeError(f"Expected tensor, got {type(feats)}")
            
            if len(feats.shape) == 1:
                feats = feats.reshape(1, -1)
            elif len(feats.shape) != 2:
                raise ValueError(f"Expected 2D features [B, D], got shape {feats.shape}")
            
            return feats
        
        return feature_fn

    def _build_soinn_bank(self):
        """
        构建SOINN bank：累积存储机制
        - 如果SOINN bank为空（首次任务或加载checkpoint后）：需要重建所有已见过的任务的features
        - 如果SOINN bank不为空（正常训练）：保留之前任务的features，只添加当前任务新类别的features
        - 这样符合增量学习的累积存储要求
        """
        # 判断是否需要重建所有features（首次任务或加载checkpoint后）
        # 检查SOINN bank是否为空
        soinn_bank_empty = (not hasattr(self.soinn, 'prototypes')) or (self.soinn.prototypes is None) or (len(self.soinn.prototypes) == 0)
        
        if soinn_bank_empty:
            # 首次任务或加载checkpoint后：需要计算所有已见过的任务的features
            all_train_dataset = self.data_manager.get_dataset(
                np.arange(0, self._total_classes), source="train", mode="test"
            )
            all_train_loader = DataLoader(
                all_train_dataset, 
                batch_size=self.batch_size, 
                shuffle=False, 
                drop_last=False, 
                num_workers=num_workers
            )
            
            # 使用统一的特征提取函数
            feature_fn = self._get_soinn_feature_fn()
            
            # 清空并重建所有features
            self.soinn.clear()
            with torch.no_grad():
                self.soinn.add_from_loader(all_train_loader, feature_fn, self._device)
            
            # 打印每个类的原型数量
            proto_info = self.soinn.prototypes_per_class()
            logging.info(f"SOINN bank built: computed prototypes for {self._total_classes} classes")
            logging.info(f"SOINN prototypes per class: {proto_info}")
        else:
            # 正常训练：累积存储机制 - 保留之前的features，只添加当前任务新类别的features
            
            # 注意：不清空SOINN bank，保留之前的features（累积存储）
            
            # 使用统一的特征提取函数
            feature_fn = self._get_soinn_feature_fn()
            
            # 只获取当前任务的训练数据（累积存储：过往features已存储在SOINN bank中）
            current_task_dataset = self.data_manager.get_dataset(
                np.arange(self._known_classes, self._total_classes), source="train", mode="test"
            )
            current_task_loader = DataLoader(
                current_task_dataset, 
                batch_size=self.batch_size, 
                shuffle=False, 
                drop_last=False, 
                num_workers=num_workers
            )
            
            # 只添加当前任务的特征到SOINN bank（累积存储：过往features已保留）
            with torch.no_grad():
                self.soinn.add_from_loader(current_task_loader, feature_fn, self._device)
            
            # 打印每个类的原型数量
            proto_info = self.soinn.prototypes_per_class()
            logging.info(f"SOINN bank updated: added prototypes for classes {self._known_classes}-{self._total_classes-1} (previous prototypes preserved)")
            logging.info(f"SOINN prototypes per class: {proto_info}")

    def _build_hc_soinn_bank(self):
        """
        构建 HC-SOINN bank：类增量学习场景下的累积存储
        - 每个任务只使用当前任务的新类别训练数据（符合类增量学习设定）
        - 旧类别的信息通过已保存的簇中心保留（在 compress 时合并）
        """
        hc_bank_empty = (not hasattr(self, "hc_soinn")) or (len(getattr(self.hc_soinn, "class_clusters", {})) == 0)
        if hc_bank_empty:
            logging.info(f"HC-SOINN bank is empty!!!")
        feature_fn = self._get_soinn_feature_fn()

        def add_from_loader(loader):
            feats, lbs = [], []
            with torch.no_grad():
                for _, inputs, targets in loader:
                    inputs = inputs.to(self._device)
                    batch_feats = feature_fn(inputs)
                    if isinstance(batch_feats, torch.Tensor):
                        batch_feats = batch_feats.detach().cpu().numpy()
                    lbs.append(targets.numpy())
                    feats.append(batch_feats)
            if len(feats) == 0:
                return
            feats_np = np.concatenate(feats, axis=0)
            lbs_np = np.concatenate(lbs, axis=0)
            self.hc_soinn.add_features(feats_np, lbs_np)

        # 类增量学习：每个任务只使用当前任务的新类别训练数据
        current_task_dataset = self.data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="test"
        )
        current_task_loader = DataLoader(
            current_task_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
        )
        add_from_loader(current_task_loader)

        # 每个任务结束后压缩一次
        try:
            self.hc_soinn.compress()
        except Exception as e:
            logging.error(f"HC-SOINN compress error: {e}", exc_info=True)

    def data_weighting(self):
        self.dw_k = torch.tensor(np.ones(self._total_classes + 1, dtype=np.float32))
        self.dw_k = self.dw_k.to(self._device)

    def get_optimizer(self):
        if len(self._multiple_gpus) > 1:
            params = list(self._network.module.prompt.parameters()) + list(self._network.module.fc.parameters())
        else:
            params = list(self._network.prompt.parameters()) + list(self._network.fc.parameters())
        
        # 验证KAC分类器的参数是否被正确包含
        if self.use_kac:
            fc_param_count = sum(p.numel() for p in self._network.fc.parameters() if p.requires_grad)
            logging.info(f"KAC classifier parameters included in optimizer: {fc_param_count:,}")
        
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(params, momentum=0.9, lr=self.init_lr,weight_decay=self.weight_decay)
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(params, lr=self.init_lr, weight_decay=self.weight_decay)
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(params, lr=self.init_lr, weight_decay=self.weight_decay)

        return optimizer

    def get_scheduler(self, optimizer):
        if self.args["scheduler"] == 'cosine':
            scheduler = CosineSchedule(optimizer, K=self.args["tuned_epoch"])
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None

        return scheduler

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        for _, epoch in enumerate(prog_bar):
            self._network.train()

            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
            
                # logits
                logits, prompt_loss = self._network(inputs, train=True)
                logits = logits[:, :self._total_classes]

                logits[:, :self._known_classes] = float('-inf')
                dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
                loss_supervised = (F.cross_entropy(logits, targets.long()) * dw_cls).mean()

                # ce loss
                loss = loss_supervised + prompt_loss.sum()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler:
                scheduler.step()
            
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if (epoch + 1) % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['tuned_epoch'],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['tuned_epoch'],
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)

        logging.info(info)

    def _eval_fc(self, loader):
        """使用原始FC层进行评估（支持KAC分类器）"""
        self._network.eval()
        y_pred, y_true = [], []
        # 实验模式：只使用第一个任务的类别
        if self.test_only_first_task_classes:
            eval_classes = self.init_cls
        else:
            eval_classes = self._total_classes
        
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)[:, :eval_classes]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)
    
    def _eval_kac(self, loader):
        """使用KAC分类器进行评估（与_eval_fc相同，但用于区分）"""
        # KAC分类器已经集成在self._network.fc中，所以可以直接使用_eval_fc
        return self._eval_fc(loader)
    
    def _eval_knn(self, loader):
        """使用KNN分类器进行评估"""
        self._network.eval()
        y_pred, y_true = [], []
        with torch.no_grad():
            for _, (_, inputs, targets) in enumerate(loader):
                inputs = inputs.to(self._device)
                # 提取特征（与训练时添加到KNN bank的特征提取方式保持一致）
                if isinstance(self._network, nn.DataParallel):
                    feats = self._network.module(inputs, pen=True, train=False)
                else:
                    feats = self._network(inputs, pen=True, train=False)
                
                # 确保特征形状正确 [B, D]
                if isinstance(feats, tuple):
                    feats = feats[0]
                
                if not isinstance(feats, torch.Tensor):
                    raise TypeError(f"Expected tensor, got {type(feats)}")
                
                feats = feats.detach().cpu()
                
                if len(feats.shape) == 1:
                    feats = feats.reshape(1, -1)
                elif len(feats.shape) != 2:
                    raise ValueError(f"Expected 2D features [B, D], got shape {feats.shape}")
                
                # 转换为 numpy 并预测（启用 GPU 加速，并跟踪使用频率以支持节点剪枝）
                feats_np = feats.numpy()
                topk_pred = self.knn.predict_topk(
                    feats_np, self.topk, self._total_classes, device=self._device, track_usage=True
                )
                y_pred.append(topk_pred)
                y_true.append(targets.cpu().numpy())
        
        if len(y_pred) == 0:
            logging.warning("No predictions generated from KNN evaluation")
            return np.array([]), np.array([])
        
        return np.concatenate(y_pred), np.concatenate(y_true)

    def _eval_ncm_fc(self, loader):
        """使用NCM FC层进行评估（CosineLinear会自动归一化）"""
        self._network.eval()
        y_pred, y_true = [], []
        # 实验模式：只使用第一个任务的类别
        if self.test_only_first_task_classes:
            ncm_classes = self.init_cls
        else:
            ncm_classes = self._total_classes
        
        with torch.no_grad():
            for _, (_, inputs, targets) in enumerate(loader):
                inputs = inputs.to(self._device)
                # 提取特征（不需要归一化，CosineLinear会自动归一化）
                if isinstance(self._network, nn.DataParallel):
                    features = self._network.module.extract_vector(inputs)
                    # CosineLinear返回字典格式 {'logits': ...}
                    ncm_output = self._network.module.ncm_fc(features)
                else:
                    features = self._network.extract_vector(inputs)
                    # CosineLinear返回字典格式 {'logits': ...}
                    ncm_output = self._network.ncm_fc(features)
                # 提取logits
                ncm_logits = ncm_output['logits']
                # 只保留已学习类别（实验模式下只保留第一个任务的类别）
                ncm_logits = ncm_logits[:, :ncm_classes]
                # Top-k预测
                predicts = torch.topk(
                    ncm_logits, k=self.topk, dim=1, largest=True, sorted=True
                )[1]
                y_pred.append(predicts.cpu().numpy())
                y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)

    def _eval_soinn(self, loader):
        """使用SOINN分类器进行评估"""
        self._network.eval()
        y_pred, y_true = [], []
        with torch.no_grad():
            for _, (_, inputs, targets) in enumerate(loader):
                inputs = inputs.to(self._device)
                # 提取特征（与训练时添加到SOINN bank的特征提取方式保持一致）
                feature_fn = self._get_soinn_feature_fn()
                feats = feature_fn(inputs)
                
                feats = feats.detach().cpu()
                
                if len(feats.shape) == 1:
                    feats = feats.reshape(1, -1)
                elif len(feats.shape) != 2:
                    raise ValueError(f"Expected 2D features [B, D], got shape {feats.shape}")
                
                # 转换为 numpy 并预测（启用 GPU 加速）
                feats_np = feats.numpy()
                topk_pred = self.soinn.predict_topk(feats_np, self.topk, self._total_classes, device=self._device)
                y_pred.append(topk_pred)
                y_true.append(targets.cpu().numpy())
        
        if len(y_pred) == 0:
            logging.warning("No predictions generated from SOINN evaluation")
            return np.array([]), np.array([])
        
        return np.concatenate(y_pred), np.concatenate(y_true)

    def _eval_hc_soinn(self, loader):
        """使用 HC-SOINN 分类器进行评估"""
        self._network.eval()
        y_pred, y_true = [], []
        feature_fn = self._get_soinn_feature_fn()
        with torch.no_grad():
            for _, (_, inputs, targets) in enumerate(loader):
                inputs = inputs.to(self._device)
                feats = feature_fn(inputs)
                feats = feats.detach().cpu()
                if len(feats.shape) == 1:
                    feats = feats.reshape(1, -1)
                elif len(feats.shape) != 2:
                    raise ValueError(f"Expected 2D features [B, D], got shape {feats.shape}")
                feats_np = feats.numpy()
                topk_pred = self.hc_soinn.predict_topk(
                    feats_np, self.topk, self._total_classes, device=self._device
                )
                y_pred.append(topk_pred)
                y_true.append(targets.cpu().numpy())

        if len(y_pred) == 0:
            logging.warning("No predictions generated from HC-SOINN evaluation")
            return np.array([]), np.array([])

        return np.concatenate(y_pred), np.concatenate(y_true)

    def eval_task(self):
        """
        评估任务：分别评估FC、KNN（如果启用）、NCM（如果启用）三种分类器
        返回所有分类器的精度结果
        """
        # 如果启用了knn_upperbound，在评估前重新构建KNN bank（使用最新的模型特征）
        if getattr(self, "use_knn", False) and getattr(self, "knn_upperbound", False):
            self._build_knn_bank_upperbound_at_eval()
        
        results = {}
        
        # 1. 使用原始FC分类器评估（如果使用KAC，则评估KAC分类器）
        if self.use_kac:
            y_pred_kac, y_true_kac = self._eval_kac(self.test_loader)
            results["kac"] = self._evaluate(y_pred_kac, y_true_kac)
            # 同时保留fc结果（KAC分类器就是fc层）
            results["fc"] = results["kac"]
        else:
            y_pred_fc, y_true_fc = self._eval_fc(self.test_loader)
            results["fc"] = self._evaluate(y_pred_fc, y_true_fc)
        
        # 2. 使用KNN分类器评估（如果启用）
        if getattr(self, "use_knn", False):
            y_pred_knn, y_true_knn = self._eval_knn(self.test_loader)
            results["knn"] = self._evaluate(y_pred_knn, y_true_knn)
        
        # 3. 使用 HC-SOINN / SOINN 分类器评估（互斥）
        if getattr(self, "use_hc_soinn", False):
            y_pred_hc, y_true_hc = self._eval_hc_soinn(self.test_loader)
            results["hc_soinn"] = self._evaluate(y_pred_hc, y_true_hc)
        elif getattr(self, "use_soinn", False):
            y_pred_soinn, y_true_soinn = self._eval_soinn(self.test_loader)
            results["soinn"] = self._evaluate(y_pred_soinn, y_true_soinn)
        
        # 4. 使用NCM分类器评估（如果启用且已计算类均值）
        if getattr(self, "use_ncm", True) and hasattr(self, "_class_means") and self._class_means is not None:
            y_pred_ncm, y_true_ncm = self._eval_ncm_fc(self.test_loader)
            results["ncm"] = self._evaluate(y_pred_ncm, y_true_ncm)
        
        return results

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        # 实验模式：只使用第一个任务的类别
        if self.test_only_first_task_classes:
            eval_classes = self.init_cls
        else:
            eval_classes = self._total_classes
        
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)[:, :eval_classes]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def save_checkpoint(self):
        """
        保存checkpoint，只保存模型参数本身（backbone, fc, prompt），
        不包括ncm_fc和knn相关内容
        """
        checkpoint_dir = self.args.get("checkpoint_dir", "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 构建checkpoint文件名
        checkpoint_name = self.args.get("checkpoint_name", 
            f"{self.args['model_name']}_{self.args['dataset']}_{self.args['init_cls']}_{self.args['increment']}")
        checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_name}_task{self._cur_task}.pkl")
        
        # 获取模型（如果是DataParallel，需要获取module）
        model = self._network
        if isinstance(model, nn.DataParallel):
            model = model.module
        
        # 只保存需要的部分：backbone, fc, prompt
        save_dict = {
            "task": self._cur_task,
            "known_classes": self._known_classes,
            "total_classes": self._total_classes,
            "backbone_state_dict": model.backbone.state_dict(),
            "fc_state_dict": model.fc.state_dict(),
            "prompt_state_dict": model.prompt.state_dict() if model.prompt is not None else None,
        }
        
        torch.save(save_dict, checkpoint_path)
        logging.info(f"Checkpoint saved to {checkpoint_path}")

    def _visualize_soinn_tsne(self):
        """
        SOINN t-SNE 可视化：对第一个 task 的 5 个类别进行可视化
        每个 task 结束后都生成一张图片，显示样本点和 SOINN 节点
        """
        # 只对第一个 task 的类别进行可视化
        init_cls = self.args.get("init_cls", 10)
        if init_cls < 5:
            logging.warning(f"Initial classes ({init_cls}) < 5, skipping SOINN visualization")
            return
        
        # 选择前 5 个类别
        target_classes = list(range(5))
        
        # 检查这些类别是否在 SOINN 中
        if not hasattr(self.soinn, '_class_models'):
            logging.warning("SOINN models not found, skipping visualization")
            return
        
        available_classes = [cls for cls in target_classes if cls in self.soinn._class_models]
        if len(available_classes) == 0:
            logging.warning("No target classes found in SOINN, skipping visualization")
            return
        
        # 获取第一个 task 的训练数据（用于提取样本特征）
        if not hasattr(self, 'data_manager'):
            logging.warning("DataManager not found, skipping visualization")
            return
        
        try:
            # 获取第一个 task 的 5 个类别的训练数据
            train_dataset = self.data_manager.get_dataset(
                np.arange(0, min(5, init_cls)), 
                source="train", 
                mode="test"  # 使用 test 模式关闭数据增强
            )
            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.batch_size, 
                shuffle=False, 
                drop_last=False, 
                num_workers=num_workers
            )
            
            # 提取样本特征
            feature_fn = self._get_soinn_feature_fn()
            sample_features = []
            sample_labels = []
            
            self._network.eval()
            with torch.no_grad():
                for _, inputs, targets in train_loader:
                    inputs = inputs.to(self._device)
                    feats = feature_fn(inputs)
                    feats_np = feats.detach().cpu().numpy()
                    sample_features.append(feats_np)
                    sample_labels.append(targets.numpy())
            
            sample_features = np.concatenate(sample_features, axis=0)
            sample_labels = np.concatenate(sample_labels, axis=0)
            
            # 只保留前 5 个类别的样本
            mask = np.isin(sample_labels, target_classes)
            sample_features = sample_features[mask]
            sample_labels = sample_labels[mask]
            
            if len(sample_features) == 0:
                logging.warning("No samples found for target classes, skipping visualization")
                return
            
            # 重要：归一化样本特征（与 SOINN 节点保持一致）
            # SOINN 节点在训练时就已经归一化了，所以样本特征也需要归一化
            sample_features = sample_features / (np.linalg.norm(sample_features, axis=1, keepdims=True) + 1e-8)
            
            # 收集 SOINN 节点和边信息
            soinn_nodes = []
            soinn_labels = []
            soinn_edges = []  # [(node_idx1, node_idx2), ...] 全局索引
            
            node_offset = 0
            for cls in available_classes:
                model = self.soinn._class_models[cls]
                if len(model.nodes) == 0:
                    continue
                
                # 添加该类的所有节点
                for node in model.nodes:
                    soinn_nodes.append(node)
                    soinn_labels.append(cls)
                
                # 添加该类的边（使用全局索引）
                for node_i, neighbors in model.edges.items():
                    for node_j in neighbors.keys():
                        if node_j > node_i:  # 避免重复边
                            global_i = node_offset + node_i
                            global_j = node_offset + node_j
                            soinn_edges.append((global_i, global_j))
                
                node_offset += len(model.nodes)
            
            if len(soinn_nodes) == 0:
                logging.warning("No SOINN nodes found, skipping visualization")
                return
            
            soinn_nodes = np.array(soinn_nodes)
            
            # 先验证高维空间中的实际距离（余弦距离）
            # 注意：此时 sample_features 和 soinn_nodes 都已经归一化了
            logging.info("Verifying distances between samples and SOINN nodes in high-dimensional space...")
            for cls in available_classes:
                cls_sample_mask = sample_labels == cls
                if not np.any(cls_sample_mask):
                    continue
                cls_samples = sample_features[cls_sample_mask]
                
                # 找到该类别的 SOINN 节点
                cls_soinn_mask = np.array(soinn_labels) == cls
                if not np.any(cls_soinn_mask):
                    continue
                cls_soinn_nodes = soinn_nodes[cls_soinn_mask]
                
                # 计算每个样本到最近 SOINN 节点的余弦距离（都已归一化）
                cosine_sims = np.dot(cls_samples, cls_soinn_nodes.T)  # [N_samples, N_nodes]
                cosine_dists = 1.0 - cosine_sims  # 余弦距离
                min_dists = cosine_dists.min(axis=1)  # 每个样本到最近节点的距离
                
                avg_min_dist = min_dists.mean()
                logging.info(f"Class {cls}: Average cosine distance from samples to nearest SOINN node: {avg_min_dist:.4f} "
                           f"(min={min_dists.min():.4f}, max={min_dists.max():.4f})")
            
            # 合并样本和节点特征（都已归一化）
            all_features = np.vstack([sample_features, soinn_nodes])
            all_labels = np.concatenate([sample_labels, np.array(soinn_labels)])
            is_sample = np.concatenate([
                np.ones(len(sample_features), dtype=bool),  # True 表示样本
                np.zeros(len(soinn_nodes), dtype=bool)      # False 表示 SOINN 节点
            ])
            
            # t-SNE 降维（所有特征已经归一化）
            logging.info(f"Computing t-SNE for {len(all_features)} points (samples + SOINN nodes)...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
            embeddings = tsne.fit_transform(all_features)
            
            # 分离样本和节点的嵌入
            sample_embeddings = embeddings[is_sample]
            soinn_embeddings = embeddings[~is_sample]
            sample_labels_plot = all_labels[is_sample]
            soinn_labels_plot = all_labels[~is_sample]
            
            # 创建图形
            plt.figure(figsize=(12, 10))
            
            # 为每个类别定义颜色
            colors = plt.cm.tab10(np.linspace(0, 1, 10))[:5]  # 使用前 5 种颜色
            
            # 绘制样本点（圆点）
            for cls in available_classes:
                cls_mask = sample_labels_plot == cls
                if np.any(cls_mask):
                    plt.scatter(
                        sample_embeddings[cls_mask, 0],
                        sample_embeddings[cls_mask, 1],
                        c=[colors[cls]],
                        marker='o',
                        s=30,
                        alpha=0.6,
                        label=f'Class {cls} (samples)',
                        edgecolors='black',
                        linewidths=0.5
                    )
            
            # 绘制 SOINN 节点（x 标记）
            for cls in available_classes:
                cls_mask = soinn_labels_plot == cls
                if np.any(cls_mask):
                    plt.scatter(
                        soinn_embeddings[cls_mask, 0],
                        soinn_embeddings[cls_mask, 1],
                        c=[colors[cls]],
                        marker='x',
                        s=100,
                        alpha=0.8,
                        label=f'Class {cls} (SOINN nodes)',
                        linewidths=2
                    )
            
            # 绘制边（连接 SOINN 节点）
            # soinn_edges 中的索引是相对于 soinn_embeddings 的
            # 需要映射到 all_features 中的位置
            sample_count = len(sample_features)
            for edge_i, edge_j in soinn_edges:
                # edge_i 和 edge_j 是在 soinn_nodes 中的索引
                # 在 all_features 中的位置是 sample_count + edge_i/j
                global_i = sample_count + edge_i
                global_j = sample_count + edge_j
                if global_i < len(embeddings) and global_j < len(embeddings):
                    plt.plot(
                        [embeddings[global_i, 0], embeddings[global_j, 0]],
                        [embeddings[global_i, 1], embeddings[global_j, 1]],
                        'gray',
                        alpha=0.3,
                        linewidth=0.5
                    )
            
            plt.title(f'SOINN t-SNE Visualization (Task {self._cur_task})', fontsize=14, fontweight='bold')
            plt.xlabel('t-SNE Dimension 1', fontsize=12)
            plt.ylabel('t-SNE Dimension 2', fontsize=12)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # 保存图片
            # 尝试从 args 中获取日志目录，如果没有则使用默认路径
            model_name = self.args.get("model_name", "coda_prompt")
            dataset = self.args.get("dataset", "cifar224")
            init_cls = self.args.get("init_cls", 10)
            increment = self.args.get("increment", 10)
            
            vis_dir = f"logs/{model_name}/{dataset}/{init_cls}/{increment}/soinn_visualizations"
            os.makedirs(vis_dir, exist_ok=True)
            
            vis_path = os.path.join(vis_dir, f"tsne_task{self._cur_task}.png")
            plt.savefig(vis_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"SOINN t-SNE visualization saved to {vis_path}")
            
        except Exception as e:
            logging.error(f"Error in SOINN t-SNE visualization: {e}", exc_info=True)

    def _visualize_hc_soinn_tsne(self, target_classes=None):
        """
        HC-SOINN t-SNE 可视化：可视化class 0-9，每个task结束时都可视化一次
        显示样本点（正常大小圆点）、HC-SOINN点（更大圆点）和NCM点（×标记）
        
        参数:
            target_classes: 要可视化的类别列表。如果为 None，则使用 class 0-9。
        """
        # 如果没有指定类别，则使用 class 0-9
        if target_classes is None:
            init_cls = self.args.get("init_cls", 10)
            target_classes = list(range(min(10, init_cls)))
        
        if len(target_classes) == 0:
            logging.warning(f"No classes to visualize for task {self._cur_task}, skipping visualization")
            return
        
        logging.info(f"Visualizing HC-SOINN t-SNE for task {self._cur_task} classes: {target_classes}")
        
        # 检查 HC-SOINN 是否有数据
        if not hasattr(self, 'hc_soinn') or not hasattr(self.hc_soinn, 'class_mu') or len(self.hc_soinn.class_mu) == 0:
            logging.warning("HC-SOINN class centers are empty, skipping visualization")
            return
        
        # 获取可用的类别（有 NCM 中心的类别）
        available_classes = [cls for cls in target_classes if cls in self.hc_soinn.class_mu]
        if len(available_classes) == 0:
            logging.warning("No target classes found in HC-SOINN, skipping visualization")
            return
        
        # 检查 data_manager 是否可用
        if not hasattr(self, 'data_manager'):
            logging.warning("DataManager not found, skipping visualization")
            return
        
        try:
            # 1. 提取选定类别的训练样本特征
            train_dataset = self.data_manager.get_dataset(
                np.array(available_classes),
                source="train",
                mode="test"  # 使用 test 模式关闭数据增强
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=num_workers
            )
            
            sample_features = []
            sample_labels = []
            
            self._network.eval()
            with torch.no_grad():
                for _, inputs, targets in train_loader:
                    inputs = inputs.to(self._device)
                    # 使用extract_vector提取特征
                    feats = self._network.extract_vector(inputs)
                    feats_np = feats.detach().cpu().numpy()
                    sample_features.append(feats_np)
                    sample_labels.append(targets.numpy())
            
            if len(sample_features) == 0:
                logging.warning("No sample features extracted, skipping visualization")
                return
            
            sample_features = np.concatenate(sample_features, axis=0)
            sample_labels = np.concatenate(sample_labels, axis=0)
            
            # 只保留目标类别的样本
            mask = np.isin(sample_labels, available_classes)
            sample_features = sample_features[mask]
            sample_labels = sample_labels[mask]
            
            if len(sample_features) == 0:
                logging.warning("No samples found for target classes, skipping visualization")
                return
            
            # 归一化样本特征
            sample_features = sample_features / (np.linalg.norm(sample_features, axis=1, keepdims=True) + 1e-8)
            
            # 2. 收集 NCM 点（全局类中心）
            ncm_features = []
            ncm_labels = []
            
            for cls in available_classes:
                if cls in self.hc_soinn.class_mu:
                    ncm_features.append(self.hc_soinn.class_mu[cls])
                    ncm_labels.append(cls)
            
            if len(ncm_features) == 0:
                logging.warning("No NCM centers found, skipping visualization")
                return
            
            ncm_features = np.stack(ncm_features, axis=0)
            ncm_labels = np.array(ncm_labels)
            
            # 归一化 NCM 特征
            ncm_features = ncm_features / (np.linalg.norm(ncm_features, axis=1, keepdims=True) + 1e-8)
            
            # 3. 收集聚类 SOINN 点（HC-SOINN 原型）和边信息
            cluster_features = []
            cluster_labels = []
            cluster_class_mapping = []  # 记录每个 cluster 属于哪个类，以及在该类中的索引
            class_cluster_edges = {}  # 存储每个类的边信息：{cls: {node_idx: set(neighbor_idx)}}
            
            for cls in available_classes:
                if cls in self.hc_soinn.class_clusters:
                    clusters = self.hc_soinn.class_clusters[cls]
                    if len(clusters) > 0:
                        cls_cluster_centers = np.stack([c.center for c in clusters], axis=0)
                        cluster_features.append(cls_cluster_centers)
                        cluster_labels.extend([cls] * len(clusters))
                        # 记录每个 cluster 的类别和索引
                        start_idx = len(cluster_class_mapping)
                        for i in range(len(clusters)):
                            cluster_class_mapping.append((cls, i))  # (类别, 在该类中的索引)
                        
                        # 获取该类的边信息（如果存在）
                        if hasattr(self.hc_soinn, 'class_edges') and cls in self.hc_soinn.class_edges:
                            class_cluster_edges[cls] = self.hc_soinn.class_edges[cls]
            
            if len(cluster_features) > 0:
                cluster_features = np.concatenate(cluster_features, axis=0)
                cluster_labels = np.array(cluster_labels)
                # 归一化聚类特征
                cluster_features = cluster_features / (np.linalg.norm(cluster_features, axis=1, keepdims=True) + 1e-8)
            else:
                cluster_features = np.empty((0, ncm_features.shape[1]))
                cluster_labels = np.array([])
            
            # 4. 合并所有特征进行 t-SNE 降维
            all_features_list = [sample_features, ncm_features]
            all_labels_list = [sample_labels, ncm_labels]
            
            if len(cluster_features) > 0:
                all_features_list.append(cluster_features)
                all_labels_list.append(cluster_labels)
            
            all_features = np.vstack(all_features_list)
            
            if len(all_features) < 2:
                logging.warning("Not enough features for t-SNE, skipping visualization")
                return
            
            # 计算分割点
            sample_end = len(sample_features)
            ncm_end = sample_end + len(ncm_features)
            
            # 使用UMAP实现固定坐标系（支持transform新数据）
            # 如果是第一个任务，fit并保存模型；否则加载并使用已有模型
            model_name = self.args.get("model_name", "coda_prompt")
            dataset = self.args.get("dataset", "cifar224")
            init_cls = self.args.get("init_cls", 10)
            increment = self.args.get("increment", 10)
            use_star = self.args.get("use_feature_alignment", False)
            
            # 确定保存目录
            if use_star:
                vis_base_dir = f"logs/{model_name}/{dataset}/{init_cls}/{increment}/hcsoinn_star_visualizations"
            else:
                vis_base_dir = f"logs/{model_name}/{dataset}/{init_cls}/{increment}/hcsoinn_visualizations"
            os.makedirs(vis_base_dir, exist_ok=True)
            
            umap_model_path = os.path.join(vis_base_dir, "umap_model.pkl")
            
            # 检查是否指定了使用另一个实验的UMAP模型（用于横向对比）
            reference_umap_path = self.args.get("reference_umap_model_path", None)
            if reference_umap_path:
                # 处理相对路径和绝对路径
                if not os.path.isabs(reference_umap_path):
                    # 相对路径：从项目根目录开始
                    reference_umap_path = os.path.join(os.getcwd(), reference_umap_path)
                if os.path.exists(reference_umap_path):
                    umap_model_path = reference_umap_path
                    logging.info(f"Using reference UMAP model from {reference_umap_path} for cross-experiment comparison")
                else:
                    logging.warning(f"Reference UMAP model not found at {reference_umap_path}, will use local model or fallback to t-SNE")
            
            if UMAP_AVAILABLE:
                if self._cur_task == 0 and reference_umap_path is None:
                    # 第一个任务且没有指定参考模型：fit UMAP模型并保存
                    logging.info(f"Fitting UMAP model for fixed coordinate system ({len(all_features)} points)...")
                    reducer = umap.UMAP(
                        n_components=2,
                        n_neighbors=15,
                        min_dist=0.1,
                        metric='cosine',
                        random_state=42
                    )
                    embeddings = reducer.fit_transform(all_features)
                    # 保存模型
                    import pickle
                    with open(umap_model_path, 'wb') as f:
                        pickle.dump(reducer, f)
                    logging.info(f"UMAP model saved to {umap_model_path}")
                else:
                    # 后续任务或使用参考模型：加载已有模型并transform
                    import pickle
                    if os.path.exists(umap_model_path):
                        logging.info(f"Loading UMAP model from {umap_model_path} for fixed coordinate system...")
                        with open(umap_model_path, 'rb') as f:
                            reducer = pickle.load(f)
                        embeddings = reducer.transform(all_features)
                        logging.info(f"Transformed {len(all_features)} points using fixed UMAP model")
                    else:
                        # 如果模型不存在，fallback到t-SNE
                        logging.warning(f"UMAP model not found at {umap_model_path}, falling back to t-SNE")
                        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_features) - 1), max_iter=1000)
                        embeddings = tsne.fit_transform(all_features)
            else:
                # UMAP不可用，使用t-SNE
                logging.info(f"Computing t-SNE for {len(all_features)} points ({len(sample_features)} samples + {len(ncm_features)} NCM + {len(cluster_features)} clusters)...")
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_features) - 1), max_iter=1000)
                embeddings = tsne.fit_transform(all_features)
            
            # 分离不同类型的嵌入
            sample_embeddings = embeddings[:sample_end]
            ncm_embeddings = embeddings[sample_end:ncm_end]
            cluster_embeddings = embeddings[ncm_end:] if len(cluster_features) > 0 else None
            
            # 5. 创建图形
            plt.figure(figsize=(16, 12))
            
            # 为每个类别定义颜色
            num_classes = len(available_classes)
            if num_classes <= 10:
                colors = plt.cm.tab10(np.linspace(0, 1, 10))[:num_classes]
            else:
                colors = plt.cm.tab20(np.linspace(0, 1, 20))[:num_classes]
            
            # 创建类别到颜色的映射
            cls_to_color = {cls: colors[i] for i, cls in enumerate(available_classes)}
            
            # 绘制样本点（正常大小圆点）
            for idx, cls in enumerate(available_classes):
                cls_mask = sample_labels == cls
                if np.any(cls_mask):
                    plt.scatter(
                        sample_embeddings[cls_mask, 0],
                        sample_embeddings[cls_mask, 1],
                        c=[cls_to_color[cls]],
                        marker='o',  # 圆点
                        s=30,  # 正常大小
                        alpha=0.4,
                        label=f'Samples (Class {cls})' if idx == 0 else '',
                        edgecolors='none',
                        zorder=1  # 在最底层
                    )
            
            # 绘制 HC-SOINN 点（更大圆点）和边
            if cluster_embeddings is not None and len(cluster_embeddings) > 0:
                # 首先绘制边（在节点之前，这样边在节点下面）
                for cls in available_classes:
                    if cls in class_cluster_edges and len(class_cluster_edges[cls]) > 0:
                        # 找到该类在 cluster_embeddings 中的索引范围
                        cls_mask = cluster_labels == cls
                        cls_indices = np.where(cls_mask)[0]
                        if len(cls_indices) == 0:
                            continue
                        
                        # 建立类内索引到全局 cluster_embeddings 索引的映射
                        cls_cluster_start = cls_indices[0]
                        cls_edges = class_cluster_edges[cls]
                        
                        # 绘制边（只显示每个节点的前2个最近邻，减少边的数量）
                        for node_idx, neighbors in cls_edges.items():
                            if node_idx >= len(cls_indices):
                                continue  # 跳过无效索引（可能因为截断导致）
                            global_idx_i = cls_cluster_start + node_idx
                            if global_idx_i >= len(cluster_embeddings):
                                continue
                            
                            # 只取前2个邻居，减少边的数量
                            try:
                                limited_neighbors = list(neighbors)[:2]
                            except (TypeError, ValueError):
                                limited_neighbors = []
                            
                            for neighbor_idx in limited_neighbors:
                                if neighbor_idx >= len(cls_indices):
                                    continue
                                global_idx_j = cls_cluster_start + neighbor_idx
                                if global_idx_j >= len(cluster_embeddings):
                                    continue
                                
                                # 只绘制一次边（i < j 避免重复）
                                if node_idx < neighbor_idx:
                                    x_coords = [cluster_embeddings[global_idx_i, 0], cluster_embeddings[global_idx_j, 0]]
                                    y_coords = [cluster_embeddings[global_idx_i, 1], cluster_embeddings[global_idx_j, 1]]
                                    plt.plot(
                                        x_coords, y_coords,
                                        color=cls_to_color[cls],
                                        alpha=0.4,  # 边的透明度
                                        linewidth=2.0,  # 更粗的边
                                        zorder=2  # 在样本点之上，但在节点之下
                                    )
                
                # 然后绘制节点
                for idx, cls in enumerate(available_classes):
                    cls_mask = cluster_labels == cls
                    if np.any(cls_mask):
                        plt.scatter(
                            cluster_embeddings[cls_mask, 0],
                            cluster_embeddings[cls_mask, 1],
                            c=[cls_to_color[cls]],
                            marker='o',  # 圆点
                            s=120,  # 明显更大
                            alpha=0.8,
                            label=f'HC-SOINN (Class {cls})' if idx == 0 else '',
                            edgecolors='black',
                            linewidths=1.5,
                            zorder=3  # 在样本点和边之上
                        )
            
            # 绘制 NCM 点（×标记，缩小并用黑色勾线）
            for idx, cls in enumerate(available_classes):
                cls_mask = ncm_labels == cls
                if np.any(cls_mask):
                    # 先绘制黑色勾线（稍大一些）
                    plt.scatter(
                        ncm_embeddings[cls_mask, 0],
                        ncm_embeddings[cls_mask, 1],
                        c='black',  # 黑色
                        marker='x',  # ×标记
                        s=180,  # 稍大一些，形成勾线效果
                        alpha=0.9,
                        linewidths=3.5,  # 更粗的黑色线条
                        zorder=9  # 在彩色标记之下
                    )
                    # 再绘制彩色标记（稍小一些）
                    plt.scatter(
                        ncm_embeddings[cls_mask, 0],
                        ncm_embeddings[cls_mask, 1],
                        c=[cls_to_color[cls]],
                        marker='x',  # ×标记
                        s=150,  # 缩小尺寸
                        alpha=0.9,
                        label=f'NCM (Class {cls})',
                        linewidths=2.5,
                        zorder=10  # 确保 NCM 点在最上层
                    )
            
            # 根据使用的降维方法设置标题和标签
            if UMAP_AVAILABLE and (self._cur_task == 0 or os.path.exists(umap_model_path)):
                plt.title(f'HC-SOINN UMAP Visualization (Task {self._cur_task}, Fixed Coordinate)', fontsize=14, fontweight='bold')
                plt.xlabel('UMAP Dimension 1', fontsize=12)
                plt.ylabel('UMAP Dimension 2', fontsize=12)
            else:
                plt.title(f'HC-SOINN t-SNE Visualization (Task {self._cur_task})', fontsize=14, fontweight='bold')
                plt.xlabel('t-SNE Dimension 1', fontsize=12)
                plt.ylabel('t-SNE Dimension 2', fontsize=12)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # 保存图片（vis_base_dir已在前面定义）
            vis_dir = vis_base_dir
            
            vis_path = os.path.join(vis_dir, f"tsne_task{self._cur_task}.png")
            plt.savefig(vis_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"HC-SOINN t-SNE visualization saved to {vis_path}")
            logging.info(f"  - Samples: {len(sample_features)}")
            logging.info(f"  - NCM centers: {len(ncm_features)}")
            logging.info(f"  - Cluster prototypes: {len(cluster_features)}")
            
        except Exception as e:
            logging.error(f"Error in HC-SOINN t-SNE visualization: {e}", exc_info=True)

    def load_checkpoint(self, checkpoint_path):
        """
        从checkpoint加载模型参数
        
        参数:
            checkpoint_path: checkpoint文件路径
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self._device)
        
        # 获取模型（如果是DataParallel，需要获取module）
        model = self._network
        if isinstance(model, nn.DataParallel):
            model = model.module
        
        # 加载backbone
        if "backbone_state_dict" in checkpoint:
            model.backbone.load_state_dict(checkpoint["backbone_state_dict"])
            logging.info("Backbone loaded successfully")
        
        # 加载fc层（需要处理类别数可能不同的情况）
        if "fc_state_dict" in checkpoint:
            fc_state = checkpoint["fc_state_dict"]
            current_fc_state = model.fc.state_dict()
            
            # 如果checkpoint中的fc类别数与当前不同，需要调整
            if fc_state["weight"].shape[0] != current_fc_state["weight"].shape[0]:
                logging.warning(
                    f"FC layer size mismatch: checkpoint has {fc_state['weight'].shape[0]} classes, "
                    f"current model has {current_fc_state['weight'].shape[0]} classes. "
                    f"Loading compatible weights only."
                )
                # 只加载兼容的部分
                min_classes = min(fc_state["weight"].shape[0], current_fc_state["weight"].shape[0])
                current_fc_state["weight"][:min_classes] = fc_state["weight"][:min_classes]
                if "bias" in fc_state and "bias" in current_fc_state:
                    current_fc_state["bias"][:min_classes] = fc_state["bias"][:min_classes]
                model.fc.load_state_dict(current_fc_state)
            else:
                model.fc.load_state_dict(fc_state)
            logging.info("FC layer loaded successfully")
        
        # 加载prompt
        if "prompt_state_dict" in checkpoint and checkpoint["prompt_state_dict"] is not None:
            if model.prompt is not None:
                model.prompt.load_state_dict(checkpoint["prompt_state_dict"])
                logging.info("Prompt loaded successfully")
            else:
                logging.warning("Checkpoint contains prompt state but current model has no prompt")
        
        # 恢复任务相关的状态
        if "task" in checkpoint:
            # 恢复任务编号（加载的是task N的checkpoint，接下来要训练task N+1）
            self._cur_task = checkpoint["task"]
            logging.info(f"Restored task: {self._cur_task}")
        
        if "known_classes" in checkpoint:
            self._known_classes = checkpoint["known_classes"]
            logging.info(f"Restored known_classes: {self._known_classes}")
        
        if "total_classes" in checkpoint:
            # 恢复total_classes（但后续可能会根据任务划分重新计算）
            self._total_classes = checkpoint["total_classes"]
            logging.info(f"Restored total_classes: {self._total_classes}")
        
        # 确保模型在正确的设备上
        self._network.to(self._device)
        logging.info("Checkpoint loaded successfully")
    
    def _load_checkpoint_for_task(self):
        """
        为当前任务加载checkpoint（用于自动加载所有checkpoint模式）
        """
        checkpoint_dir = self.args.get("checkpoint_dir", "checkpoints")
        checkpoint_name = self.args.get("checkpoint_name", 
            f"{self.args['model_name']}_{self.args['dataset']}_{self.args['init_cls']}_{self.args['increment']}")
        checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_name}_task{self._cur_task}.pkl")
        
        if os.path.exists(checkpoint_path):
            logging.info(f"Loading checkpoint for task {self._cur_task}: {checkpoint_path}")
            self.load_checkpoint(checkpoint_path)
        else:
            raise FileNotFoundError(
                f"Checkpoint not found for task {self._cur_task}: {checkpoint_path}. "
                f"Please ensure all checkpoints are saved with save_checkpoint=True"
            )


class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class CosineSchedule(_LRScheduler):

    def __init__(self, optimizer, K):
        self.K = K
        super().__init__(optimizer, -1)

    def cosine(self, base_lr):
        # 处理 K=1 的情况，避免除以零
        if self.K <= 1:
            return base_lr
        return base_lr * math.cos((99 * math.pi * (self.last_epoch)) / (200 * (self.K-1)))

    def get_lr(self):
        return [self.cosine(base_lr) for base_lr in self.base_lrs]