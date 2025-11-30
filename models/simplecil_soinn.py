import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet,SimpleCosineIncrementalNet,SimpleVitNet,SimpleVitNetKNN
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy
from utils.soinn_classifier import SOINNClassifier


num_workers = 8
batch_size = 128

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = SimpleVitNetKNN(args, True)
        self.args = args
        # SOINN 分类器
        # 处理 seed 参数：如果传入的是列表，取第一个元素
        seed = args.get("seed", None)
        if isinstance(seed, (list, tuple)) and len(seed) > 0:
            seed = seed[0]
        
        self.soinn = SOINNClassifier(
            ad=args.get("soinn_ad", 20),
            lam=args.get("soinn_lam", 20),
            max_nodes_per_class=args.get("soinn_max_nodes_per_class", None),
            seed=seed,
            threshold_scale=args.get("soinn_threshold_scale", 0.3),
            k_neighbors=args.get("soinn_k_neighbors", 3),
        )

    def after_task(self):
        self._known_classes = self._total_classes

    def store_class_features(self, trainloader, model, args):
        """
        存储每个类的所有训练样本特征，用于SOINN分类
        
        注意：传入的trainloader应该使用 mode="test" 的数据集，以确保没有数据增强
        数据增强会让SOINN的节点变得更多更杂乱，影响分类效果
        """
        model = model.eval()
        embedding_list = []
        label_list = []

        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_, data, label) = batch
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = model.backbone(data)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())

        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        # 为每个类别存储所有样本的特征
        class_list = np.unique(self.train_dataset.labels)
        for class_index in class_list:
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            class_embeddings = embedding_list[data_index]

            # 将特征转换为numpy并添加到SOINN分类器
            class_features = class_embeddings.numpy()
            class_labels = np.full((len(class_features),), class_index, dtype=np.int64)
            
            # 使用SOINN分类器的add_features方法
            self.soinn.add_features(class_features, class_labels)

        # 打印每个类的原型数量
        proto_info = self.soinn.prototypes_per_class()
        logging.info(f"SOINN prototypes per class: {proto_info}")

        return model

    def incremental_train(self, data_manager):
        logging.info(f"Starting incremental_train: cur_task={self._cur_task}, known_classes={self._known_classes}")
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        logging.info(f"After update: cur_task={self._cur_task}, total_classes={self._total_classes}, known_classes={self._known_classes}")
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train", )
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # 重要：用于构建SOINN节点的训练数据必须使用 mode="test" 来关闭数据增强
        # 数据增强会让SOINN的节点变得更多更杂乱，影响分类效果
        train_dataset_for_soinn = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train", 
            mode="test"  # 使用test模式关闭数据增强
        )
        # 注意：shuffle=False 确保特征提取的一致性
        self.train_loader_for_soinn = DataLoader(
            train_dataset_for_soinn, 
            batch_size=batch_size, 
            shuffle=False,  # 改为False，确保特征提取的一致性
            num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader, self.train_loader_for_soinn)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader, train_loader_for_soinn):
        self._network.to(self._device)
        # 存储新类别的特征用于SOINN分类
        self.store_class_features(train_loader_for_soinn, self._network, None)

    def _eval_cnn(self, loader):
        """
        重写评估函数，使用SOINN进行分类
        """
        self._network.eval()
        y_pred, y_true = [], []

        with torch.no_grad():
            for _, (_, inputs, targets) in enumerate(loader):
                inputs = inputs.to(self._device)

                # 提取特征
                features = self._network.extract_vector(inputs)
                features = tensor2numpy(features)

                # 使用SOINN分类器进行预测（传入 device 以启用 GPU 加速）
                topk_pred = self.soinn.predict_topk(features, self.topk, self._total_classes, device=self._device)
                y_pred.append(topk_pred)
                y_true.append(targets.numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)
    
    def eval_task(self):
        """
        评估任务：使用SOINN分类器进行评估
        返回字典格式，与trainer.py的期望一致
        """
        y_pred, y_true = self._eval_cnn(self.test_loader)
        soinn_accy = self._evaluate(y_pred, y_true)
        
        # 返回字典格式，key为"soinn"表示这是SOINN分类器的结果
        return {"soinn": soinn_accy}

