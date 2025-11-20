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
from utils.knn_classifier import KNNClassifier


num_workers = 8
batch_size = 128

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = SimpleVitNetKNN(args, True)
        self.args = args
        # 通用 KNN 分类器
        self.knn = KNNClassifier(
            metric=args.get("knn_metric", "euclidean"),
            use_all_samples=args.get("knn_use_all", True),
            k_neighbors=args.get("knn_k", None),
        )

    def after_task(self):
        self._known_classes = self._total_classes

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

        # 重要：用于构建KNN节点的训练数据必须使用 mode="test" 来关闭数据增强
        # 数据增强会让KNN的节点变得更多更杂乱，影响分类效果
        train_dataset_for_knn = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train", 
            mode="test"  # 使用test模式关闭数据增强
        )
        # 注意：shuffle=False 确保特征提取的一致性（虽然对KNN bank构建影响不大）
        self.train_loader_for_knn = DataLoader(
            train_dataset_for_knn, 
            batch_size=batch_size, 
            shuffle=False,  # 改为False，确保特征提取的一致性
            num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader, self.train_loader_for_knn)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader, train_loader_for_knn):
        self._network.to(self._device)
        self._network.eval()

        # 将当前任务的特征加入 KNN 特征库
        def feature_fn(x):
            return self._network.extract_vector(x)

        self.knn.add_from_loader(train_loader_for_knn, feature_fn, self._device)

    def _eval_cnn(self, loader):
        """
        重写评估函数，使用KNN进行分类
        """
        self._network.eval()
        y_pred, y_true = [], []

        with torch.no_grad():
            for _, (_, inputs, targets) in enumerate(loader):
                inputs = inputs.to(self._device)

                # 提取特征
                features = self._network.extract_vector(inputs).detach().cpu().numpy()

                # 使用通用 KNN 分类器进行 top-k 预测
                batch_topk = self.knn.predict_topk(features, self.topk, self._total_classes)

                y_pred.append(batch_topk)
                y_true.append(targets.numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)
    
    def eval_task(self):
        """
        评估任务：使用KNN分类器进行评估
        返回字典格式，与trainer.py的期望一致
        """
        y_pred, y_true = self._eval_cnn(self.test_loader)
        knn_accy = self._evaluate(y_pred, y_true)

        # 返回字典格式，key为"knn"表示这是KNN分类器的结果
        return {"knn": knn_accy}
