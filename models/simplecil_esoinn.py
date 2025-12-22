import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.inc_net import SimpleVitNetKNN
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
from utils.esoinn_classifier import ESOINNClassifier


num_workers = 8
batch_size = 128


class Learner(BaseLearner):
    """
    使用 ESOINN 原型分类器的 SimpleCIL 变体

    - 与 simplecil_knn / simplecil_soinn 类似，只是分类头换成 ESOINNClassifier
    - 训练阶段只负责提取各类特征并交给 ESOINN 构建原型
    - 评估阶段使用 ESOINN 的最近原型进行 top-k 预测
    """

    def __init__(self, args):
        super().__init__(args)
        self._network = SimpleVitNetKNN(args, True)
        self.args = args

        # ESOINN 分类器超参数
        self.esoinn = ESOINNClassifier(
            max_edge_age=args.get("esoinn_max_edge_age", 50),
            iter_threshold=args.get("esoinn_iter_threshold", 100),
            c1=args.get("esoinn_c1", 0.001),
        )

    def after_task(self):
        """每个 task 结束后，更新已知类别数。"""
        self._known_classes = self._total_classes

    def _extract_class_features(self, trainloader, model):
        """
        提取各类的所有特征，供 ESOINN 构建原型。

        注意：传入的 trainloader 应使用 mode="test" 的数据集以关闭数据增强。
        """
        model.eval()
        embedding_list = []
        label_list = []

        with torch.no_grad():
            for _, data, label in trainloader:
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = model.backbone(data)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())

        if len(embedding_list) == 0:
            return

        embeddings = torch.cat(embedding_list, dim=0)
        labels = torch.cat(label_list, dim=0)

        feats_np = embeddings.numpy()
        labels_np = labels.numpy()
        self.esoinn.add_features(feats_np, labels_np)

        proto_info = self.esoinn.prototypes_per_class()
        logging.info(f"ESOINN prototypes per class: {proto_info}")

    def incremental_train(self, data_manager):
        logging.info(
            f"Starting incremental_train: cur_task={self._cur_task}, known_classes={self._known_classes}"
        )
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        logging.info(
            f"After update: cur_task={self._cur_task}, total_classes={self._total_classes}, "
            f"known_classes={self._known_classes}"
        )
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="train"
        )
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        # 用于构建 ESOINN 节点的训练数据：必须关闭数据增强（mode="test"）
        train_dataset_for_esoinn = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="test",
        )
        self.train_loader_for_esoinn = DataLoader(
            train_dataset_for_esoinn,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        if len(self._multiple_gpus) > 1:
            logging.info("Using multiple GPUs")
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._train(self.train_loader, self.test_loader, self.train_loader_for_esoinn)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader, train_loader_for_esoinn):
        """训练阶段：仅负责用当前任务数据更新 ESOINN 原型。"""
        self._network.to(self._device)
        self._extract_class_features(train_loader_for_esoinn, self._network)

    def _eval_cnn(self, loader):
        """
        使用 ESOINN 原型进行分类评估。
        """
        self._network.eval()
        y_pred, y_true = [], []

        with torch.no_grad():
            for _, (_, inputs, targets) in enumerate(loader):
                inputs = inputs.to(self._device)
                features = self._network.extract_vector(inputs)
                features = tensor2numpy(features)

                topk_pred = self.esoinn.predict_topk(
                    features, self.topk, self._total_classes, device=self._device
                )
                y_pred.append(topk_pred)
                y_true.append(targets.numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)

    def eval_task(self):
        """
        评估任务：使用 ESOINN 分类器进行评估。
        返回字典格式，与 trainer.py 的期望一致。
        """
        y_pred, y_true = self._eval_cnn(self.test_loader)
        esoinn_accy = self._evaluate(y_pred, y_true)
        return {"esoinn": esoinn_accy}







