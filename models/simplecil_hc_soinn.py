import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.inc_net import SimpleVitNetKNN
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
from utils.hc_soinn_classifier import HCSOINNClassifier


num_workers = 8
batch_size = 128


class Learner(BaseLearner):
    """
    SimpleCIL + HC-SOINN 分类头

    - 训练阶段仅提取特征并交给 HC-SOINN 生成原型
    - 压缩仅在任务结束时进行（compress），无在线新模式发现
    - 推理阶段使用 NCM+簇中心融合距离
    """

    def __init__(self, args):
        super().__init__(args)
        self._network = SimpleVitNetKNN(args, True)
        self.args = args

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
        )

    def after_task(self):
        # 每个 task 结束时压缩，并更新已知类别
        try:
            self.hc_soinn.compress()
        except Exception as e:
            logging.error(f"HC-SOINN compress error: {e}", exc_info=True)
        self._known_classes = self._total_classes

    def _extract_class_features(self, loader, model):
        """
        提取各类特征用于 HC-SOINN；要求 loader 关闭数据增强（mode='test'）
        """
        model.eval()
        feats, lbs = [], []
        with torch.no_grad():
            for _, data, label in loader:
                data = data.to(self._device)
                label = label.to(self._device)
                emb = model.backbone(data)
                feats.append(emb.cpu())
                lbs.append(label.cpu())

        if len(feats) == 0:
            return
        feats = torch.cat(feats, dim=0).numpy()
        lbs = torch.cat(lbs, dim=0).numpy()
        self.hc_soinn.add_features(feats, lbs)

        proto_info = self.hc_soinn.prototypes_per_class()
        logging.info(f"HC-SOINN prototypes per class: {proto_info}")

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

        # 关闭数据增强用于特征提取
        train_dataset_for_hc = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="test",
        )
        self.train_loader_for_hc = DataLoader(
            train_dataset_for_hc,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        if len(self._multiple_gpus) > 1:
            logging.info("Using multiple GPUs")
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._train(self.train_loader, self.test_loader, self.train_loader_for_hc)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader, train_loader_for_hc):
        # 这里只负责特征提取与 HC-SOINN 更新
        self._network.to(self._device)
        self._extract_class_features(train_loader_for_hc, self._network)

    def _eval_cnn(self, loader):
        """
        使用 HC-SOINN 进行分类评估
        """
        self._network.eval()
        y_pred, y_true = [], []

        with torch.no_grad():
            for _, (_, inputs, targets) in enumerate(loader):
                inputs = inputs.to(self._device)
                features = self._network.extract_vector(inputs)
                features = tensor2numpy(features)

                topk_pred = self.hc_soinn.predict_topk(
                    features, self.topk, self._total_classes, device=self._device
                )
                y_pred.append(topk_pred)
                y_true.append(targets.numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)

    def eval_task(self):
        """
        评估任务：返回与 trainer.py 兼容的结果字典
        """
        y_pred, y_true = self._eval_cnn(self.test_loader)
        acc = self._evaluate(y_pred, y_true)
        # 使用 "hc_soinn" 作为 key 以在日志中正确显示 HC-SOINN
        return {"hc_soinn": acc}


