import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
import math
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import SEMAVitNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
from backbone.sema_block import SEMAModules
from utils.hc_soinn_classifier import HCSOINNClassifier

num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = SEMAVitNet(args, True)
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.args = args
        
        # HC-SOINN plugin
        self.use_hc_soinn = args.get("use_hc_soinn", False)
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
            )

    def after_task(self):
        # HC-SOINN compress: 每个任务结束后压缩当前任务的特征
        if getattr(self, "use_hc_soinn", False):
            try:
                self.hc_soinn.compress()
            except Exception as e:
                logging.error(f"HC-SOINN compress error: {e}", exc_info=True)
        
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        if self._cur_task == 0:
            self._network.fc = nn.Linear(768, data_manager.nb_classes)
            nn.init.kaiming_uniform_(self._network.fc.weight, a=math.sqrt(5))
            nn.init.zeros_(self._network.fc.bias)
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train", )
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="test", )
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        
        # 训练结束后，构建 HC-SOINN bank（如果启用）
        if getattr(self, "use_hc_soinn", False):
            self._build_hc_soinn_bank()

    def _train(self, train_loader, test_loader):
        
        self._network.to(self._device)
        
        if self._cur_task == 0:
            # show total parameters and trainable parameters
            total_params = sum(p.numel() for p in self._network.parameters())
            print(f'{total_params:,} total parameters.')
            total_trainable_params = sum(
                p.numel() for p in self._network.parameters() if p.requires_grad)
            print(f'{total_trainable_params:,} training parameters.')
            self._train_new(train_loader, test_loader)
        else:
            for module in self._network.backbone.modules():
                if isinstance(module, SEMAModules):
                    module.detecting_outlier = True
            detect_loader = DataLoader(train_loader.dataset, batch_size=self.args["detect_batch_size"], shuffle=True, num_workers=num_workers)     
            added = self._detect_outlier(detect_loader, train_loader, test_loader, 0)

            for module in self._network.backbone.modules():
                if isinstance(module, SEMAModules):
                    module.detecting_outlier = False
            if added == 0:
                self.update_optimizer_and_scheduler(num_epoch=self.args['func_epoch'], lr=self.init_lr)
                self._init_train(self.args['func_epoch'], train_loader, test_loader, self.optimizer, self.scheduler, phase='func')
            
        for module in self._network.backbone.modules():
            if isinstance(module, SEMAModules):
                module.end_of_task_training()

    def _train_new(self, train_loader, test_loader):
        self.update_optimizer_and_scheduler(num_epoch=self.args['func_epoch'], lr=self.init_lr)
        self._init_train(self.args['func_epoch'], train_loader, test_loader, self.optimizer, self.scheduler, phase='func')
        self.update_rd_optimizer_and_scheduler(num_epoch=self.args['rd_epoch'], lr=self.args['rd_lr'])
        self._init_train(self.args['rd_epoch'], train_loader, test_loader, self.rd_optimizer, self.rd_scheduler, phase='rd')

    def _detect_outlier(self, detect_loader, train_loader, test_loader, added):
        #检测并训练新组件
        is_added = False
        for i, (_, inputs, targets) in enumerate(detect_loader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            model_outcome = self._network(inputs)
            added_record = model_outcome["added_record"]

            if sum(added_record) > 0:
                added += 1
                is_added = True
                for module in self._network.backbone.modules():
                    if isinstance(module, SEMAModules):
                        module.detecting_outlier = False
                self._train_new(train_loader, test_loader)
                for module in self._network.backbone.modules():
                    if isinstance(module, SEMAModules):
                        module.detecting_outlier = True
                for module in self._network.backbone.modules():
                    if isinstance(module, SEMAModules):
                        module.freeze_functional()
                        module.freeze_rd()
                        module.reset_newly_added_status()
        
        if is_added:
            return self._detect_outlier(detect_loader, train_loader, test_loader, added)
        else:
            return added

    def _init_train(self, total_epoch, train_loader, test_loader, optimizer, scheduler, phase='func'):
        prog_bar = tqdm(range(total_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outcome = self._network(inputs)

                logits = outcome["logits"]
                logits = logits[:, :self._total_classes]
                if self._cur_task > 0:
                    logits[:, :self._known_classes] = -float('inf')

                if phase == "func":
                    loss = F.cross_entropy(logits, targets.long())
                elif phase == "rd":
                    logits = outcome["logits"]
                    loss = outcome["rd_loss"]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if (epoch + 1) % 5 == 0 or epoch == total_epoch - 1:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "{} Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    phase,
                    self._cur_task,
                    epoch + 1,
                    total_epoch,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "{} Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    phase,
                    self._cur_task,
                    epoch + 1,
                    total_epoch,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)

        logging.info(info)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outcome = self._network(inputs)
                logits = outcome["logits"]
                outputs = logits[:, :self._total_classes]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]
    
    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outcome = self._network(inputs)
                logits = outcome["logits"]
                outputs = logits[:, :self._total_classes]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)
    
    def _get_soinn_feature_fn(self):
        """
        获取SOINN特征提取函数（统一特征提取逻辑）
        """
        def feature_fn(x):
            if isinstance(self._network, nn.DataParallel):
                feats = self._network.module.extract_vector(x)
            else:
                feats = self._network.extract_vector(x)
            
            # SEMA 的 extract_vector 返回字典，需要提取 "features"
            if isinstance(feats, dict):
                feats = feats["features"]
            
            if not isinstance(feats, torch.Tensor):
                raise TypeError(f"Expected tensor, got {type(feats)}")
            
            if len(feats.shape) == 1:
                feats = feats.reshape(1, -1)
            elif len(feats.shape) != 2:
                raise ValueError(f"Expected 2D features [B, D], got shape {feats.shape}")
            
            return feats
        
        return feature_fn
    
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
        logging.info(f"Building HC-SOINN bank: adding new classes ({self._known_classes}-{self._total_classes-1})")
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
        
        # 注意：compress() 在 after_task() 中调用，这里只添加特征
    
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


    def update_optimizer_and_scheduler(self, num_epoch=20, lr=None):
        lr = self.args["init_lr"] if lr is None else lr
        func_params = [p for n,p in self._network.named_parameters() if ('functional' in n or 'router' in n or 'fc' in n) and p.requires_grad]
        if self.args['optimizer']=='sgd':
            self.optimizer = optim.SGD(func_params, momentum=0.9, lr=lr,weight_decay=self.args["weight_decay"])
        elif self.args['optimizer']=='adam':
            self.optimizer = optim.AdamW(func_params, lr=lr, weight_decay=self.args["weight_decay"])

        min_lr = self.args['min_lr'] if self.args['min_lr'] is not None else 1e-8
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epoch, eta_min=min_lr)    

    def update_rd_optimizer_and_scheduler(self, num_epoch=20, lr=None):
        lr = self.args["rd_lr"] if lr is None else lr
        rd_params = [p for n,p in self._network.named_parameters() if 'rd' in n and p.requires_grad]
        if self.args['optimizer']=='sgd':
            self.rd_optimizer = optim.SGD(rd_params, momentum=0.9, lr=lr,weight_decay=self.args["weight_decay"])
        elif self.args['optimizer']=='adam':
            self.rd_optimizer = optim.AdamW(rd_params, lr=lr, weight_decay=self.args["weight_decay"])
        
        min_lr = self.args['min_lr'] if self.args['min_lr'] is not None else 1e-8
        self.rd_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.rd_optimizer, T_max=num_epoch, eta_min=min_lr) if self.rd_optimizer else None
        
    def save_checkpoint(self, filename):
        state_dict = self._network.state_dict()
        save_dict = {}
        for k, v in state_dict.items():
            if 'adapter' in k or ('fc' in k and 'block' not in k):
                save_dict[k] = v
        torch.save(save_dict, "{}.pth".format(filename))

    def load_checkpoint(self, filename):
        self._network.load_state_dict(torch.load(filename), strict=False)
    
    def eval_task(self):
        results = {}
        
        # 1. 使用原始FC分类器评估
        y_pred_fc, y_true_fc = self._eval_cnn(self.test_loader)
        results["fc"] = self._evaluate(y_pred_fc, y_true_fc)
        
        # 2. 使用 HC-SOINN 分类器评估（如果启用）
        if getattr(self, "use_hc_soinn", False):
            y_pred_hc, y_true_hc = self._eval_hc_soinn(self.test_loader)
            results["hc_soinn"] = self._evaluate(y_pred_hc, y_true_hc)
        
        # 3. 使用NME分类器评估（如果存在）
        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
            results["nme"] = nme_accy
        else:
            nme_accy = None

        # 为了向后兼容，如果只有fc结果，返回元组格式
        if len(results) == 1 and "fc" in results:
            return results["fc"], nme_accy
        
        return results

