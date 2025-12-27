import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import OurNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
from utils.hc_soinn_classifier import HCSOINNClassifier
from utils.cluster_structure_analyzer import ClusterStructureAnalyzer
import random

num_workers = 8


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]


def compute_orthogonality_loss(previous_weights_list, current_weights, epsilon=1e-8):
    total_ortho_loss = 0.0
    current_norm = torch.norm(current_weights.flatten())
    current_normalized = current_weights.flatten() / (current_norm + epsilon)

    for prev_weights in previous_weights_list:
        # Normalize previous weights
        prev_norm = torch.norm(prev_weights.flatten())
        prev_normalized = prev_weights.flatten() / (prev_norm + epsilon)

        # Compute absolute dot product (should be close to 0 for orthogonal vectors)
        dot_product = torch.abs(torch.sum(prev_normalized * current_normalized))

        total_ortho_loss += dot_product

    # Average over all previous tasks
    if len(previous_weights_list) > 0:
        total_ortho_loss /= len(previous_weights_list)

    return total_ortho_loss

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = OurNet(args, True)

        self.args = args
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.init_cls = args["init_cls"]
        self.inc = args["increment"]

        self.use_exemplars = args["use_old_data"]
        self.use_init_ptm = args["use_init_ptm"]
        self.use_diagonal = args["use_diagonal"]

        self.recalc_sim = args["recalc_sim"]
        self.alpha = args["alpha"] # forward_reweight is divide by _cur_task
        self.beta = args["beta"]

        self.moni_adam = args["moni_adam"]
        self.adapter_num = args["adapter_num"]

        if self.moni_adam:
            self.use_init_ptm = True
            self.alpha = 1
            self.beta = 1

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


        # 簇结构分析实验：验证特征漂移时簇内部结构是否改变
        self.analyze_cluster_structure_drift = args.get("analyze_cluster_structure_drift", False)
        self.cluster_analyzer = None  # 簇结构分析器（延迟初始化）

        if self.analyze_cluster_structure_drift:
            logging.info("簇结构分析实验启用：将计算Procrustes距离验证特征漂移时簇结构是否改变")
            logging.info("注意：CL-LoRA 将只使用 general_lora 提取特征（不包含 specific_lora）")
            # 定义特征提取函数（适配 CL-LoRA 的网络结构）
            # 重要：只使用 general_lora，不包含 specific_lora
            def feature_extractor(x):
                if isinstance(self._network, nn.DataParallel):
                    backbone = self._network.module.backbone
                else:
                    backbone = self._network.backbone
                
                # 只使用 general_lora 提取特征（不包含 specific_lora）
                B = x.shape[0]
                x = backbone.patch_embed(x)
                cls_tokens = backbone.cls_token.expand(B, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
                x = x + backbone.pos_embed
                x = backbone.pos_drop(x)
                
                # 只使用 general_pos 的 adapter（如果 general_pos 为空，则使用原始 backbone）
                if len(backbone.general_pos) > 0:
                    for j in backbone.general_pos:
                        pos = backbone.adapt_pos.index(j)
                        adapt = backbone.cur_adapter[pos]
                        x = backbone.blocks[j](x, adapt)
                else:
                    # 如果没有 general_pos，则只使用原始 backbone（不使用任何 adapter）
                    for j in range(len(backbone.blocks)):
                        x = backbone.blocks[j](x, adapt=None, prompt=None, rank_prompt=None, block_weight=None)
                
                x = backbone.norm(x)
                feats = x[:, 0, :]  # 提取 CLS token
                return feats

            self.cluster_analyzer = ClusterStructureAnalyzer(
                feature_extractor=feature_extractor,
                device=self._device,
                args=args
            )

    def after_task(self):
        """
        每个 task 结束后的处理流程
        """
        # ========== 压缩 HC-SOINN（生成当前任务的节点）==========
        # 目的：为当前任务的新类别生成 SOINN 原型节点
        if self.use_hc_soinn:
            try:
                # 注意：compress 方法目前不需要额外参数
                # 因为 buffers 中存储的是完整特征，compress 会处理
                # 在 predict_topk 中会按 adapter 分段提取簇中心
                self.hc_soinn.compress()
            except Exception as e:
                logging.error(f"HC-SOINN compress error: {e}", exc_info=True)

        # ========== Step 4: 簇结构分析实验：保存Task 1样本或计算Procrustes距离==========
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

        self._known_classes = self._total_classes
        self._network.freeze()
        self._network.backbone.add_adapter_to_list()

    def get_cls_range(self, task_id):
        if task_id == 0:
            start_cls = 0
            end_cls = self.init_cls
        else:
            start_cls = self.init_cls + (task_id - 1) * self.inc
            end_cls = start_cls + self.inc

        return start_cls, end_cls

    def replace_fc_proxy(self):
        model = self._network
        model = model.eval()
        model.fc.weight.data[self._known_classes:self._total_classes, :] = model.proxy_fc.weight.data
        model.fc.bias.data[self._known_classes:self._total_classes] = model.proxy_fc.bias.data

    def replace_fc(self, train_loader):
        model = self._network
        model = model.eval()

        with torch.no_grad():
            # replace proto for each adapter in the current task
            if self.use_init_ptm:
                start_idx = -1
            else:
                start_idx = 0

            for index in range(start_idx, self._cur_task + 1):
                if self.moni_adam:
                    if index > self.adapter_num - 1:
                        break
                # only use the diagonal feature, index = -1 denotes using init PTM, index = self._cur_task denotes the last adapter's feature
                elif self.use_diagonal and index != -1 and index != self._cur_task:
                    continue

                embedding_list, label_list = [], []
                for i, batch in enumerate(train_loader):
                    (_, data, label) = batch
                    data = data.to(self._device)
                    label = label.to(self._device)
                    embedding = model.backbone.forward_proto(data, adapt_index=index)
                    embedding_list.append(embedding.cpu())
                    label_list.append(label.cpu())

                embedding_list = torch.cat(embedding_list, dim=0)
                label_list = torch.cat(label_list, dim=0)

                class_list = np.unique(self.train_dataset_for_protonet.labels)
                for class_index in class_list:
                    data_index = (label_list == class_index).nonzero().squeeze(-1)
                    embedding = embedding_list[data_index]
                    proto = embedding.mean(0)
                    if self.use_init_ptm:
                        model.fc.weight.data[class_index, (index+1)*self._network.out_dim:(index+2)*self._network.out_dim] = proto
                    else:
                        model.fc.weight.data[class_index, index*self._network.out_dim:(index+1)*self._network.out_dim] = proto
        return

    def get_A_B_Ahat(self, task_id):
        if self.use_init_ptm:
            start_dim = (task_id + 1) * self._network.out_dim
            end_dim = start_dim + self._network.out_dim
        else:
            start_dim = task_id * self._network.out_dim
            end_dim = start_dim + self._network.out_dim

        start_cls, end_cls = self.get_cls_range(task_id)

        # W(Ti)  i is the i-th task index, T is the cur task index, W is a T*T matrix
        A = self._network.fc.weight.data[self._known_classes:, start_dim : end_dim]
        #A = self._network.fc.weight.data[0:, start_dim : end_dim]
        # W(TT)
        B = self._network.fc.weight.data[self._known_classes:, -self._network.out_dim:]
        #B = self._network.fc.weight.data[0:, -self._network.out_dim:]
        # W(ii)
        A_hat = self._network.fc.weight.data[start_cls : end_cls, start_dim : end_dim]

        return A.cpu(), B.cpu(), A_hat.cpu()

    def solve_similarity(self):
        for task_id in range(self._cur_task):
            # print('Solve_similarity adapter:{}'.format(task_id))
            start_cls, end_cls = self.get_cls_range(task_id=task_id)

            A, B, A_hat = self.get_A_B_Ahat(task_id=task_id)

            # calculate similarity matrix between A_hat(old_cls1) and A(new_cls1).
            similarity = torch.zeros(len(A_hat), len(A))
            for i in range(len(A_hat)):
                for j in range(len(A)):
                    similarity[i][j] = torch.cosine_similarity(A_hat[i], A[j], dim=0)

            # softmax the similarity, it will be failed if not use it
            similarity = F.softmax(similarity, dim=1)

            # weight the combination of B(new_cls2)
            B_hat = torch.zeros(A_hat.shape[0], B.shape[1])
            for i in range(len(A_hat)):
                for j in range(len(A)):
                    B_hat[i] += similarity[i][j] * B[j]

            # B_hat(old_cls2)
            self._network.fc.weight.data[start_cls : end_cls, -self._network.out_dim:] = B_hat.to(self._device)

    def solve_sim_reset(self):
        for task_id in range(self._cur_task):
            if self.moni_adam and task_id > self.adapter_num - 2:
                break

            if self.use_init_ptm:
                range_dim = range(task_id + 2, self._cur_task + 2)
            else:
                range_dim = range(task_id + 1, self._cur_task + 1)
            for dim_id in range_dim:
                if self.moni_adam and dim_id > self.adapter_num:
                    break
                # print('Solve_similarity adapter:{}, {}'.format(task_id, dim_id))
                start_cls, end_cls = self.get_cls_range(task_id=task_id)

                start_dim = dim_id * self._network.out_dim
                end_dim = (dim_id + 1) * self._network.out_dim

                # Use the element above the diagonal to calculate
                if self.use_init_ptm:
                    start_cls_old = self.init_cls + (dim_id - 2) * self.inc
                    end_cls_old = self._total_classes
                    start_dim_old = (task_id + 1) * self._network.out_dim
                    end_dim_old = (task_id + 2) * self._network.out_dim
                else:
                    start_cls_old = self.init_cls + (dim_id - 1) * self.inc
                    end_cls_old = self._total_classes
                    start_dim_old = task_id * self._network.out_dim
                    end_dim_old = (task_id + 1) * self._network.out_dim

                A = self._network.fc.weight.data[start_cls_old:end_cls_old, start_dim_old:end_dim_old].cpu()
                B = self._network.fc.weight.data[start_cls_old:end_cls_old, start_dim:end_dim].cpu()
                A_hat = self._network.fc.weight.data[start_cls:end_cls, start_dim_old:end_dim_old].cpu()

                # calculate similarity matrix between A_hat(old_cls1) and A(new_cls1).
                similarity = torch.zeros(len(A_hat), len(A))
                for i in range(len(A_hat)):
                    for j in range(len(A)):
                        similarity[i][j] = torch.cosine_similarity(A_hat[i], A[j], dim=0)

                # softmax the similarity, it will be failed if not use it
                similarity = F.softmax(similarity, dim=1) # dim=1, not dim=0

                # weight the combination of B(new_cls2)
                B_hat = torch.zeros(A_hat.shape[0], B.shape[1])
                for i in range(len(A_hat)):
                    for j in range(len(A)):
                        B_hat[i] += similarity[i][j] * B[j]

                # B_hat(old_cls2)
                self._network.fc.weight.data[start_cls : end_cls, start_dim : end_dim] = B_hat.to(self._device)

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)

        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        self.data_manager = data_manager
        self.train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="train", )
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        self.test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        self.train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="test", )
        self.train_loader_for_protonet = DataLoader(self.train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        self._network.add_fc()
        self.replace_fc(self.train_loader_for_protonet)

        # 训练结束后，构建 HC-SOINN bank（如果启用）
        if self.use_hc_soinn:
            self._build_hc_soinn_bank()

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)

        if self._cur_task == 0 or self.init_cls == self.inc:
            optimizer = self.get_optimizer(lr=self.args["init_lr"])
            scheduler = self.get_scheduler(optimizer, self.args["init_epochs"])
        else:
            # for base 0 setting, the later_lr and later_epochs are not used
            # for base N setting, the later_lr and later_epochs are used
            if "later_lr" not in self.args or self.args["later_lr"] == 0:
                self.args["later_lr"] = self.args["init_lr"]
            if "later_epochs" not in self.args or self.args["later_epochs"] == 0:
                self.args["later_epochs"] = self.args["init_epochs"]

            optimizer = self.get_optimizer(lr=self.args["later_lr"])
            scheduler = self.get_scheduler(optimizer, self.args["later_epochs"])

        self._init_train(train_loader, test_loader, optimizer, scheduler)


    def get_optimizer(self, lr):
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                momentum=0.9,
                lr=lr,
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=lr,
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=lr,
                weight_decay=self.weight_decay
            )

        return optimizer

    def get_scheduler(self, optimizer, epoch):
        if self.args["scheduler"] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epoch, eta_min=self.min_lr)
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None

        return scheduler

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        # torch.autograd.set_detect_anomaly(True)
        if self.moni_adam:
            if self._cur_task > self.adapter_num - 1:
                return

        if self._cur_task == 0 or self.init_cls == self.inc:
            epochs = self.args['init_epochs']
        else:
            epochs = self.args['later_epochs']

        prog_bar = tqdm(range(epochs))

        for _, epoch in enumerate(prog_bar):
            self._network.train()

            losses = 0.0
            correct, total = 0, 0

            if not self._network.backbone.msa_adapt:

                for name, param in self._network.backbone.cur_adapter[0].named_parameters():
                    print(f"Parameter: {name}, Requires Gradient: {param.requires_grad}")
            else:
                for name, param in self._network.backbone.cur_adapter[0][1].named_parameters():
                    print(f"Parameter: {name}, Requires Gradient: {param.requires_grad}")
                for name, param in self._network.backbone.cur_adapter[-1][1].named_parameters():
                    print(f"Parameter: {name}, Requires Gradient: {param.requires_grad}")


            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                aux_targets = targets.clone()

                aux_targets = torch.where(
                    aux_targets - self._known_classes >= 0,
                    aux_targets - self._known_classes,
                    -1,
                )
                output = self._network(inputs, test=False)

                logits = output["logits"]

                loss = F.cross_entropy(logits, aux_targets.long())

                if self._cur_task > 0:
                    kd_ratio = 5.
                    Temperature = 2

                    out_new, out_teacher = self._network.forward_kd(inputs, self._cur_task)
                    out_new_logits = out_new["logits"]
                    out_teacher_logits = out_teacher["logits"]
                    loss_kd = kd_ratio * _KD_loss(out_new_logits, out_teacher_logits, T=Temperature)

                    optimizer.zero_grad()

                    loss_kd.backward()

                    for j in range(len(self._network.backbone.general_pos)):
                        pos = self._network.backbone.adapt_pos.index(self._network.backbone.general_pos[j])
                        for jj in range(len(self._network.backbone.msa)):
                            if self._network.backbone.msa[jj] == 1:
                                temp_weights = 1. * torch.norm(self._network.backbone.old_adapter_list[self._cur_task-1][pos][jj].lora_A.weight,dim=1)
                                temp_weights = 1. * len(temp_weights) * temp_weights / torch.sum(temp_weights)
                                self._network.backbone.cur_adapter[pos][jj].lora_A.weight.grad = temp_weights.unsqueeze(1) * self._network.backbone.cur_adapter[pos][jj].lora_A.weight.grad
                    optimizer.step()
                if self._cur_task > 0:
                    orth_loss_specific = compute_orthogonality_loss(self._network.backbone.block_weight_list, self._network.backbone.block_weight)
                    loss += 0.0001 * orth_loss_specific


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)

                correct += preds.eq(aux_targets.expand_as(preds)).cpu().sum()
                total += len(aux_targets)

            if scheduler:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)

        logging.info(info)


    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model.forward(inputs, test=True)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        calc_task_acc = True

        if calc_task_acc:
            task_correct, task_acc, total = 0, 0, 0

        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)

            with torch.no_grad():
                outputs = self._network.forward(inputs, test=True)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

            # calculate the accuracy by using task_id
            if calc_task_acc:
                task_ids = (targets - self.init_cls) // self.inc + 1
                task_logits = torch.zeros(outputs.shape).to(self._device)
                for i, task_id in enumerate(task_ids):
                    if task_id == 0:
                        start_cls = 0
                        end_cls = self.init_cls
                    else:
                        start_cls = self.init_cls + (task_id-1)*self.inc
                        end_cls = self.init_cls + task_id*self.inc
                    task_logits[i, start_cls:end_cls] += outputs[i, start_cls:end_cls]
                # calculate the accuracy of task_id
                pred_task_ids = (torch.max(outputs, dim=1)[1] - self.init_cls) // self.inc + 1
                task_correct += (pred_task_ids.cpu() == task_ids).sum()

                pred_task_y = torch.max(task_logits, dim=1)[1]
                task_acc += (pred_task_y.cpu() == targets).sum()
                total += len(targets)

        if calc_task_acc:
            logging.info("Task correct: {}".format(tensor2numpy(task_correct) * 100 / total))
            logging.info("Task acc: {}".format(tensor2numpy(task_acc) * 100 / total))

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _get_hc_soinn_feature_fn(self):
        """
        获取HC-SOINN特征提取函数（统一特征提取逻辑）
        
        重要：CL-LoRA的backbone在训练和测试模式下返回的特征不同：
        - 训练模式（test=False）：只返回当前任务的adapter特征（768维）
        - 测试模式（test=True）：返回所有adapter的concat特征（768 * (task_num+1)维）
        
        为了保持一致性，HC-SOINN应该使用测试模式的特征（所有adapter的concat），
        这样在推理时才能正确匹配。
        """
        def feature_fn(x):
            # 确保模型处于eval模式
            if isinstance(self._network, nn.DataParallel):
                backbone = self._network.module.backbone
            else:
                backbone = self._network.backbone
            
            # 使用test=True模式，提取所有adapter的concat特征
            # 这样训练和测试时特征维度一致
            feats = backbone(x, test=True, use_init_ptm=self.use_init_ptm)
            
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
        if not self.use_hc_soinn:
            return

        # 确保模型处于 eval 模式（重要：特征提取时应该关闭 dropout 等）
        self._network.eval()

        hc_bank_empty = (not hasattr(self, "hc_soinn")) or (len(getattr(self.hc_soinn, "class_clusters", {})) == 0)
        if hc_bank_empty:
            logging.info(f"HC-SOINN bank is empty!!!")
        
        feature_fn = self._get_hc_soinn_feature_fn()

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
            # HC-SOINN 标准模式：存储完整特征的原型
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

        # 注意：compress() 在 after_task() 中已经调用，这里不需要重复调用
        # 但是，如果这是第一个任务，after_task 可能在 _build_hc_soinn_bank 之前调用，
        # 所以这里需要确保在添加特征后再次压缩（如果还没有压缩过）
        # 实际上，after_task 在 incremental_train 之后调用，所以这里不需要再次压缩

    def _eval_hc_soinn(self, loader):
        """
        使用 HC-SOINN 分类器进行评估（CL-LoRA 对角 NCM 模式）
        
        根据 CL-LoRA 的对角 NCM 逻辑：
        - 每个 adapter 只负责特定类别范围
        - Adapter 0: 类别 0-19
        - Adapter 1: 类别 20-39
        - ...
        - 每个类别只使用"属于它的 adapter"的特征和原型
        
        实现方式：
        - 从 HC-SOINN 获取完整特征的原型（num_adapters * out_dim 维）
        - 手动实现对角 NCM：对每个 adapter，提取对应的特征段和原型段，计算相似度
        - 合并所有 adapter 的 logits
        """
        self._network.eval()
        y_pred, y_true = [], []
        feature_fn = self._get_hc_soinn_feature_fn()
        out_dim = self._network.out_dim
        num_adapters = self._cur_task + 1
        
        # 获取 HC-SOINN 的所有类别原型（完整特征）
        hc_soinn = self.hc_soinn
        all_classes = sorted(hc_soinn.class_mu.keys())
        
        if len(all_classes) == 0:
            logging.warning("HC-SOINN has no prototypes, returning zero predictions")
            for _, (_, inputs, targets) in enumerate(loader):
                batch_size = inputs.shape[0]
                topk_pred = np.zeros((batch_size, self.topk), dtype=np.int64)
                y_pred.append(topk_pred)
                y_true.append(targets.cpu().numpy())
            return np.concatenate(y_pred), np.concatenate(y_true)
        
        with torch.no_grad():
            for _, (_, inputs, targets) in enumerate(loader):
                inputs = inputs.to(self._device)
                feats = feature_fn(inputs)  # [B, num_adapters * out_dim]
                feats = feats.detach().cpu()
                if len(feats.shape) == 1:
                    feats = feats.reshape(1, -1)
                elif len(feats.shape) != 2:
                    raise ValueError(f"Expected 2D features [B, D], got shape {feats.shape}")
                
                batch_size = feats.shape[0]
                feats_t = torch.from_numpy(feats.numpy()).float().to(self._device)  # [B, num_adapters * out_dim]
                
                # CL-LoRA 对角 NCM：按 adapter 分段计算 logits，然后合并
                all_logits = []
                
                for adapter_idx in range(num_adapters):
                    # 提取当前 adapter 的特征段
                    adapter_feats = feats_t[:, adapter_idx * out_dim:(adapter_idx + 1) * out_dim]  # [B, out_dim]
                    
                    # 确定当前 adapter 对应的类别范围
                    if adapter_idx == 0:
                        start_cls = 0
                        end_cls = self.init_cls
                    else:
                        start_cls = self.init_cls + (adapter_idx - 1) * self.inc
                        end_cls = start_cls + self.inc
                    
                    # 构建当前 adapter 的 NCM 中心（从完整原型中提取对应段）
                    ncm_centers = []
                    valid_classes = []
                    for cls in all_classes:
                        if start_cls <= cls < end_cls:  # 只处理属于当前 adapter 的类别
                            cls_mu_full = hc_soinn.class_mu[cls]  # [num_adapters * out_dim]
                            if cls_mu_full.shape[0] >= (adapter_idx + 1) * out_dim:
                                # 提取对应 adapter 段的原型
                                cls_mu_adapter = cls_mu_full[adapter_idx * out_dim:(adapter_idx + 1) * out_dim]
                                ncm_centers.append(cls_mu_adapter)
                                valid_classes.append(cls)
                    
                    if len(ncm_centers) == 0:
                        # 如果没有有效的原型，创建零 logits
                        adapter_logits = torch.zeros((batch_size, end_cls - start_cls), device=self._device)
                    else:
                        # 计算 NCM 相似度
                        ncm_centers_t = torch.from_numpy(np.stack(ncm_centers)).float().to(self._device)  # [C, out_dim]
                        
                        # 归一化
                        adapter_feats_norm = torch.nn.functional.normalize(adapter_feats, p=2, dim=1)
                        ncm_centers_norm = torch.nn.functional.normalize(ncm_centers_t, p=2, dim=1)
                        
                        # 计算相似度 [B, out_dim] @ [out_dim, C] -> [B, C]
                        sim_ncm = torch.mm(adapter_feats_norm, ncm_centers_norm.t())  # [B, C]
                        
                        # 计算子簇距离（如果有）
                        dist_sub_list = []
                        for cls in valid_classes:
                            clusters = hc_soinn.class_clusters.get(cls, [])
                            if clusters:
                                # 提取每个子簇中心对应的 adapter 段
                                cluster_centers = []
                                for c in clusters:
                                    if c.center.shape[0] >= (adapter_idx + 1) * out_dim:
                                        cluster_center_adapter = c.center[adapter_idx * out_dim:(adapter_idx + 1) * out_dim]
                                        cluster_centers.append(cluster_center_adapter)
                                
                                if cluster_centers:
                                    cluster_centers_t = torch.from_numpy(np.stack(cluster_centers)).float().to(self._device)
                                    cluster_centers_norm = torch.nn.functional.normalize(cluster_centers_t, p=2, dim=1)
                                    sim_cluster = torch.mm(adapter_feats_norm, cluster_centers_norm.t())  # [B, num_clusters]
                                    dist_cluster = 1.0 - sim_cluster
                                    min_dist, _ = dist_cluster.min(dim=1)  # [B]
                                    dist_sub_list.append(min_dist)
                                else:
                                    # 如果没有有效的子簇，使用 NCM 距离
                                    cls_idx = valid_classes.index(cls)
                                    dist_sub_list.append(1.0 - sim_ncm[:, cls_idx])
                            else:
                                # 如果没有子簇，使用 NCM 距离
                                cls_idx = valid_classes.index(cls)
                                dist_sub_list.append(1.0 - sim_ncm[:, cls_idx])
                        
                        if dist_sub_list:
                            dist_sub = torch.stack(dist_sub_list, dim=1)  # [B, C]
                        else:
                            dist_sub = 1.0 - sim_ncm
                        
                        # 融合分数
                        dist_ncm = 1.0 - sim_ncm  # [B, C]
                        final_scores = hc_soinn.alpha * dist_ncm + (1.0 - hc_soinn.alpha) * dist_sub  # [B, C]
                        
                        # 构建完整的 logits（包含所有 start_cls 到 end_cls 的类别）
                        adapter_logits = torch.full((batch_size, end_cls - start_cls), float('inf'), device=self._device)
                        for i, cls in enumerate(valid_classes):
                            if start_cls <= cls < end_cls:
                                adapter_logits[:, cls - start_cls] = final_scores[:, i]
                    
                    all_logits.append(adapter_logits)
                
                # 合并所有 adapter 的 logits
                combined_logits = torch.cat(all_logits, dim=1)  # [B, total_classes]
                
                # 获取 topk 预测
                _, topk_indices = torch.topk(combined_logits, k=self.topk, dim=1, largest=False)
                topk_pred = topk_indices.cpu().numpy()
                
                y_pred.append(topk_pred)
                y_true.append(targets.cpu().numpy())

        if len(y_pred) == 0:
            logging.warning("No predictions generated from HC-SOINN evaluation")
            return np.array([]), np.array([])

        return np.concatenate(y_pred), np.concatenate(y_true)

    def eval_task(self):
        """
        评估任务：使用FC分类器进行评估，如果启用HC-SOINN则同时评估
        返回字典格式，与LAMDA-PILOT_2的trainer.py的期望一致
        """
        results = {}
        
        # 1. 使用原始FC分类器评估
        y_pred, y_true = self._eval_cnn(self.test_loader)
        results["fc"] = self._evaluate(y_pred, y_true)
        
        # 2. 使用 HC-SOINN 分类器评估（如果启用）
        if getattr(self, "use_hc_soinn", False):
            y_pred_hc, y_true_hc = self._eval_hc_soinn(self.test_loader)
            results["hc_soinn"] = self._evaluate(y_pred_hc, y_true_hc)
        
        return results

