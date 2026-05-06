import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import numpy as np
from datetime import datetime
import time


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):

    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["prefix"],
        args["seed"],
        args["backbone_type"],
        timestamp,
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random(args["seed"])
    _set_device(args)
    print_args(args)

    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args,
    )
    
    args["nb_classes"] = data_manager.nb_classes # update args
    args["nb_tasks"] = data_manager.nb_tasks
    model = factory.get_model(args["model_name"], args)

    fc_curve = {"top1": [], "top5": []}
    knn_curve = {"top1": [], "top5": []}
    ncm_curve = {"top1": [], "top5": []}
    hc_soinn_curve = {"top1": [], "top5": []}
    fc_matrix, knn_matrix, ncm_matrix, hc_soinn_matrix = [], [], [], []

    start_task = 0
    if args.get("load_all_checkpoints", False):
        logging.info("Auto-loading all checkpoints mode enabled: will load checkpoints for all tasks and skip training")
        start_task = 0

    for task in range(start_task, data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        
        is_last_task = (task == data_manager.nb_tasks - 1)
        train_time = None
        eval_time = None
        
        if is_last_task:
            train_start = time.time()
            model.incremental_train(data_manager)
            train_end = time.time()
            train_time = train_end - train_start
            
            eval_start = time.time()
            eval_results = model.eval_task()
            eval_end = time.time()
            eval_time = eval_end - eval_start
            
            logging.info("=" * 60)
            logging.info("Last Task (Task {}) Time Statistics:".format(task))
            logging.info("  Training time: {:.2f} seconds ({:.2f} minutes)".format(train_time, train_time / 60))
            logging.info("  Evaluation time: {:.2f} seconds ({:.2f} minutes)".format(eval_time, eval_time / 60))
            logging.info("  Total time: {:.2f} seconds ({:.2f} minutes)".format(train_time + eval_time, (train_time + eval_time) / 60))
            logging.info("=" * 60)
        else:
            model.incremental_train(data_manager)
            eval_results = model.eval_task()
        
        model.after_task()

        if "fc" in eval_results:
            fc_accy = eval_results["fc"]
            logging.info("FC: {}".format(fc_accy["grouped"]))
            fc_keys = [key for key in fc_accy["grouped"].keys() if '-' in key]
            fc_values = [fc_accy["grouped"][key] for key in fc_keys]
            fc_matrix.append(fc_values)
            fc_curve["top1"].append(fc_accy["top1"])
            fc_curve["top5"].append(fc_accy["top5"])

        if "knn" in eval_results:
            knn_accy = eval_results["knn"]
            logging.info("KNN: {}".format(knn_accy["grouped"]))
            knn_keys = [key for key in knn_accy["grouped"].keys() if '-' in key]
            knn_values = [knn_accy["grouped"][key] for key in knn_keys]
            knn_matrix.append(knn_values)
            knn_curve["top1"].append(knn_accy["top1"])
            knn_curve["top5"].append(knn_accy["top5"])

        if "ncm" in eval_results:
            ncm_accy = eval_results["ncm"]
            logging.info("NCM: {}".format(ncm_accy["grouped"]))
            ncm_keys = [key for key in ncm_accy["grouped"].keys() if '-' in key]
            ncm_values = [ncm_accy["grouped"][key] for key in ncm_keys]
            ncm_matrix.append(ncm_values)
            ncm_curve["top1"].append(ncm_accy["top1"])
            ncm_curve["top5"].append(ncm_accy["top5"])

        if "hc_soinn" in eval_results:
            hc_soinn_accy = eval_results["hc_soinn"]
            logging.info("HC-SOINN: {}".format(hc_soinn_accy["grouped"]))
            hc_soinn_keys = [key for key in hc_soinn_accy["grouped"].keys() if '-' in key]
            hc_soinn_values = [hc_soinn_accy["grouped"][key] for key in hc_soinn_keys]
            hc_soinn_matrix.append(hc_soinn_values)
            hc_soinn_curve["top1"].append(hc_soinn_accy["top1"])
            hc_soinn_curve["top5"].append(hc_soinn_accy["top5"])

        curves_to_log = []
        if "fc" in eval_results:
            curves_to_log.append(("FC", fc_curve))
        if "knn" in eval_results:
            curves_to_log.append(("KNN", knn_curve))
        if "ncm" in eval_results:
            curves_to_log.append(("NCM", ncm_curve))
        if "hc_soinn" in eval_results:
            curves_to_log.append(("HC-SOINN", hc_soinn_curve))

        for name, curve in curves_to_log:
            logging.info("{} top1 curve: {}".format(name, curve["top1"]))
            logging.info("{} top5 curve: {}".format(name, curve["top5"]))

        avg_accs = []
        if "fc" in eval_results and len(fc_curve["top1"]) > 0:
            avg_fc = sum(fc_curve["top1"]) / len(fc_curve["top1"])
            avg_accs.append(("FC", avg_fc))
            print('Average Accuracy (FC):', avg_fc)
            logging.info("Average Accuracy (FC): {}".format(avg_fc))

        if "knn" in eval_results and len(knn_curve["top1"]) > 0:
            avg_knn = sum(knn_curve["top1"]) / len(knn_curve["top1"])
            avg_accs.append(("KNN", avg_knn))
            print('Average Accuracy (KNN):', avg_knn)
            logging.info("Average Accuracy (KNN): {}".format(avg_knn))

        if "ncm" in eval_results and len(ncm_curve["top1"]) > 0:
            avg_ncm = sum(ncm_curve["top1"]) / len(ncm_curve["top1"])
            avg_accs.append(("NCM", avg_ncm))
            print('Average Accuracy (NCM):', avg_ncm)
            logging.info("Average Accuracy (NCM): {}".format(avg_ncm))

        if "hc_soinn" in eval_results and len(hc_soinn_curve["top1"]) > 0:
            avg_hc_soinn = sum(hc_soinn_curve["top1"]) / len(hc_soinn_curve["top1"])
            avg_accs.append(("HC-SOINN", avg_hc_soinn))
            print('Average Accuracy (HC-SOINN):', avg_hc_soinn)
            logging.info("Average Accuracy (HC-SOINN): {}".format(avg_hc_soinn))

        logging.info("")

    if 'print_forget' in args.keys() and args['print_forget'] is True:
        if len(fc_matrix) > 0:
            np_acctable = np.zeros([task + 1, task + 1])
            for idxx, line in enumerate(fc_matrix):
                idxy = len(line)
                np_acctable[idxx, :idxy] = np.array(line)
            np_acctable = np_acctable.T
            forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
            print('Accuracy Matrix (FC):')
            print(np_acctable)
            logging.info('Forgetting (FC): {}'.format(forgetting))
        if len(knn_matrix) > 0:
            np_acctable = np.zeros([task + 1, task + 1])
            for idxx, line in enumerate(knn_matrix):
                idxy = len(line)
                np_acctable[idxx, :idxy] = np.array(line)
            np_acctable = np_acctable.T
            forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
            print('Accuracy Matrix (KNN):')
            print(np_acctable)
            logging.info('Forgetting (KNN): {}'.format(forgetting))
        if len(ncm_matrix) > 0:
            np_acctable = np.zeros([task + 1, task + 1])
            for idxx, line in enumerate(ncm_matrix):
                idxy = len(line)
                np_acctable[idxx, :idxy] = np.array(line)
            np_acctable = np_acctable.T
            forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
            print('Accuracy Matrix (NCM):')
            print(np_acctable)
            logging.info('Forgetting (NCM): {}'.format(forgetting))
        if len(hc_soinn_matrix) > 0:
            np_acctable = np.zeros([task + 1, task + 1])
            for idxx, line in enumerate(hc_soinn_matrix):
                idxy = len(line)
                np_acctable[idxx, :idxy] = np.array(line)
            np_acctable = np_acctable.T
            forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
            print('Accuracy Matrix (HC-SOINN):')
            print(np_acctable)
            logging.info('Forgetting (HC-SOINN): {}'.format(forgetting))


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))