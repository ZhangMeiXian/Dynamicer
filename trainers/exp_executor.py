# -*- coding: UTF-8 -*-
# !/usr/bin/python3.8

"""
@ModuleName: experiment executor
@author: zhangmeixian
"""
import os
import torch
import random
import numpy as np
import torch.nn as nn
from trainers.methods.transformers import Autoformer, FEDformer, Informer, Transformer
from trainers.methods.ns_transformers import ns_Autoformer, ns_FEDformer, ns_Informer, ns_Transformer
from trainers.dataset_libs.data_preprocessor import DataPreprocessor
from trainers.dataset_libs.batch_loader import BatchLoader
from commons.variabler import Variabler
from commons.logger import get_logger

logger = get_logger(__name__)


class ExpExecutor(object):
    """
    experiment entrance
    """
    def __init__(self, args):
        self.args = args
        self.device = self._get_device()
        self.model = self._get_model()
        if self.model is not None:
            self.model = self.model.to(self.device)

    def execute(self):
        """
        experiment executor
        """
        # get gpu param
        self.args.use_gpu = True if torch.cuda.is_available() and self.args.use_gpu else False
        # random seed config
        fix_seed = self.args.seed
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)
        # GPU config
        if self.args.use_gpu:
            if self.args.use_multi_gpu:
                self.args.devices = self.args.devices.replace(" ", "")
                device_ids = self.args.devices.split(",")
                self.args.device_ids = [int(id_) for id_ in device_ids]
                self.args.gpu = self.args.device_ids[0]
            else:
                torch.cuda.set_device(self.args.gpu)
        logger.info("experiment parameters: {}".format(self.args.__dict__))
        dataset = self.args.dataset
        sample_path = "./data/dataset/{}/{}.pickle".format(dataset, dataset)
        if not os.path.exists(sample_path):
            # get and save samples
            total_samples = DataPreprocessor.execute(self.args)[0]
        else:
            # load samples
            total_samples = Variabler.load_variable(sample_path)
        if self.args.do_training:
            # get train, valid and test samples
            train_samples, valid_samples, test_samples = DataPreprocessor.split_samples(
                total_samples, self.args.train_ratio, self.args.test_ratio)
            train_dataset, train_loader = BatchLoader.get_dataset(train_samples, self.args, flag="train")
            valid_dataset, valid_loader = BatchLoader.get_dataset(valid_samples, self.args, flag="valid")
            test_dataset, test_loader = BatchLoader.get_dataset(test_samples, self.args, flag="test")
            # model training
            saved_prefix = self.get_saved_prefix(self.args)
            self.train(saved_prefix, train_dataset, train_loader, valid_dataset, valid_loader, test_dataset,
                       test_loader, do_anomaly_detection=self.args.do_anomaly_detection, alarm_threshold=0.2)

    def get_saved_prefix(self, args):
        """
        get the saved prefix of results
        """
        # saved path
        saved_prefix = '{}_{}_ft{}_md{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
            args.model,
            args.dataset,
            args.forcast_task,
            args.model_des,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.n_iter)
        return saved_prefix

    def train(self, saved_path, train_dataset, train_dataloader, valid_dataset, valid_loader, test_dataset,
              test_loader, do_anomaly_detection=False, alarm_threshold=0.2):
        """
        model training & anomaly detection
        """
        pass

    def _get_device(self):
        """
        get device
        """
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device("cuda:{}".format(self.args.gpu))
            print("Use GPU: cuda:{}".format(self.args.gpu))
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device
    
    def _get_model(self):
        """
        get model
        """
        try:
            model_dict = {
                "Transformer": Transformer,
                "Informer": Informer,
                "Autoformer": Autoformer,
                "FEDformer": FEDformer,
                "ns_Transformer": ns_Transformer,
                "ns_Informer": ns_Informer,
                "ns_Autoformer": ns_Autoformer,
                "ns_FEDformer": ns_FEDformer
            }
            model = model_dict[self.args.model].Model(self.args).float()

            if self.args.use_multi_gpu and self.args.use_gpu:
                model = nn.DataParallel(model, device_ids=self.args.device_ids)
            return model
        except Exception as e:
            logger.error("model not found: {}".format(e), exc_info=True)
            return None