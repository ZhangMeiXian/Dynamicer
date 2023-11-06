# -*- coding: UTF-8 -*-
# !/usr/bin/python3.8

"""
@ModuleName: experiment executor
@author: zhangmeixian
"""
import os
import time
import torch
import random
import numpy as np
import torch.nn as nn
from torch import optim
from trainers.methods.adj_ns_transformers import adj_ns_Autoformer, adj_ns_FEDformer, adj_ns_Informer, \
    adj_ns_Transformer
from trainers.dataset_libs.data_preprocessor import DataPreprocessor
from trainers.dataset_libs.batch_loader import BatchLoader
from trainers.methods.net_libs.losses import QuantileLoss, SignMseLoss, WeightedQuantileLoss
from trainers.methods.net_libs.tools import adjust_learning_rate, EarlyStopping
from sklearn.metrics import roc_auc_score
from configs.constants import status_exception, status_exception_no
from commons.variabler import Variabler
from commons.logger import get_logger

logger = get_logger(__name__)


class ExpExecutor(object):
    """
    experiment entrance
    """

    def __init__(self, args):
        self.args = args
        # get gpu param
        self.args.use_gpu = True if torch.cuda.is_available() and self.args.use_gpu else False
        self.device = self._get_device()
        self.model = self._get_model()
        if self.model is not None:
            self.model = self.model.to(self.device)

    def execute(self):
        """
        experiment executor
        """
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
        if self.args.is_training:
            # get train, valid and test samples
            train_samples, valid_samples, test_samples = DataPreprocessor.split_samples(
                total_samples, self.args.train_ratio, self.args.test_ratio)
            train_dataset, train_loader = BatchLoader.get_dataset(train_samples, self.args, flag="train")
            valid_dataset, valid_loader = BatchLoader.get_dataset(valid_samples, self.args, flag="valid")
            test_dataset, test_loader = BatchLoader.get_dataset(test_samples, self.args, flag="test")
            for iter_index in range(self.args.itr):
                # model training
                model_prefix = self.get_model_prefix(self.args, iter_index)
                self.train(model_prefix, train_loader, valid_loader, test_loader,
                           do_anomaly_detection=self.args.do_anomaly_detection)

    def get_model_prefix(self, args, iter_index):
        """
        get the saved prefix of results
        """
        # saved path
        model_prefix = '{}_{}_ft{}_md{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
            args.model, args.dataset, args.forcast_task, args.model_des, args.seq_len, args.label_len, args.pred_len,
            args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.factor, args.embed, args.distil,
            iter_index)
        return model_prefix

    def train(self, saved_prefix, train_loader, valid_loader, test_loader, do_anomaly_detection):
        """
        model training & anomaly detection
        """
        model_saved_path = os.path.join("models", saved_prefix)
        if not os.path.exists(model_saved_path):
            os.makedirs(model_saved_path)
        time_now = time.time()
        train_steps = len(train_loader)
        criterion, criterion_des = self._get_criterion(
            self.args.loss, self.args.lambda_weight, self.args.q_value, self.args.alpha, self.args.beta)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, criterion_des=criterion_des)
        model_optim = self._select_optimizer()
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            if do_anomaly_detection:
                train_anomaly_loss = []
                train_precision, train_recall, train_roc_auc = [], [], []

            self.model.train()
            epoch_time = time.time()
            for i, batch_data in enumerate(train_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_source, batch_label, batch_day_weights, \
                batch_time_weights = batch_data
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_source = batch_source.float().to(self.device)
                batch_day_weights = batch_day_weights.float().to(self.device)
                batch_time_weights = batch_time_weights.float().to(self.device)
                f_dim = -1 if self.args.forcast_task == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if do_anomaly_detection:
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs, output_probs, _ = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark, x_source=batch_source, y_enc=batch_y,
                                    batch_day_weights=batch_day_weights, batch_time_weights=batch_time_weights)
                            else:
                                outputs, output_probs = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark, x_source=batch_source, y_enc=batch_y,
                                    batch_day_weights=batch_day_weights, batch_time_weights=batch_time_weights)
                    else:
                        if self.args.output_attention:
                            outputs, output_probs, _ = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark, x_source=batch_source, y_enc=batch_y,
                                batch_day_weights=batch_day_weights, batch_time_weights=batch_time_weights)
                        else:
                            outputs, output_probs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark, x_source=batch_source, y_enc=batch_y,
                                batch_day_weights=batch_day_weights, batch_time_weights=batch_time_weights)
                else:
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_day_weights=batch_day_weights,
                                    batch_time_weights=batch_time_weights)[0]
                            else:
                                outputs = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_day_weights=batch_day_weights,
                                    batch_time_weights=batch_time_weights)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_day_weights=batch_day_weights,
                                    batch_time_weights=batch_time_weights)[0]
                        else:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_day_weights=batch_day_weights,
                                batch_time_weights=batch_time_weights)
                # outputs = outputs[:, -self.args.pred_len:, f_dim:]

                if do_anomaly_detection:
                    batch_label = batch_label.float().to(self.device)
                    anomaly_criterion = self._BCELoss()
                    anomaly_loss = anomaly_criterion(output_probs, batch_label)
                    pred_loss = criterion(outputs, batch_y)
                    loss = self.args.anomaly_loss_weight * anomaly_loss + pred_loss
                    train_anomaly_loss.append(anomaly_loss.item())
                    train_loss.append(pred_loss.item())
                    output_labels = (output_probs >= 0.5).float()
                    cur_precision, cur_recall, cur_roc_auc = self.get_anomaly_metrics(output_labels, batch_label)
                    if cur_precision is not None:
                        train_precision.append(cur_precision)
                    if cur_recall is not None:
                        train_recall.append(cur_recall)
                    if cur_roc_auc is not None:
                        train_roc_auc.append(cur_roc_auc)
                else:
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    if do_anomaly_detection:
                        print("\titers: {0}, epoch: {1} | mse loss: {2:.7f} | bce loss: {3:.7f}".format(
                            i + 1, epoch + 1, pred_loss.item(), anomaly_loss.item()))
                    else:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss, train_anomaly_loss = np.average(train_loss), np.average(train_anomaly_loss)
            if do_anomaly_detection:
                train_precision, train_recall, train_roc_auc = np.average(train_precision), np.average(
                    train_recall), np.average(train_roc_auc)
                vali_loss, vali_anomaly_loss, vali_precision, vali_recall, vali_roc_auc = self.vali(
                    valid_loader, criterion, do_anomaly_detection)
                test_loss, test_anomaly_loss, test_precision, test_recall, test_roc_auc = self.vali(
                    test_loader, criterion, do_anomaly_detection)
            else:
                vali_loss = self.vali(valid_loader, criterion, do_anomaly_detection)
                test_loss = self.vali(test_loader, criterion, do_anomaly_detection)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            if do_anomaly_detection:
                print("Epoch: {0}, Steps: {1} | Train Anomaly Loss: {2:.7f} Vali Anomaly Loss: {3:.7f} "
                      "Test Anomaly Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_anomaly_loss, vali_anomaly_loss, test_anomaly_loss))
                print("Epoch: {0}, Steps: {1} | Train Precision: {2:.7f} Vali Precision: {3:.7f} "
                      "Test Precision: {4:.7f}".format(
                    epoch + 1, train_steps, train_precision, vali_precision, test_precision))
                print("Epoch: {0}, Steps: {1} | Train Recall: {2:.7f} Vali Recall: {3:.7f} Test Recall: {4:.7f}".
                      format(epoch + 1, train_steps, train_recall, vali_recall, test_recall))
                print("Epoch: {0}, Steps: {1} | Train ROC-AUC: {2:.7f} Vali ROC-AUC: {3:.7f} Test ROC-AUC: {4:.7f}".
                      format(epoch + 1, train_steps, train_roc_auc, vali_roc_auc, test_roc_auc))
            early_stopping(vali_loss, self.model, model_saved_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = model_saved_path + '/' + 'checkpoint_{}.pth'.format(criterion_des)
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, vali_loader, criterion, do_anomaly_detection):
        """
        validation to adjust hyper-parameters
        """
        total_loss, total_anomaly_loss = [], []
        if do_anomaly_detection:
            total_precision, total_recall, total_roc_auc = [], [], []
        self.model.eval()
        with torch.no_grad():
            for i, batch_data in enumerate(vali_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_source, batch_label, batch_day_weights, \
                batch_time_weights = batch_data
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_source = batch_source.float().to(self.device)
                batch_day_weights = batch_day_weights.float().to(self.device)
                batch_time_weights = batch_time_weights.float().to(self.device)
                f_dim = -1 if self.args.forcast_task == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if do_anomaly_detection:
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs, output_probs, _ = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark, x_source=batch_source, y_enc=batch_y,
                                    batch_day_weights=batch_day_weights, batch_time_weights=batch_time_weights)
                            else:
                                outputs, output_probs = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark, x_source=batch_source, y_enc=batch_y,
                                    batch_day_weights=batch_day_weights, batch_time_weights=batch_time_weights)
                    else:
                        if self.args.output_attention:
                            outputs, output_probs, _ = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark, x_source=batch_source, y_enc=batch_y,
                                batch_day_weights=batch_day_weights, batch_time_weights=batch_time_weights)
                        else:
                            outputs, output_probs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark, x_source=batch_source, y_enc=batch_y,
                                batch_day_weights=batch_day_weights, batch_time_weights=batch_time_weights)
                else:
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_day_weights=batch_day_weights,
                                    batch_time_weights=batch_time_weights)[0]
                            else:
                                outputs = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_day_weights=batch_day_weights,
                                    batch_time_weights=batch_time_weights)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_day_weights=batch_day_weights,
                                batch_time_weights=batch_time_weights)[0]
                        else:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_day_weights=batch_day_weights,
                                batch_time_weights=batch_time_weights)
                # outputs = outputs[:, -self.args.pred_len:, f_dim:]

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                if do_anomaly_detection:
                    batch_label = batch_label.float().to(self.device)
                    anomaly_criterion = self._BCELoss()
                    anomaly_loss = anomaly_criterion(output_probs, batch_label)
                    pred_loss = criterion(outputs, batch_y)
                    total_loss.append(pred_loss)
                    total_anomaly_loss.append(anomaly_loss)
                    output_labels = (output_probs >= 0.5).float()
                    cur_precision, cur_recall, cur_roc_auc = self.get_anomaly_metrics(output_labels, batch_label)
                    if cur_precision is not None:
                        total_precision.append(cur_precision)
                    if cur_recall is not None:
                        total_recall.append(cur_recall)
                    if cur_roc_auc is not None:
                        total_roc_auc.append(cur_roc_auc)
                else:
                    loss = criterion(pred, true)
                    total_loss.append(loss)

        total_loss = np.average(total_loss)
        if do_anomaly_detection:
            total_precision = np.average(total_precision)
            total_recall = np.average(total_recall)
            total_roc_auc = np.average(total_roc_auc)
            total_anomaly_loss = np.average(total_anomaly_loss)
        self.model.train()
        return total_loss, total_anomaly_loss, total_precision, total_recall, total_roc_auc \
            if do_anomaly_detection else total_loss

    def get_anomaly_metrics(self, pred_labels, true_labels):
        """
        get anomaly metrics in view of exception point
        """
        pred_labels = pred_labels.to("cpu").squeeze().tolist()
        true_labels = true_labels.to("cpu").squeeze().tolist()
        total_count = len(pred_labels)
        total_exceptions, ex_to_normal, normal_to_ex = 0, 0, 0
        for pred_label, true_label in zip(pred_labels, true_labels):
            if true_label == status_exception:
                total_exceptions += 1
            if true_label == status_exception and pred_label == status_exception_no:
                ex_to_normal += 1
            if true_label == status_exception_no and pred_label == status_exception:
                normal_to_ex += 1
        ex_to_ex = total_exceptions - ex_to_normal
        total_normals = total_count - total_exceptions
        ex_precision = ex_to_ex / (ex_to_ex + normal_to_ex) if (ex_to_ex + normal_to_ex) != 0 else None
        ex_recall = ex_to_ex / total_exceptions if total_exceptions != 0 else None
        roc_auc = roc_auc_score(true_labels, pred_labels) if total_exceptions != 0 and total_normals != 0 else None
        return ex_precision, ex_recall, roc_auc

    def _select_optimizer(self):
        """
        获取优化器
        """
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _get_criterion(self, loss_name, lambda_weight, q_value, alpha, beta):
        """
        get description of loss function
        """
        if loss_name == "quantile_loss":
            return QuantileLoss(q=q_value), "ql_q{}".format(str(q_value))
        if loss_name == "weighted_quantile_loss":
            return WeightedQuantileLoss(q=q_value, alpha=alpha, beta=beta), "wql_q{}_a{}_b{}".format(str(q_value),
                                                                                                     str(alpha),
                                                                                                     str(beta))
        if loss_name == "sign_mse":
            return SignMseLoss(lambda_weight=lambda_weight), "smse_lamb{}".format(str(lambda_weight))

        return nn.MSELoss(), "mse"

    def _BCELoss(self):
        """
        cross entropy loss
        """
        loss = nn.BCELoss()
        return loss

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
                "adj_ns_Transformer": adj_ns_Transformer,
                "adj_ns_Informer": adj_ns_Informer,
                "adj_ns_Autoformer": adj_ns_Autoformer,
                "adj_ns_FEDformer": adj_ns_FEDformer
            }
            model = model_dict[self.args.model].Model(self.args).float()

            if self.args.use_multi_gpu and self.args.use_gpu:
                model = nn.DataParallel(model, device_ids=self.args.device_ids)
            return model
        except Exception as e:
            logger.error("model not found: {}".format(e), exc_info=True)
            return None
