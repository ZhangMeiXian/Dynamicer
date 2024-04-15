# -*- coding: UTF-8 -*-
# !/usr/bin/python3.8

"""
@ModuleName: entrence of experiment executor
@author: zhangmeixian
"""
import os
import time
import torch
import random
import numpy as np
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torch import optim
from sklearn.preprocessing import StandardScaler
from trainers.methods.adj_ns_transformers import adj_ns_Autoformer, adj_ns_FEDformer, adj_ns_Informer, \
    adj_ns_Transformer, adj_ns_DLinear
from trainers.methods.ts_dynamicer import tsDynamicer
from trainers.dataset_libs.data_preprocessor import DataPreprocessor
from trainers.dataset_libs.batch_loader import BatchLoader
from trainers.methods.net_libs.losses import QuantileLoss, SignMseLoss, WeightedQuantileLoss
from trainers.methods.net_libs.tools import adjust_learning_rate, EarlyStopping
from trainers.methods.ts_dynamicer.anomaly_detector import DynamicAnomalyDetector, AnomalyDetector
from sklearn.metrics import roc_auc_score
from trainers.methods.net_libs.metrics import metric
from configs.constants import status_exception, status_exception_no, column_label, column_data, column_time
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
        self.detect_model = self._get_detect_model()
        if self.detect_model is not None:
            self.detect_model = self.detect_model.to(self.device)
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
        # get train, valid and test samples
        train_samples, valid_samples, test_samples, showed_samples = DataPreprocessor.grousampling(
            total_samples, self.args.train_ratio, self.args.test_ratio, self.args.dataset)
        train_dataset, train_loader = BatchLoader.get_dataset(train_samples, self.args, flag="train")
        valid_dataset, valid_loader = BatchLoader.get_dataset(valid_samples, self.args, flag="valid")
        test_dataset, test_loader = BatchLoader.get_dataset(test_samples, self.args, flag="test")
        if dataset in ["WADI"]:
            showed_dataset, showed_loader = test_dataset, test_loader
        else:
            showed_dataset, showed_loader = BatchLoader.get_dataset(showed_samples, self.args, flag="test")
        if self.args.is_training:
            for iter_index in range(self.args.itr):
                # model training
                model_prefix = self.get_model_prefix(self.args, iter_index)
                self.train(model_prefix, train_loader, valid_loader, test_loader, showed_loader)

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

    def train(self, saved_prefix, train_loader, valid_loader, test_loader, showed_loader):
        """
        model training & anomaly detection
        """
        model_saved_path = os.path.join("models", saved_prefix)
        if not os.path.exists(model_saved_path):
            os.makedirs(model_saved_path)
        train_steps = len(train_loader)
        criterion, criterion_des = self._get_criterion(
            self.args.loss, self.args.lambda_weight, self.args.q_value, self.args.alpha, self.args.beta)
        # train the forecasting model
        self.forecast_train(criterion, criterion_des, model_saved_path, saved_prefix, train_steps, train_loader,
                            valid_loader, test_loader)
        # train the anomaly detection model
        self.detect_train(criterion, criterion_des, model_saved_path, saved_prefix, train_steps, train_loader,
                          valid_loader, test_loader)
        # test evaluation
        self.anomaly_detect(saved_prefix, showed_loader, criterion, criterion_des, "END", "FINAL", is_saved=True,
                            model_type="FINAL")
        # return the models
        best_model_path = model_saved_path + '/' + '{}_checkpoint_{}.pth'.format("forecast", criterion_des)
        self.model.load_state_dict(torch.load(best_model_path))
        best_detect_model = model_saved_path + "/" + '{}_checkpoint_{}.pth'.format("detect", criterion_des)
        self.detect_model.load_state_dict(torch.load(best_detect_model))

        return self.model, self.detect_model

    def forecast_train(self, criterion, criterion_des, model_saved_path, saved_prefix, train_steps, train_loader,
                       valid_loader, test_loader):
        """
        train forecast model
        """
        time_now = time.time()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, criterion_des=criterion_des)
        model_optim = self._select_optimizer()
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        for epoch in range(self.args.train_epochs):
            iter_count, train_loss = 0, []
            self.model.train()
            epoch_time = time.time()
            for i, batch_data in enumerate(train_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_source, batch_label, batch_day_weights, \
                batch_time_weights, batch_unique_keys = batch_data
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_source = batch_source.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_day_weights = batch_day_weights.float().to(self.device)
                batch_time_weights = batch_time_weights.float().to(self.device)
                f_dim = -1 if self.args.forcast_task == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
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
                loss = criterion(outputs, batch_y)
                # loss = self.get_forecast_loss(criterion, outputs, batch_y, batch_label, batch_source)
                train_loss.append(loss.item())
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                if self.args.is_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                if (i + 1) % 100 == 0:
                    print("\t[forecast training] iters: {0}, epoch: {1} | total loss: {2:.7f}".format(
                        i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\t[forecast training] speed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            print("[forecast train to validate] Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, _ = self.vali(valid_loader, criterion)
            test_loss, _ = self.vali(test_loader, criterion)
            self.anomaly_detect(
                saved_prefix, test_loader, criterion, criterion_des, epoch + 1, train_steps, is_saved=False)
            print("[forecast val] Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} "
                  "Test Loss: {4:.7f}".format(epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, model_saved_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

    def detect_train(self, criterion, criterion_des, model_saved_path, saved_prefix, train_steps, train_loader,
                     valid_loader, test_loader):
        """
        train detect model
        """
        time_now = time.time()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, criterion_des=criterion_des)
        detect_model_optim = self._select_detect_optimizer()
        for epoch in range(self.args.train_epochs):
            iter_count, train_detect_loss, pred_loss = 0, [], []
            self.model.eval()
            self.detect_model.train()
            epoch_time = time.time()
            for i, batch_data in enumerate(train_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_source, batch_label, batch_day_weights, \
                batch_time_weights, batch_unique_keys = batch_data
                iter_count += 1
                detect_model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_source = batch_source.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_day_weights = batch_day_weights.float().to(self.device)
                batch_time_weights = batch_time_weights.float().to(self.device)
                f_dim = -1 if self.args.forcast_task == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
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
                loss = criterion(outputs, batch_y)
                # loss = self.get_forecast_loss(criterion, outputs, batch_y, batch_label, batch_source)
                pred_loss.append(loss.item())
                if self.args.is_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                pred_probs = self.detect_model(
                    batch_source.float().to(self.device), outputs.detach().float().to(self.device))
                class_weights = torch.ones(batch_label.size()).to(self.device)
                class_weights[batch_label == status_exception] *= self.args.anomaly_class_weight
                detection_criterion = self._BCELoss(class_weight=class_weights)
                detect_loss = detection_criterion(
                    pred_probs.float().to(self.device), batch_label.float().to(self.device))
                train_detect_loss.append(detect_loss.item())
                detect_loss.backward()
                detect_model_optim.step()
                if (i + 1) % 100 == 0:
                    print("\t[detect training] iters: {0}, epoch: {1} | total pred loss: {2:.7f} | "
                          "total anomaly loss: {3:.7f}".format(i + 1, epoch + 1, loss.item(), detect_loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\t[detect training] speed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            print("[detect train to validate] Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            pred_loss, train_detect_loss = np.average(pred_loss), np.average(train_detect_loss)
            vali_pred_loss, vali_detect_loss = self.vali(valid_loader, criterion)
            test_pred_loss, test_detect_loss = self.vali(test_loader, criterion)
            self.anomaly_detect(
                saved_prefix, test_loader, criterion, criterion_des, epoch + 1, train_steps, is_saved=False,
                model_type="detect")

            print("[detect val] Epoch: {0}, Steps: {1} | Train Pred Loss: {2:.7f} Vali Pred Loss: {3:.7f} "
                  "Test Pred Loss: {4:.7f} | Train Detect Loss: {5:.7f} Vali Detect Loss: {6:.7f} "
                  "Test Detect Loss: {7:.7f}".format(
                epoch + 1, train_steps, pred_loss, vali_pred_loss, test_pred_loss, train_detect_loss, vali_detect_loss,
                test_detect_loss))
            early_stopping(vali_detect_loss, self.detect_model, model_saved_path, model_type="detect")
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(detect_model_optim, epoch + 1, self.args)

    def anomaly_detect(self, saved_prefix, test_loader, criterion, criterion_des, epoch_num, iter_num, is_saved=True,
                       threshold=0.5, model_type="forecast"):
        """
        detection anomaly by threshold
        """
        results_path = os.path.join("results", saved_prefix)
        true_values, pred_values, batch_sources, batch_labels, batch_unique_keys = [], [], [], [], []
        pred_labels, pred_probs = [], []
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        self.model.eval()
        self.detect_model.eval()
        with torch.no_grad():
            for i, batch_data in enumerate(test_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_source, batch_label, batch_day_weights, \
                batch_time_weights, batch_unique_key = batch_data
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_source = batch_source.float().to(self.device)
                batch_label = batch_label.float().to(self.device)
                batch_labels.append(batch_label)
                batch_sources.append(batch_source)
                batch_unique_keys.append(list(batch_unique_key[0]))
                batch_day_weights = batch_day_weights.float().to(self.device)
                batch_time_weights = batch_time_weights.float().to(self.device)
                f_dim = -1 if self.args.forcast_task == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark, x_source=batch_source, y_enc=batch_y,
                                batch_day_weights=batch_day_weights, batch_time_weights=batch_time_weights)[0]
                        else:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark, x_source=batch_source, y_enc=batch_y,
                                batch_day_weights=batch_day_weights, batch_time_weights=batch_time_weights)
                else:
                    if self.args.output_attention:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark, x_source=batch_source, y_enc=batch_y,
                            batch_day_weights=batch_day_weights, batch_time_weights=batch_time_weights)[0]
                    else:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark, x_source=batch_source, y_enc=batch_y,
                            batch_day_weights=batch_day_weights, batch_time_weights=batch_time_weights)
                # anomaly detection
                pred_prob = self.detect_model(batch_source.float().to(self.device), outputs.detach())
                pred_prob = pred_prob.detach().cpu()
                pred_probs.append(pred_prob)
                pred_label = (pred_prob > threshold).float()
                pred_labels.append(pred_label)
                # outputs = outputs[:, -self.args.pred_len:, f_dim:]
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                pred_values.append(pred)
                true_values.append(true)
        batch_labels = torch.cat(batch_labels, dim=0).detach().cpu()
        pred_probs = torch.cat(pred_probs, dim=0).detach().cpu()
        pred_labels = torch.cat(pred_labels, dim=0).detach().cpu()
        pred_values = torch.stack(pred_values, dim=0).detach().cpu().squeeze(dim=2)
        true_values = torch.stack(true_values, dim=0).detach().cpu().squeeze(dim=2)
        batch_sources = torch.stack(batch_sources, dim=0).detach().cpu().squeeze(dim=1)
        test_loss = criterion(pred_values.float().to(self.device), true_values.float().to(self.device))
        # test_loss = self.get_forecast_loss(criterion, pred_values, true_values, batch_labels, batch_sources)
        class_weights = torch.ones_like(batch_labels).float()
        class_weights[batch_labels == status_exception] *= self.args.anomaly_class_weight
        detection_criterion = self._BCELoss(class_weight=class_weights)
        detect_loss = detection_criterion(pred_probs, batch_labels)
        print("[{} test] Epoch: {}, Steps: {} | Test Loss: {} | Test Detect Loss: {} ".format(
            model_type, epoch_num, iter_num, test_loss.item(), detect_loss.item()))
        mae, mse, rmse, mape, mspe, qe = metric(pred_values.numpy(), true_values.numpy())
        print("[test] MAE: {}, MSE: {}, MAPE: {}, MSPE: {}, QE: {}".format(mae, mse, rmse, mape, mspe, qe))
        test_precision, test_recall, test_roc_auc = self.get_anomaly_metrics(pred_labels, batch_labels)
        print("[{} test] Epoch: {}, Steps: {} | Test Precision: {} Test Recall: {} Test ROC-AUC: {}".format(
            model_type, epoch_num, iter_num, test_precision, test_recall, test_roc_auc))

        if is_saved:
            # save pics (anomaly points)
            self.save_pics(saved_prefix, batch_sources, pred_values, batch_unique_keys, batch_labels)
            # save results
            save_prefix = "/".join([results_path, self.args.dataset])
            if not os.path.exists(save_prefix):
                os.makedirs(save_prefix)
            np.save(save_prefix + "/sources.npy", batch_sources)
            np.save(save_prefix + "/preds.npy", pred_values)
            np.save(save_prefix + "/trues.npy", true_values)
            np.save(save_prefix + "/unique_keys.npy", batch_unique_keys)
            print("success to save pics and data.")

    def anomaly_detect_by_rule(self, saved_prefix, test_loader, criterion, criterion_des, epoch_num, iter_num,
                               is_saved=True):
        """
        detection anomaly by threshold
        """
        results_path = os.path.join("results", saved_prefix)
        true_values, pred_values, batch_sources, batch_labels, batch_unique_keys = [], [], [], [], []
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        self.model.eval()
        with torch.no_grad():
            for i, batch_data in enumerate(test_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_source, batch_label, batch_day_weights, \
                batch_time_weights, batch_unique_key = batch_data
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_source = batch_source.float().to(self.device)
                batch_label = batch_label.float().to(self.device)
                batch_labels.append(batch_label)
                batch_sources.append(batch_source)
                batch_unique_keys.append(list(batch_unique_key[0]))
                batch_day_weights = batch_day_weights.float().to(self.device)
                batch_time_weights = batch_time_weights.float().to(self.device)
                f_dim = -1 if self.args.forcast_task == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark, x_source=batch_source, y_enc=batch_y,
                                batch_day_weights=batch_day_weights, batch_time_weights=batch_time_weights)[0]
                        else:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark, x_source=batch_source, y_enc=batch_y,
                                batch_day_weights=batch_day_weights, batch_time_weights=batch_time_weights)
                else:
                    if self.args.output_attention:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark, x_source=batch_source, y_enc=batch_y,
                            batch_day_weights=batch_day_weights, batch_time_weights=batch_time_weights)[0]
                    else:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark, x_source=batch_source, y_enc=batch_y,
                            batch_day_weights=batch_day_weights, batch_time_weights=batch_time_weights)
                # outputs = outputs[:, -self.args.pred_len:, f_dim:]
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                pred_values.append(pred)
                true_values.append(true)
        batch_labels = torch.cat(batch_labels, dim=0).detach().cpu()
        pred_values = torch.stack(pred_values, dim=0).detach().cpu().squeeze(dim=2)
        true_values = torch.stack(true_values, dim=0).detach().cpu().squeeze(dim=2)
        print(pred_values.shape)
        batch_sources = torch.stack(batch_sources, dim=0).detach().cpu().squeeze(dim=1)
        test_loss = criterion(pred_values.float().to(self.device), true_values.float().to(self.device))
        print("[test] Epoch: {}, Steps: {} | Test Loss: {} ".format(epoch_num, iter_num, test_loss.item()))
        mae, mse, rmse, mape, mspe, qe = metric(pred_values.numpy(), true_values.numpy())
        print("[test] MAE: {}, MSE: {}, MAPE: {}, MSPE: {}, QE: {}".format(mae, mse, rmse, mape, mspe, qe))
        anomaly_detector = DynamicAnomalyDetector(self.args)
        output_labels = anomaly_detector(batch_sources.float().to(self.device), pred_values.float().to(self.device))
        test_precision, test_recall, test_roc_auc = self.get_anomaly_metrics(output_labels, batch_labels)
        print("[test] Epoch: {}, Steps: {} | Test Precision: {} Test Recall: {} Test ROC-AUC: {}".format(
            epoch_num, iter_num, test_precision, test_recall, test_roc_auc))

        if is_saved:
            # save pics (anomaly points)
            self.save_pics(saved_prefix, batch_sources, pred_values, batch_unique_keys, batch_labels)
            # save results
            save_prefix = "/".join([results_path, self.args.dataset])
            if not os.path.exists(save_prefix):
                os.makedirs(save_prefix)
            np.save(save_prefix + "/sources.npy", batch_sources)
            np.save(save_prefix + "/preds.npy", pred_values)
            np.save(save_prefix + "/trues.npy", true_values)
            np.save(save_prefix + "/unique_keys.npy", batch_unique_keys)
            print("success to save pics and data.")

    def save_pics(self, saved_prefix, batch_sources, pred_values, batch_unique_keys, batch_labels, plot_limit_count=10):
        """
        plot anomaly points and save
        """
        pic_path = os.path.join("pics", saved_prefix)
        batch_sources = batch_sources.squeeze().tolist()
        pred_values = pred_values.squeeze().tolist()
        batch_labels = batch_labels.squeeze().tolist()
        index_unique_keys, time_list = self.get_total_unique_index_keys(batch_unique_keys)
        sample_df = pd.DataFrame()
        sample_df["index_unique_key"] = index_unique_keys
        sample_df[column_label] = batch_labels
        sample_df[column_time] = time_list
        sample_df[column_data] = pred_values
        sample_df["data_source"] = batch_sources
        grouped_dfs = sample_df.groupby("index_unique_key")
        plot_count = 0
        for index_unique_key, sub_df in grouped_dfs:
            plot_count += 1
            index_unique_key = index_unique_key.replace("/", "-") if "/" in index_unique_key else index_unique_key
            sub_df["recovered_data"] = sub_df.apply(lambda x: self.recover_data(x.data_source, x.data), axis=1)
            sub_df["origin_data"] = sub_df["data_source"].apply(lambda x: x[-1])
            sub_df.sort_values(by=column_time, inplace=True)
            sub_df = sub_df if len(sub_df) <= 200 else sub_df[:200]
            labels = sub_df[column_label].tolist()
            true_datas = sub_df["origin_data"].tolist()
            pred_datas = sub_df["recovered_data"].tolist()
            time_list = sub_df[column_time].tolist()
            plot_index = range(len(time_list))
            pic_sub_path = pic_path + "/{}".format(index_unique_key)
            if not os.path.exists(pic_sub_path):
                os.makedirs(pic_sub_path)
            fig, axes = plt.subplots(self.args.n_index, 1, figsize=(12, 5 * self.args.n_index))
            start_index, end_index = 0, 500
            while end_index <= len(sub_df) + 500 or start_index == 0:
                if end_index > len(sub_df):
                    end_index = len(sub_df)
                if start_index >= end_index:
                    break
                for i in range(self.args.n_index):
                    cur_labels = labels[start_index: end_index]
                    pred_data = [x[i] for x in pred_datas] if self.args.n_index > 1 else pred_datas
                    true_data = [x[i] for x in true_datas] if self.args.n_index > 1 else true_datas
                    # pred_colors = ['blue' if label == status_exception_no else 'green' for label in labels]
                    # true_colors = ['yellow' if label == status_exception_no else 'red' for label in labels]
                    plot_pred_data = pred_data[start_index: end_index]
                    plot_true_data = true_data[start_index: end_index]
                    cur_plot_index = plot_index[start_index: end_index]
                    plot_anomaly_index = [cur_plot_index[i] for i in range(len(cur_labels)) if
                                          cur_labels[i] == status_exception]
                    anomaly_true_points = [true_data[index] for index in plot_anomaly_index]
                    anomaly_pred_data = [pred_data[index] for index in plot_anomaly_index]
                    plot_time_list = time_list[start_index: end_index]
                    axe_obj = axes[i] if self.args.n_index > 1 else axes
                    axe_obj.plot(cur_plot_index, plot_pred_data, label="pred_data")
                    axe_obj.plot(cur_plot_index, plot_true_data, label="true_data")
                    axe_obj.scatter(plot_anomaly_index, anomaly_true_points, color='red', label="true_anomaly_data")
                    axe_obj.scatter(plot_anomaly_index, anomaly_pred_data, color='green', label="pred_theoretical_data")
                    if len(plot_time_list) <= 1:
                        continue
                    plot_gap = len(plot_time_list) // 40
                    if plot_gap == 0:
                        plot_gap = 2
                    # print(plot_gap, len(plot_time_list), start_index, end_index)
                    axe_obj.set_xticks(cur_plot_index[::len(plot_time_list) // plot_gap])
                    axe_obj.set_xticklabels(plot_time_list[::len(plot_time_list) // plot_gap], rotation=10, ha='right')
                    axe_obj.legend()
                file_name = pic_sub_path + "/test_results_{}_{}.png".format(start_index, end_index)
                plt.savefig(file_name)
                # print("success to save: {}".format(file_name))
                plt.clf()
                start_index = end_index
                end_index += 500
                if plot_count >= plot_limit_count:
                    break
        plt.close()

    def recover_data(self, data_source, pred_data):
        """
        get recovered data
        """
        scalar_index = (self.args.sample_time_window_before + 1 + self.args.sample_time_window_after) * \
                       self.args.sample_day_window \
            if self.args.sample_day_window > 0 else int(self.args.sample_time_window_before * 2 / 3)
        origin_history_target = data_source[:scalar_index]
        origin_history_target = np.array(origin_history_target).reshape(-1, 1)
        # print(origin_history_target.shape)
        scaler = StandardScaler()
        scaler.fit(origin_history_target)
        pred_data = np.array(pred_data).reshape(1, -1)
        # print(pred_data.shape)
        recovered_data = scaler.inverse_transform(pred_data).tolist()[0]
        return recovered_data

    def get_total_unique_index_keys(self, batch_unique_keys):
        """
        get index unique keys
        """
        index_unique_keys, time_list = [], []
        for batch_unique_key in batch_unique_keys:
            index_unique_key = "_".join(batch_unique_key[0].split("_")[:-1])
            time_list.append(batch_unique_key[0].split("_")[-1])
            index_unique_keys.append(index_unique_key)
        return index_unique_keys, time_list

    def detect(self, saved_prefix, test_loader, criterion, criterion_des, is_saved=True):
        """
        anomaly detection on test data set
        """
        true_labels, pred_labels, pred_probs = [], [], []
        true_values, pred_values = [], []
        batch_sources, batch_unique_keys = [], []
        results_path = os.path.join("results", saved_prefix)
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        self.model.eval()
        with torch.no_grad():
            for i, batch_data in enumerate(test_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_source, batch_label, batch_day_weights, \
                batch_time_weights, batch_unique_key = batch_data
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_source = batch_source.float().to(self.device)
                batch_sources.append(batch_source)
                batch_unique_keys.append(batch_unique_key)
                batch_day_weights = batch_day_weights.float().to(self.device)
                batch_time_weights = batch_time_weights.float().to(self.device)
                f_dim = -1 if self.args.forcast_task == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
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
                # outputs = outputs[:, -self.args.pred_len:, f_dim:]
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                batch_label = batch_label.float().to(self.device)
                batch_label = batch_label.detach().cpu()
                output_label = (output_probs >= 0.5).float()
                output_label = output_label.detach().cpu()
                output_probs = output_probs.detach().cpu()
                pred_values.append(pred)
                true_values.append(true)
                pred_labels.append(output_label)
                pred_probs.append(output_probs)
                true_labels.append(batch_label)
        pred_labels = torch.cat(pred_labels, dim=0).detach().cpu()
        true_labels = torch.cat(true_labels, dim=0).detach().cpu()
        pred_probs = torch.cat(pred_probs, dim=0).detach().cpu()
        pred_values = torch.stack(pred_values, dim=0).detach().cpu()
        true_values = torch.stack(true_values, dim=0).detach().cpu()
        test_precision, test_recall, test_roc_auc = self.get_anomaly_metrics(pred_labels, true_labels)
        class_weight = torch.ones_like(true_labels).float()
        class_weight[true_labels == status_exception] *= self.args.anomaly_class_weight
        anomaly_criterion = self._BCELoss(class_weight)
        test_loss = criterion(pred_values, true_values)
        test_anomaly_loss = anomaly_criterion(pred_probs, pred_labels)
        print("[END] Test Loss: {} ".format(1 / 2 * (test_loss.item() + test_anomaly_loss.item())))
        print("[END] Test Precision: {} Test Recall: {} Test ROC-AUC: {}".format(
            test_precision, test_recall, test_roc_auc))

        if is_saved:
            np.save(results_path + "sources_{}.npy".format(criterion_des),
                    torch.stack(batch_sources, dim=0).detach().cpu())
            np.save(results_path + "preds_{}.npy".format(criterion_des), pred_values)
            np.save(results_path + "trues_{}.npy".format(criterion_des), true_values)

    def vali(self, vali_loader, criterion):
        """
        validation to adjust hyper-parameters
        """
        total_loss, total_detect_loss = [], []
        self.model.eval()
        self.detect_model.eval()
        with torch.no_grad():
            for i, batch_data in enumerate(vali_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_source, batch_label, batch_day_weights, \
                batch_time_weights, batch_unique_keys = batch_data
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_day_weights = batch_day_weights.float().to(self.device)
                batch_time_weights = batch_time_weights.float().to(self.device)
                f_dim = -1 if self.args.forcast_task == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
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
                # anomaly detection
                pred_probs = self.detect_model(batch_source.float().to(self.device), outputs.detach())
                pred_probs = pred_probs.detach()
                class_weights = torch.ones_like(batch_label).float().to(self.device)
                class_weights[batch_label == status_exception] *= self.args.anomaly_class_weight
                detection_criterion = self._BCELoss(class_weight=class_weights)
                detect_loss = detection_criterion(pred_probs, batch_label.float().to(self.device))
                total_detect_loss.append(detect_loss.item())

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        total_detect_loss = np.average(total_detect_loss)
        self.model.train()
        self.detect_model.train()
        return total_loss, total_detect_loss

    def get_anomaly_metrics(self, pred_labels, true_labels):
        """
        get anomaly metrics in view of exception point
        """
        pred_labels = pred_labels.to("cpu").squeeze().tolist() if isinstance(pred_labels, torch.Tensor) else pred_labels
        true_labels = true_labels.to("cpu").squeeze().tolist() if isinstance(true_labels, torch.Tensor) else true_labels
        total_count = len(pred_labels)
        total_exceptions, ex_to_normal, normal_to_ex, ex_to_ex = 0, 0, 0, 0
        for pred_label, true_label in zip(pred_labels, true_labels):
            if true_label == status_exception:
                total_exceptions += 1
            if true_label == status_exception and pred_label == status_exception:
                ex_to_ex += 1
            if true_label == status_exception and pred_label == status_exception_no:
                ex_to_normal += 1
            if true_label == status_exception_no and pred_label == status_exception:
                normal_to_ex += 1
        total_normals = total_count - total_exceptions
        ex_precision = ex_to_ex / (ex_to_ex + normal_to_ex) if (ex_to_ex + normal_to_ex) != 0 else None
        ex_recall = ex_to_ex / total_exceptions if total_exceptions != 0 else None
        roc_auc = roc_auc_score(true_labels, pred_labels) if total_exceptions != 0 and total_normals != 0 else None
        print("precision: {}/{}={}, recall: {}/{}={}".format(ex_to_ex, ex_to_ex + normal_to_ex, ex_precision, ex_to_ex,
                                                             total_exceptions, ex_recall))
        return ex_precision, ex_recall, roc_auc

    def _select_optimizer(self):
        """
        get optimizer
        """
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_detect_optimizer(self):
        """
        get detect optimizer
        """
        model_optim = optim.Adam(self.detect_model.parameters(), lr=self.args.learning_rate)
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

    def get_forecast_loss(self, criterion, outputs, bench_y, true_labels, batch_source):
        """
        get adjusted forecast loss
        """
        # get normalized true data
        true_y = batch_source[:, -self.args.pred_len:, :]
        history_index = self.args.sample_time_window_before \
            if self.args.sample_day_window > 0 else self.args.sample_time_window_before // 3
        history_source = batch_source[:, :-(history_index), :]
        mean = torch.mean(history_source, dim=(0, 1), keepdim=True)
        std = torch.std(history_source, dim=(0, 1), keepdim=True)
        true_y = (true_y - mean) / std
        # get classified data
        pos_outputs = outputs[true_labels == status_exception_no]
        neg_outputs = outputs[true_labels == status_exception]
        pos_trues = true_y[true_labels == status_exception_no]
        neg_trues = true_y[true_labels == status_exception]
        neg_benchs = bench_y[true_labels == status_exception]
        # pos loss
        pos_loss = criterion(pos_outputs, pos_trues)
        # neg loss
        neg_true_loss = criterion(neg_outputs, neg_trues)
        neg_bench_loss = criterion(neg_outputs, neg_benchs)
        neg_loss = torch.where(neg_true_loss == 0, neg_bench_loss, neg_bench_loss / neg_true_loss)
        if torch.isnan(neg_loss):
            return pos_loss
        return pos_loss + neg_loss

    def _BCELoss(self, class_weight=None):
        """
        cross entropy loss
        """
        loss = nn.BCELoss(weight=class_weight)
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

    def _get_detect_model(self):
        """
        get anomaly detection model
        """
        try:
            detect_model = AnomalyDetector(self.args).float()
            # for param in detect_model.parameters():
            #     if param.requires_grad and len(param.size()) >= 2:
            #         nn.init.kaiming_uniform_(param, a=0, mode='fan_in', nonlinearity='leaky_relu')
            return detect_model
        except Exception as e:
            return None

    def _get_model(self):
        """
        get model
        """
        try:
            model_dict = {
                "tsDynamicer": tsDynamicer,
                "adj_ns_DLinear": adj_ns_DLinear,
                "adj_ns_Transformer": adj_ns_Transformer,
                "adj_ns_Informer": adj_ns_Informer,
                "adj_ns_Autoformer": adj_ns_Autoformer,
                "adj_ns_FEDformer": adj_ns_FEDformer
            }
            model = model_dict[self.args.model].Model(self.args).float()

            if self.args.use_multi_gpu and self.args.use_gpu:
                model = nn.DataParallel(model, device_ids=self.args.device_ids)

            # # init model weight
            # for param in model.parameters():
            #     if param.requires_grad and len(param.size()) >= 2:
            #         nn.init.kaiming_uniform_(param, a=0, mode='fan_in', nonlinearity='leaky_relu')
            return model
        except Exception as e:
            logger.error("model not found: {}".format(e), exc_info=True)
            return None
