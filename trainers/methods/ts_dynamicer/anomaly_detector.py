# coding=utf-8
# author=zhangmeixian
"""
anomaly detection module details
"""

import torch
import torch.nn as nn
from configs.constants import status_up, status_down, status_both

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class DiffBlock(nn.Module):
    """
    block to process diff
    """

    def __init__(self, in_channels, dropout):
        super(DiffBlock, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        forward cal
        """
        x = self.conv_layers(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class TrendBlock(nn.Module):
    """
    trend feature extract block
    """

    def __init__(self, in_channels, dropout):
        super(TrendBlock, self).__init__()
        self.avg1 = moving_avg(kernel_size=5, stride=1)
        self.avg2 = moving_avg(kernel_size=10, stride=1)
        self.avg3 = moving_avg(kernel_size=20, stride=1)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=64, out_channels=16, kernel_size=1)
        )

    def padding(self, x, batch_size, seq_len, dimension):
        """
        padding
        """
        padding_tensor = torch.zeros(batch_size, seq_len - x.shape[1], dimension)
        padding_tensor = padding_tensor.to(device)
        return torch.cat((x, padding_tensor), dim=1)

    def forward(self, x):
        """
        forward cal
        """
        x = x.permute(0, 2, 1)
        B, L, D = x.shape
        avg_x1 = self.padding(self.avg1(x), B, L, D)
        avg_x2 = self.padding(self.avg2(x), B, L, D)
        avg_x3 = self.padding(self.avg3(x), B, L, D)
        x = x + avg_x1 + avg_x2 + avg_x3
        # x = x.permute(0, 2, 1)
        x = self.conv_layers(x)
        return x.permute(0, 2, 1)


class DynamicAnomalyDetector(nn.Module):
    """
    dynamic anomaly detector
    """

    def __init__(self, args):
        super(DynamicAnomalyDetector, self).__init__()
        self.sample_time_window_before = args.sample_time_window_before
        self.sample_time_window_after = args.sample_time_window_after
        self.sample_day_window = args.sample_day_window
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.n_index = args.n_index
        self.moving_avg = args.moving_avg
        self.anomaly_state = args.anomaly_state
        self.anomaly_ratio = args.anomaly_ratio
        self.index_ratio = args.index_ratio
        # smooth layer
        self.mv_avg = moving_avg(kernel_size=self.moving_avg, stride=1)

    def forward(self, source_array, outputs_y):
        """
        anomaly detection by threshold
        """
        # get history array
        history_index = self.sample_time_window_before \
            if self.sample_day_window > 0 else self.sample_time_window_before // 3
        history_array = source_array[:, :-(history_index), :]
        mean = torch.mean(history_array, dim=(0, 1), keepdim=True)
        std = torch.std(history_array, dim=(0, 1), keepdim=True)
        # get de-normalized outputs_y, batch_y
        true_y = source_array[:, -self.pred_len:, :]
        true_y = (true_y - mean) / std
        # de_outputs_y = outputs_y * std + mean
        # get concat data
        concat_before_data = source_array[:, :-self.pred_len, :]
        concat_before_data = (concat_before_data - mean) / std
        concat_outputs = torch.cat([concat_before_data, outputs_y], dim=1)
        concat_batch = torch.cat([concat_before_data, true_y], dim=1)
        # smooth concat data
        concat_outputs = concat_outputs.permute(0, 2, 1)
        concat_batch = concat_batch.permute(0, 2, 1)
        smoothed_outputs = self.mv_avg(concat_outputs)
        smoothed_batch = self.mv_avg(concat_batch)
        # get current smoothed data
        output_res = smoothed_outputs[:, :, -self.pred_len:]
        batch_res = smoothed_batch[:, :, -self.pred_len:]
        # get diff of pred data and real data
        res_diff = torch.where(output_res != 0, (batch_res - output_res) / output_res, batch_res)
        if self.anomaly_state == status_up:
            features_anomalies = (res_diff >= self.anomaly_ratio).sum(dim=1)
            anomalies = (features_anomalies >= self.index_ratio * self.n_index).float()
            return anomalies
        if self.anomaly_state == status_down:
            features_anomalies = (res_diff <= -self.anomaly_ratio).sum(dim=1)
            anomalies = (features_anomalies >= self.index_ratio * self.n_index).float()
            return anomalies
        res_diff = torch.abs(res_diff)
        features_anomalies = (res_diff >= self.anomaly_ratio).sum(dim=1)
        anomalies = (features_anomalies >= self.index_ratio * self.n_index).float()
        # print(res_diff.shape, features_anomalies.shape, anomalies.shape)
        return anomalies


class AnomalyDetector(nn.Module):
    """
    anomaly detector net
    """

    def __init__(self, args, n_class=1):
        super(AnomalyDetector, self).__init__()
        self.sample_time_window_before = args.sample_time_window_before
        self.sample_time_window_after = args.sample_time_window_after
        self.sample_day_window = args.sample_day_window
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.n_index = args.n_index
        self.dropout = args.dropout
        self.moving_avg = args.moving_avg
        # smooth layer
        self.mv_avg = moving_avg(kernel_size=self.moving_avg, stride=1)
        # diff block
        self.diff_block = DiffBlock(args.pred_len, self.dropout)
        # trend_block
        self.trend_block = TrendBlock(args.seq_len + args.pred_len, self.dropout)
        # self.trend_weights = nn.Parameter(torch.randn(args.e_layers, requires_grad=True))
        # classifier
        # classifier_in_channels = self.n_index * 16 \
        #     if not args.model == "tsDynamicer" else 16 * ((self.seq_len - 14) // 5 + 1 + self.n_index)
        self.classifier = nn.Sequential(
            nn.Linear(self.n_index * (self.pred_len + 16), 64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, n_class),
            nn.Sigmoid()
        )

    def forward(self, source_array, outputs_y, batch_y=None):
        """
        get pred label
        """
        B, _, _ = outputs_y.shape
        # get history array
        history_index = self.sample_time_window_before + 1 \
            if self.sample_day_window > 0 else self.sample_time_window_before // 3
        history_array = source_array[:, :-(history_index), :]

        mean = torch.mean(history_array, dim=(0, 1), keepdim=True)
        std = torch.std(history_array, dim=(0, 1), keepdim=True)

        # get de-normalized outputs_y, batch_y
        true_y = source_array[:, -self.pred_len:, :]
        true_y = (true_y - mean) / std
        # de_outputs_y = outputs_y * std + mean
        # get concat data
        concat_before_data = source_array[:, :-self.pred_len, :]
        concat_before_data = (concat_before_data - mean) / std
        concat_outputs = torch.cat([concat_before_data, outputs_y], dim=1)
        concat_batch = torch.cat([concat_before_data, true_y], dim=1)
        # smooth concat data
        concat_outputs = concat_outputs.permute(0, 2, 1)
        concat_batch = concat_batch.permute(0, 2, 1)
        smoothed_outputs = self.mv_avg(concat_outputs)
        smoothed_batch = self.mv_avg(concat_batch)
        # get current smoothed data
        output_res = smoothed_outputs[:, :, -self.pred_len:]
        batch_res = smoothed_batch[:, :, -self.pred_len:]
        # get diff of pred data and real data
        res_diff = output_res - batch_res
        # process diff res
        res_diff = self.diff_block(res_diff)
        # print(res_diff.shape)
        # use trend infos
        trend_out = self.trend_block(concat_batch)
        # print(trend_out.shape)
        # expand diff with trend
        res_diff = torch.cat((res_diff, trend_out), dim=-1)
        res_diff = res_diff.permute(0, 2, 1).reshape(B, -1)
        # get pred label
        labels = self.classifier(res_diff)
        # # labels = labels.squeeze(2)
        # output_res = (output_res.permute(0, 2, 1) - mean) / std
        # batch_res = (batch_res.permute(0, 2, 1) - mean) / std
        return labels
