# coding=utf-8
# author=zhangmeixian
"""
TSDynamicer Encoder-Decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """

    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        """
        forward
        """
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        """
        forward
        """
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class MultiPeriodDecomp(nn.Module):
    """
    time series season decomposition with different period
    """

    def __init__(self, moving_kernel, n_index, single_period_len, seq_len, dropout):
        super(MultiPeriodDecomp, self).__init__()
        self.n_index = n_index
        self.single_period_len = single_period_len
        self.seq_len = seq_len
        self.periods = self.get_periods(moving_kernel, single_period_len, seq_len)
        self.moving_avg = moving_avg(moving_kernel, stride=1)
        self.season_convolutions = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(in_channels=n_index, out_channels=n_index, kernel_size=period),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
                for period in self.periods])
        self.seasonal_weights = nn.Parameter(torch.randn(len(self.periods) + 1, requires_grad=True))

    def get_periods(self, moving_kernel, single_period_len, seq_len, topk=2):
        """
        get multi periods
        """
        periods = [moving_kernel]
        n_periods = seq_len // single_period_len - 1
        for i in range(n_periods):
            periods.append(single_period_len * (i + 1))
            if i >= topk:
                break
        return periods

    def padding(self, results, x):
        """
        padding with differernt results
        """
        _, _, D = x.shape
        padded_results = [x]
        for result in results:
            result = F.pad(result, (0, D - result.shape[2]))
            padded_results.append(result)
        return padded_results

    def forward(self, x):
        """
        forward cal
        """
        trend_res = self.moving_avg(x)
        x = x.permute(0, 2, 1)
        seasonal_reses = self.padding([conv(x) for conv in self.season_convolutions], x)
        seasonal_weights = torch.nn.functional.softmax(self.seasonal_weights, dim=0)
        weighted_seasonal_reses = [w * sr for w, sr in zip(seasonal_weights, seasonal_reses)]
        seasonal_res = torch.sum(torch.stack(weighted_seasonal_reses), dim=0)
        seasonal_res = seasonal_res.permute(0, 2, 1)
        seasonal_res = seasonal_res - trend_res
        return seasonal_res, trend_res


class EncoderLayer(nn.Module):
    """
    tsDynamicer encoder layer with the progressive decomposition architecture
    """

    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu",
                 single_period_len=None, seq_len=None):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = MultiPeriodDecomp(moving_avg, d_model, single_period_len, seq_len, dropout)
        self.decomp2 = MultiPeriodDecomp(moving_avg, d_model, single_period_len, seq_len, dropout)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None, batch_day_weights=None, batch_time_weights=None):
        """
        forward
        """
        # auto correlation
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta, batch_day_weights=batch_day_weights, batch_time_weights=batch_time_weights
        )
        # residual block
        x = x + self.dropout(new_x)
        # series decomposition
        x, trend_x = self.decomp1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        # series decomposition
        res, trend_y = self.decomp2(x + y)
        trend_res = trend_x + trend_y
        return res, trend_res, attn


class Encoder(nn.Module):
    """
    tsDynamicer encoder
    """

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None, batch_day_weights=None, batch_time_weights=None):
        """
        forward
        """
        attns, trends = [], []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, trend_x, attn = attn_layer(
                    x, attn_mask=attn_mask, tau=tau, delta=delta, batch_day_weights=batch_day_weights,
                    batch_time_weights=batch_time_weights)
                x = conv_layer(x)
                attns.append(attn)
                trends.append(trend_x)
            x, trend_x, attn = self.attn_layers[-1](
                x, tau=tau, delta=delta, batch_day_weights=batch_day_weights, batch_time_weights=batch_time_weights)
            attns.append(attn)
            trends.append(trend_x)
        else:
            for attn_layer in self.attn_layers:
                x, trend_x, attn = attn_layer(
                    x, attn_mask=attn_mask, tau=tau, delta=delta, batch_day_weights=batch_day_weights,
                    batch_time_weights=batch_time_weights)
                attns.append(attn)
                trends.append(trend_x)

        if self.norm is not None:
            x = self.norm(x)

        return x, trends, attns


class DecoderLayer(nn.Module):
    """
    tsDynamicer decoder layer with the progressive decomposition architecture
    """

    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu", single_period_len=None, seq_len=None):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = MultiPeriodDecomp(moving_avg, d_model, single_period_len, seq_len, dropout)
        self.decomp2 = MultiPeriodDecomp(moving_avg, d_model, single_period_len, seq_len, dropout)
        self.decomp3 = MultiPeriodDecomp(moving_avg, d_model, single_period_len, seq_len, dropout)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        """
        forward
        """
        # Note that multichannel and delta only used for Self-Attention(x_enc with x_enc)
        # and Cross-Attention(x_enc with x_dec),
        # but not suitable for Self-Attention(x_dec with x_dec)

        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x, trend1 = self.decomp1(x)
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])
        x, trend2 = self.decomp2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend


class Decoder(nn.Module):
    """
    Autoformer encoder
    """

    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None, tau=None, delta=None):
        """
        forward
        """
        for layer in self.layers:
            x, residual_trend = layer(
                x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x, trend
