# coding=utf-8
# author=zhangmeixian
"""
framework of TSDynamicer
"""

import torch
import torch.nn as nn
from trainers.methods.ts_dynamicer.tsDynamicer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, \
    my_Layernorm, MultiPeriodDecomp
from trainers.methods.ts_dynamicer.MultichannelCorrelation import MultichannelCorrelation, DSAutoCorrelation
from trainers.methods.net_libs.embedder import DataEmbedding_wo_pos
from trainers.methods.ts_dynamicer.anomaly_detector import AnomalyDetector


class Projector(nn.Module):
    """
    MLP to learn the De-stationary factors
    """

    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
        super(Projector, self).__init__()

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding,
                                     padding_mode='circular', bias=False)

        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()]

        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        """
        forward
        """
        # x:     B x S x E
        # stats: B x 1 x E
        # y:     B x O
        batch_size = x.shape[0]
        x = self.series_conv(x)  # B x 1 x E
        x = torch.cat([x, stats], dim=1)  # B x 2 x E
        x = x.view(batch_size, -1)  # B x 2E
        y = self.backbone(x)  # B x O

        return y


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.do_anomaly_detection = configs.do_anomaly_detection
        self.forcast_task = configs.forcast_task
        self.f_dim = -1 if self.forcast_task == 'MS' else 0
        self.sample_day_window = configs.sample_day_window
        self.sample_time_window_before = configs.sample_time_window_before
        self.sample_time_window_after = configs.sample_time_window_after
        self.single_period_len = (
                self.sample_time_window_before + 1 + self.sample_time_window_after) if self.sample_day_window > 0 \
            else self.sample_time_window_before // 3

        # MultiPeriodDecomp
        self.decomp = MultiPeriodDecomp(
            configs.moving_avg, configs.n_index, self.single_period_len, self.seq_len, configs.dropout)

        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(
            configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        # Multichannel Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    MultichannelCorrelation(
                        DSAutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads, dropout=configs.dropout),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    single_period_len=self.get_single_period_len(self.seq_len),
                    seq_len=self.seq_len
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Multichannel Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    MultichannelCorrelation(
                        DSAutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=False),
                        configs.d_model, configs.n_heads, dropout=configs.dropout),
                    MultichannelCorrelation(
                        DSAutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=False),
                        configs.d_model, configs.n_heads, dropout=configs.dropout),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    single_period_len=self.get_single_period_len(self.label_len + 1),
                    seq_len=configs.label_len + 1
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

        self.tau_learner = Projector(enc_in=configs.enc_in, seq_len=configs.seq_len, hidden_dims=configs.p_hidden_dims,
                                     hidden_layers=configs.p_hidden_layers, output_dim=1)
        self.delta_learner = Projector(enc_in=configs.enc_in, seq_len=configs.seq_len,
                                       hidden_dims=configs.p_hidden_dims, hidden_layers=configs.p_hidden_layers,
                                       output_dim=configs.seq_len)
        # anomaly detector
        self.anomaly_detector = AnomalyDetector(configs)

    def get_single_period_len(self, input_seq_len):
        """
        get single period len
        """
        if input_seq_len == self.seq_len and self.sample_day_window > 0:
            return self.sample_time_window_before + 1 + self.sample_time_window_after
        return input_seq_len // 3

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, x_source=None, y_enc=None,
                batch_day_weights=None, batch_time_weights=None):
        """
        forward
        """
        x_raw = x_enc.clone().detach()

        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x_enc = x_enc / std_enc
        x_dec_new = torch.cat([x_enc[:, -self.label_len:, :], torch.zeros_like(x_dec[:, -self.pred_len:, :])],
                              dim=1).to(x_enc.device).clone()

        tau = self.tau_learner(x_raw, std_enc).exp()  # B x S x E, B x 1 x E -> B x 1, positive scalar
        delta = self.delta_learner(x_raw, mean_enc)  # B x S x E, B x 1 x E -> B x S

        # Model Inference

        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec_new.shape[0], self.pred_len, x_dec_new.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, enc_trends, attns = self.encoder(
            enc_out, attn_mask=enc_self_mask, tau=tau, delta=delta, batch_day_weights=batch_day_weights,
            batch_time_weights=batch_time_weights)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(
            dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, trend=trend_init, tau=tau, delta=None)
        # final
        dec_out = trend_part + seasonal_part

        # De-normalization
        dec_out = dec_out * std_enc + mean_enc
        dec_out = dec_out[:, -self.pred_len:, :]

        # get find predict output
        outputs = dec_out[:, -self.pred_len:, self.f_dim:]

        if self.do_anomaly_detection:
            # get pred labels
            output_labels, s_outputs, s_trues = self.anomaly_detector(x_source, outputs, y_enc, enc_trends=None)
            if self.output_attention:
                return outputs, output_labels, s_outputs, s_trues, attns
            else:
                return outputs, output_labels, s_outputs, s_trues
        else:
            if self.output_attention:
                return outputs, attns
            else:
                return outputs
