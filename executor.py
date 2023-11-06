# -*- coding: UTF-8 -*-
# !/usr/bin/python3.8

"""
@ModuleName: 数值预测模型训练入口
@author: zhangmeixian
"""

import argparse
from trainers.exp_executor import ExpExecutor


def main():
    """
    数值预测模型训练入口
    """
    parser = argparse.ArgumentParser(description="Trainer for Adjusted Non-stationary Dynamic Time Series Forecasting")

    # basic config
    parser.add_argument("--is_training", action="store_false", default=True, help="Training status. Default is True.")
    parser.add_argument("--do_anomaly_detection", action="store_false", default=True,
                        help="anomaly detection status. Default is True.")
    parser.add_argument("--anomaly_loss_weight", type=float, default=2.5, help="weight of anomaly detection loss")
    parser.add_argument("--model", choices=["adj_ns_Transformer", "adj_ns_Autoformer", "adj_ns_Informer",
                                            "adj_ns_FEDformer", "ts_Dynamicer"],
                        required=True, default="ts_Dynamicer", help="Model name. Default is ts_Dynamicer.")
    parser.add_argument("--model_des", type=str, required=True, default="CSM_train", help="model description.")
    parser.add_argument("--exp_des", type=str, default="CSM_train", help="experiment description")
    # special basic configs of FEDformer
    parser.add_argument("--version", type=str, default="Waveets",
                        help="for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]")
    parser.add_argument("--mode_select", type=str, default="random",
                        help="for FEDformer, there are two mode selection method, options: [random, low]")
    parser.add_argument("--modes", type=int, default=64, help="for FEDformer, modes to be selected random 64")
    parser.add_argument("--L", type=int, default=3, help="for FEDformer, ignore level")
    parser.add_argument("--base", type=str, default="legendre", help="for FEDformer, mwt base")
    parser.add_argument("--cross_activation", type=str, default="tanh",
                        help="for FEDformer, mwt cross atention activation function tanh or softmax")
    # special basic configs of ts_Dynamicer
    parser.add_argument("--region", type=str, default="China",
                        help="for ts_Dynamicer, get special day extra information, "
                             "can get from package e.g.: holidays, workalendar, chinese calendar")
    parser.add_argument("--neighbor_window", type=int, default=10, help="for ts_Dynamicer, neighbor time window")

    # dataset config
    parser.add_argument("--dataset", type=str, required=True, default="CSM", help="dataset field")
    parser.add_argument("--forcast_task", type=str, default="S",
                        help="forecasting task, options:[M, S, MS]; "
                             "M:multivariate predict multivariate, "
                             "S:univariate predict univariate, MS:multivariate predict univariate")
    parser.add_argument("--target_index", type=str, default="data", help="target predict index name in S or MS task")
    parser.add_argument("--n_index", type=int, default=1, help="number of index in forcast task")
    parser.add_argument("--freq", type=str, default="h",
                        help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, "
                             "d:daily, b:business days, w:weekly, m:monthly], "
                             "you can also use more detailed freq like 15min or 3h")
    parser.add_argument("--sample_time_window_before", type=int, default=30,
                        help="Neighbor window before length to process and get samples.")
    parser.add_argument("--sample_time_window_after", type=int, default=0,
                        help="Neighbor window after length to process and get samples.")
    parser.add_argument("--sample_day_window", type=int, default=14,
                        help="History day window for sample to reference, 0 represent no history day.")
    parser.add_argument("--model_save_path", type=str, default="./models/", help="model checkpoints save path")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="train data ratio.")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="test data ratio.")

    # forecasting config
    parser.add_argument("--seq_len", type=int, default=464, help="input sequence length")
    parser.add_argument("--label_len", type=int, default=48, help="start token length")
    parser.add_argument("--pred_len", type=int, default=1, help="prediction sequence length")

    # model config
    parser.add_argument("--enc_in", type=int, default=7, help="encoder input size")
    parser.add_argument("--dec_in", type=int, default=7, help="decoder input size")
    parser.add_argument("--c_out", type=int, default=7, help="output size")
    parser.add_argument("--d_model", type=int, default=512, help="dimension of model")
    parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
    parser.add_argument("--e_layers", type=int, default=2, help="num of encoder layers")
    parser.add_argument("--d_layers", type=int, default=1, help="num of decoder layers")
    parser.add_argument("--d_ff", type=int, default=2048, help="dimension of fcn")
    parser.add_argument("--moving_avg", type=int, default=5, help="window size of moving average")
    parser.add_argument("--factor", type=int, default=1, help="attn factor")
    parser.add_argument("--distil", action="store_false",
                        help="whether to use distilling in encoder, using this argument means not using distilling",
                        default=True)
    parser.add_argument("--dropout", type=float, default=0.05, help="dropout")
    parser.add_argument("--embed", type=str, default="timeF",
                        help="time features encoding, options:[timeF, fixed, learned]")
    parser.add_argument("--activation", type=str, default="gelu", help="activation")
    parser.add_argument("--output_attention", action="store_true", help="whether to output attention in encoder")

    # optimization config
    parser.add_argument("--num_workers", type=int, default=10, help="data loader num workers")
    parser.add_argument("--itr", type=int, default=1, help="experiments times")
    parser.add_argument("--train_epochs", type=int, default=10, help="train epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size of train input data")
    parser.add_argument("--patience", type=int, default=3, help="early stopping patience")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="optimizer learning rate")
    parser.add_argument("--loss", type=str, default="sign_mse_anomaly",
                        choices=["mse", "sign_mse", "sign_mse_anomaly", "quantile_loss", "weighted_quantile_loss"],
                        help="loss function")
    parser.add_argument("--lambda_weight", type=float, default=0.01, help="weight of sign_mse loss function")
    parser.add_argument("--ita_weight", type=float, default=0.01, help="weight of sign_mse_anomaly loss function")
    parser.add_argument("--q_value", type=float, default=0.5, help="quantile value of quantile_loss")
    parser.add_argument("--alpha", type=float, default=1, help="param of weighted quantile_loss")
    parser.add_argument("--beta", type=float, default=1, help="param of weighted quantile_loss")
    parser.add_argument("--lradj", type=str, default="type1", help="adjust learning rate")
    parser.add_argument("--use_amp", action="store_true", help="use automatic mixed precision training", default=False)

    # GPU config
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--use_multi_gpu", action="store_true", help="use multiple gpus", default=False)
    parser.add_argument("--devices", type=str, default="0,1,2,3", help="device ids of multile gpus")
    parser.add_argument("--seed", type=int, default=2023, help="random seed")

    # de-stationary projector params
    parser.add_argument("--p_hidden_dims", type=int, nargs="+", default=[256, 256],
                        help="hidden layer dimensions of projector (List)")
    parser.add_argument("--p_hidden_layers", type=int, default=2, help="number of hidden layers in projector")

    # get input args
    args = parser.parse_args()

    # training & testing
    ExpExecutor(args).execute()


if __name__ == "__main__":
    main()
