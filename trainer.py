# -*- coding: UTF-8 -*-
# !/usr/bin/python3.8

"""
@ModuleName: 数值预测模型训练入口
@author: zhangmeixian
"""

import argparse
from trainers.train_executor import TrainExecutor


def main():
    """
    数值预测模型训练入口
    """
    parser = argparse.ArgumentParser(description='Trainer for Adjusted Non-stationary Dynamic Time Series Forecasting')

    # basic config
    parser.add_argument('--no_training', action="store_false", default=True, help='Training status. Default is True.')
    parser.add_argument('--model_name', choices=["ns_Transformer", "Transformer", "Autoformer", "ns_Autoformer",
                                                 "Informer", "ns_Informer"], required=True, default='ns_Autoformer',
                        help='Model name. Default is ns_Autoformer.')
    parser.add_argument('--model_des', type=str, required=True, default='SMD_train', help='model description.')
    parser.add_argument('--description', type=str, default='SMD_train', help='experiment description')

    # dataset config
    parser.add_argument('--dataset_field', type=str, required=True, default='SMD', help='dataset field')
    parser.add_argument('--forcast_task', type=str, default='S',
                        help='forecasting task, options:[M, S, MS]; '
                             'M:multivariate predict multivariate, '
                             'S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target_index', type=str, default='index1', help='target predict index name in S or MS task')
    parser.add_argument('--time_embed_freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, '
                             'd:daily, b:business days, w:weekly, m:monthly], '
                             'you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--model_save_path', type=str, default='./models/', help='model checkpoints save path')

    # forecasting config
    parser.add_argument('--seq_len', type=int, default=464, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')

    # model config
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=5, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization config
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU config
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--seed', type=int, default=2023, help='random seed')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[256, 256],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # get input args
    args = parser.parse_args()

    # training & testing
    TrainExecutor.execute(args)


if __name__ == '__main__':
    main()