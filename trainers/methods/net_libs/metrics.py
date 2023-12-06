# -*- coding: UTF-8 -*-
# !/usr/bin/python3.8
# copyright: https://github.com/thuml/Nonstationary_Transformers
# author: <liuyong> (<liuyong21@mails.tsinghua.edu.cn>)
"""
metrics
"""

import numpy as np


def RSE(pred, true):
    """
    RSE
    """
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    """
    CORR
    """
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    """
    MAE
    """
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    """
    MSE
    """
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    """RMSE"""
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    """MAPE"""
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    """MSPE"""
    return np.mean(np.square((pred - true) / true))


def quantile_error(pred, true, q=0.5):
    """quantile error"""
    diff = true - pred
    error = np.where(diff >= 0, q * diff, (1 - q) * (-diff))
    return np.mean(error)


def metric(pred, true, q=0.5):
    """
    get metrics
    """
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    qe = quantile_error(pred, true, q)

    return mae, mse, rmse, mape, mspe, qe
