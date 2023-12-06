# -*- coding: utf-8 -*-

"""
@ModuleName: custom losses
@author: zhangmeixian
"""

import torch
import warnings
from torch.nn import Module
from configs.constants import status_exception, status_exception_no


def cal_sign_diff_mse_loss(pred: torch.Tensor, target: torch.Tensor, lambda_weight=0.01):
    """
    cal diff mse loss
    """
    sign_mismatch_loss = torch.relu(-target.sign() * pred)
    mae_loss = (pred - target) ** 2
    total_loss = mae_loss + lambda_weight * sign_mismatch_loss
    return total_loss


def sign_diff_mse_loss(pred: torch.Tensor, target: torch.Tensor, label=torch.Tensor, lambda_weight=0.01):
    """
    diff mse loss
    """
    exception_pred = pred[label == status_exception]
    exception_target = target[label == status_exception]
    normal_pred = pred[label == status_exception_no]
    normal_target = target[label == status_exception_no]
    exception_loss = cal_sign_diff_mse_loss(exception_pred, exception_target, lambda_weight)
    exception_loss = torch.where(exception_loss != 0, 1 / exception_loss, 1 / 0.00001)
    normal_loss = cal_sign_diff_mse_loss(normal_pred, normal_target, lambda_weight)
    final_loss = torch.cat([exception_loss, normal_loss], dim=0)
    return torch.mean(final_loss)


def sign_mae_loss(pred: torch.Tensor, target: torch.Tensor, lambda_weight=0.01):
    """
    mae loss with sign adjustment
    """
    sign_mismatch_loss = torch.mean(torch.relu(-target.sign() * pred))
    mae_loss = torch.mean(torch.abs(pred - target))
    total_loss = mae_loss + lambda_weight * sign_mismatch_loss
    return total_loss


def sign_mse_loss(pred: torch.Tensor, target: torch.Tensor, lambda_weight=0.01):
    """
    mse loss with sign adjustment
    """
    sign_mismatch_loss = torch.mean(torch.relu(-target.sign() * pred))
    mse_loss = torch.mean((pred - target) ** 2)
    total_loss = mse_loss + lambda_weight * sign_mismatch_loss
    return total_loss


def sign_quantile_loss(pred: torch.Tensor, target: torch.Tensor, q=0.5, lambda_weight=0.01, alpha=1, beta=1):
    """
    quantile loss with sign adjustment
    """
    if not (target.size() == pred.size()):
        warnings.warn(
            "Using a target size ({}) that is different to the input size ({}). "
            "This will likely lead to incorrect results due to broadcasting. "
            "Please ensure they have the same size.".format(target.size(), pred.size()),
            stacklevel=2,
        )
    expanded_input, expanded_target = torch.broadcast_tensors(pred, target)
    assert 0 <= q <= 1, "q (quantile) should be in [0, 1]"
    diff = expanded_target - expanded_input
    quantile_loss = torch.mean(torch.where(diff >= 0, alpha * q * diff, beta * (1-q) * (-diff)))
    sign_mismatch_loss = torch.mean(torch.relu(-target.sign() * pred))
    total_loss = quantile_loss + lambda_weight * sign_mismatch_loss
    return total_loss


class SignMaeLoss(Module):
    """
    adjusted mse loss
    """

    __constants__ = ['lambda_weight']

    def __init__(self, lambda_weight=0.01) -> None:
        super(SignMaeLoss, self).__init__()
        self.lambda_weight = lambda_weight

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        forward
        """
        return sign_mae_loss(input, target, self.lambda_weight)


class SignQuantileLoss(Module):
    """
    quantile loss
    """

    __constants__ = ['q']

    def __init__(self, q=0.5, lambda_weight=0.01, alpha=1, beta=1) -> None:
        super(SignQuantileLoss, self).__init__()
        self.q = q
        self.lambda_weight = lambda_weight
        self.alpha = alpha
        self.beta = beta

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        forward
        """
        return sign_quantile_loss(input, target, self.q, self.lambda_weight, self.alpha, self.beta)


class SignMseLoss(Module):
    """
    adjusted mse loss
    """

    __constants__ = ['lambda_weight']

    def __init__(self, lambda_weight=0.01) -> None:
        super(SignMseLoss, self).__init__()
        self.lambda_weight = lambda_weight

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        forward
        """
        return sign_mse_loss(input, target, self.lambda_weight)


class SignDiffMseLoss(Module):
    """
    adjusted mse loss
    """

    __constants__ = ['lambda_weight']

    def __init__(self, lambda_weight=0.01) -> None:
        super(SignDiffMseLoss, self).__init__()
        self.lambda_weight = lambda_weight

    def forward(self, input: torch.Tensor, target: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        forward
        """
        return sign_diff_mse_loss(input, target, label, self.lambda_weight)


def quantile_loss(input: torch.Tensor, target: torch.Tensor, q=0.5, alpha=1, beta=1):
    """
    quantile loss
    """
    if not (target.size() == input.size()):
        warnings.warn(
            "Using a target size ({}) that is different to the input size ({}). "
            "This will likely lead to incorrect results due to broadcasting. "
            "Please ensure they have the same size.".format(target.size(), input.size()),
            stacklevel=2,
        )
    expanded_input, expanded_target = torch.broadcast_tensors(input, target)
    assert 0 <= q <= 1, "q (quantile) should be in [0, 1]"
    diff = expanded_target - expanded_input
    loss = torch.where(diff >= 0, alpha * q * diff, beta * (1-q) * (-diff))
    return torch.mean(loss)


class QuantileLoss(Module):
    """
    quantile loss
    """

    __constants__ = ['q']

    def __init__(self, q=0.5) -> None:
        super(QuantileLoss, self).__init__()
        self.q = q

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        forward
        """
        return quantile_loss(input, target, self.q)


class WeightedQuantileLoss(Module):
    """
    weighted quantile loss
    """

    __constants__ = ['q']

    def __init__(self, q=0.5, alpha=1, beta=1) -> None:
        super(WeightedQuantileLoss, self).__init__()
        self.q = q
        self.alpha = alpha
        self.beta = beta

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        forward
        """
        return quantile_loss(input, target, self.q, self.alpha, self.beta)