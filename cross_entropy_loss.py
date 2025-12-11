import torch
import numpy as np

def binary_cross_entropy_loss(y_true, y_pred):
    ce_loss = -torch.sum(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred)) / y_true.shape[0]
    return ce_loss


def multiple_cross_entropy_loss(y_true, y_pred):
    ce_loss = -torch.sum(y_true * torch.log(y_pred)) / y_true.shape[0]
    return ce_loss
