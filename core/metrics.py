import numpy as np
import torch
from sklearn.metrics import roc_auc_score

def acc(ops, lbls):
    ops = ops.detach().cpu().numpy()
    lbls = lbls.detach().cpu().numpy()
    ops = np.argmax(ops, axis=1)
    score = np.sum(ops == lbls) / len(ops)
    
    return score


def auc(ops, lbls):
    ops = ops.detach().cpu().numpy()
    lbls = lbls.detach().cpu().numpy()
    score = roc_auc_score(lbls, ops[:, 1])
    
    return score
