import argparse
import sys
import random
import os
import time
import math
import torch
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics

def seed_torch(seed=2):
    print('seed=', seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class Logger(object):
    def __init__(self, log_name):
        self.terminal = sys.stdout
        self.log_name = log_name
        os.makedirs(os.path.dirname(log_name), exist_ok=True)
        self.log = open(self.log_name, "a", buffering=1)

    def write(self, message):
        if message.strip() != "":
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            message = f"[{timestamp}] {message}"
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return self.terminal.isatty()

def generate_flow_mask(f1, f2, tau_m1, tau_m2, tau_d):
    B, C, H, W = f1.shape

    magnitude_f1 = torch.norm(f1, dim=1)
    magnitude_f2 = torch.norm(f2, dim=1)

    R1 = magnitude_f1 >= tau_m1
    R2 = magnitude_f2 >= tau_m2
    M_m = torch.logical_or(R1, R2)

    theta_f1 = torch.atan2(f1[:, 1, :, :], f1[:, 0, :, :])
    theta_f2 = torch.atan2(f2[:, 1, :, :], f2[:, 0, :, :])

    theta_diff = torch.abs(theta_f1 - theta_f2)
    theta_diff = torch.minimum(theta_diff, 2 * torch.pi - theta_diff)
    M_d = theta_diff <= tau_d

    M_p = M_m * M_d
    M_p = M_p.unsqueeze(1).float()
    return M_p

def calculate_thresholds(f1, f2, magnitude_percentile, direction_percentile):
    magnitude_f1 = torch.norm(f1, dim=1)
    magnitude_f2 = torch.norm(f2, dim=1)

    tau_m1 = torch.quantile(magnitude_f1, float(magnitude_percentile))
    tau_m2 = torch.quantile(magnitude_f2, float(magnitude_percentile))

    theta_f1 = torch.atan2(f1[:, 1, :, :], f1[:, 0, :, :])
    theta_f2 = torch.atan2(f2[:, 1, :, :], f2[:, 0, :, :])

    theta_diff = torch.abs(theta_f1 - theta_f2)
    theta_diff = torch.minimum(theta_diff, 2 * torch.pi - theta_diff)

    tau_d = torch.quantile(theta_diff, float(direction_percentile))
    return tau_m1.item(), tau_m2.item(), tau_d.item()

def to_one_hot(inp, num_classes):
    y_onehot = torch.FloatTensor(inp.size(0), num_classes)
    y_onehot.zero_()
    y_onehot.scatter_(1, inp.unsqueeze(1).data.cpu(), 1)
    if torch.cuda.is_available():
        y_onehot = y_onehot.cuda()
    return Variable(y_onehot, requires_grad=False)

def confusionMatrix(gt, pred, show=False):
    cm = confusion_matrix(gt, pred, labels=[0, 1])
    if cm.shape != (2, 2):
        return 0.0, 0.0

    TN, FP, FN, TP = cm.ravel()
    denom_f1 = 2 * TP + FP + FN
    f1_score = (2 * TP) / denom_f1 if denom_f1 > 0 else 0.0

    num_samples = len([x for x in gt if x == 1])
    average_recall = TP / num_samples if num_samples > 0 else 0.0

    return f1_score, average_recall

def recognition_evaluation(final_gt, final_pred, show=False):
    label_dict = {'negative': 0, 'positive': 1, 'surprise': 2}
    f1_list = []
    ar_list = []

    for emotion, emotion_index in label_dict.items():
        gt_recog = [1 if x == emotion_index else 0 for x in final_gt]
        pred_recog = [1 if x == emotion_index else 0 for x in final_pred]
        f1_recog, ar_recog = confusionMatrix(gt_recog, pred_recog)
        f1_list.append(f1_recog)
        ar_list.append(ar_recog)

    UF1 = np.mean(f1_list)
    UAR = np.mean(ar_list)
    return UF1, UAR

class mAP:
    def __init__(self):
        self.type = 0

class accuracy:
    def __init__(self):
        self.type = 0

    def eval(self, pred_v, true_v):
        idx_a = [i for i, value in enumerate(pred_v) if pred_v[i] == true_v[i]]
        acc_weighted = float(len(idx_a)) / float(len(pred_v))

        labels = torch.unique(true_v)
        acc = torch.zeros(len(labels))
        for i in range(len(labels)):
            idx_c = [j for j in range(len(true_v)) if true_v[j] == labels[i]]
            acc[i] = torch.sum(pred_v[idx_c] == true_v[idx_c]).double() / float(len(idx_c))

        acc_unweighted = torch.mean(acc)
        return acc_weighted, acc_unweighted

class f1score:
    def __init__(self):
        self.type = 0

    def eval(self, pred_v, true_v):
        f1 = metrics.f1_score(true_v.float(), pred_v.float(), average='micro')
        f1_weighted = metrics.f1_score(true_v.float(), pred_v.float(), average='macro')
        return f1, f1_weighted
