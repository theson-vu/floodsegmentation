import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as F
import numpy as np


class XEDiceLoss(nn.Module):
    def __init__(self, alpha=0.5, EPS=1e-7):
        super().__init__()
        self.xe = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.EPS = EPS

    def forward(self, preds, targets):        
        # Compute Cross-Entropy loss
        xe_loss = self.xe(preds, targets)

        # Compute Dice loss
        softmax_preds = torch.softmax(preds, dim=1)
        preds = softmax_preds[:, 1]  # Assuming binary segmentation (foreground/background)
        targets = (targets == 1).float()
        intersection = torch.sum(preds * targets)
        union = torch.sum(preds) + torch.sum(targets)
        dice_loss = 1 - (2.0 * intersection) / (union + self.EPS)

        # Compute weighted combination of CrossEntropy and Dice losses
        return self.alpha * xe_loss + (1 - self.alpha) * dice_loss 
    

def tp_tn_fp_fn(preds, targets):
    tp = torch.sum(preds * targets)
    fp = torch.sum(preds) - tp
    fn = torch.sum(targets) - tp
    tn = torch.sum((1 - preds) * (1 - targets))
    return tp.item(), fp.item(), fn.item(), tn.item()


def collate(batch):
    # Initialize lists to store different data components
    all_x_data = []
    all_amps = []
    all_phases = []
    all_targets = []
    all_wavelets = []

    # Iterate through the batch and extract data from each sample
    for augmented_samples in batch:
        for sample in augmented_samples:
            all_x_data.append(np.array(sample["image"]))
            all_targets.append(np.array(sample["mask"]))
            all_amps.append(np.array(sample["amplitude"]))
            all_phases.append(np.array(sample["phase"]))   
            all_wavelets.append(np.array(sample["wavelet"]))

    # Convert lists to NumPy arrays and stack them
    all_x_data = torch.tensor(np.stack(all_x_data))
    all_targets = torch.tensor(np.stack(all_targets))
    all_amps = torch.tensor(np.stack(all_amps))
    all_phases = torch.tensor(np.stack(all_phases))
    all_wavelets = torch.tensor(np.stack(all_wavelets))

    return all_x_data, all_amps, all_phases, all_wavelets, all_targets


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count