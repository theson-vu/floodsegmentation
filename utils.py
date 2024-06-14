import torch.nn as nn
import torch

class XEDiceLoss(nn.Module):
    """
    Mixture of alpha * CrossEntropy and (1 - alpha) * DiceLoss.
    """
    def __init__(self, alpha=0.5, EPS=1e-7):
        super().__init__()
        self.xe = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.EPS = EPS

    def forward(self, preds, targets):
        """xe_loss = self.xe(preds, targets)
        
        no_ignore = targets.ne(255)
        targets = targets.masked_select(no_ignore)

        preds = torch.softmax(preds, dim=1)[:, 1]
        targets = (targets == 1).float()
        dice_loss = 1 - (2.0 * torch.sum(preds * targets)) / (torch.sum(preds + targets) + EPS)"""
        # Compute CrossEntropy loss
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

    def get_name(self):
        return "XEDiceLoss"
    
def tp_fp_fn(preds, targets):
    tp = torch.sum(preds * targets)
    fp = torch.sum(preds) - tp
    fn = torch.sum(targets) - tp
    return tp.item(), fp.item(), fn.item()


def collate(batch):
    all_x_data = []
    all_targets = []
    for augmented_samples in batch:
        for sample in augmented_samples:
            all_x_data.append(torch.tensor(sample["image"]))
            all_targets.append(torch.tensor(sample["mask"]))
    return torch.stack(all_x_data), torch.stack(all_targets)  
    #return [item for sublist in batch for item in sublist]

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