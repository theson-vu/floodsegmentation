import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as F


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
    

class ResNet34FeatureExtractor(nn.Module):
    def __init__(self, output_channels):
        super(ResNet34FeatureExtractor, self).__init__()
        self.resnet = models.resnet34(pretrained=True)
        
        # Remove the fully connected layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        
        # Upsample layers to match the input dimensions
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Convolutional layer to adjust the number of channels
        self.conv1x1 = nn.Conv2d(512, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.resnet(x)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.upsample4(x)
        x = self.upsample5(x)
        x = self.conv1x1(x)
        return x
    

def tp_tn_fp_fn(preds, targets):
    tp = torch.sum(preds * targets)
    fp = torch.sum(preds) - tp
    fn = torch.sum(targets) - tp
    tn = torch.sum((1 - preds) * (1 - targets))
    return tp.item(), fp.item(), fn.item(), tn.item()


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