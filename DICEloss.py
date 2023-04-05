
import numpy as np
import torch.nn as nn

class DICELoss(nn.Module):
    def __init__(self):
        super(DICELoss).__init__()

    def forward(self, pred, target):
        
        epsilon = 0.01
        
        # student adds: (pred*target) shape is Batch_size * n_classes * height * weight
        intersection = (pred*target).sum(dim=[2,3]) # intersection shape: Batch_size * n_classes

        numerator = 2*intersection # numerator shape: Batch_size * n_classes
        denominator = pred.sum(dim=[2,3]) + target.sum(dim=[2,3]) + epsilon # denominator shape: Batch_size * n_classes
        dice_score = numerator / denominator # dice_score shape: Batch_size * n_classes

        # compute mean for both Batch_size dimension and n_classes dimension
        dice_avg = dice_score.mean() # dice_avg shape: 1
        dice_loss = 1 -  dice_avg
        return dice_loss
