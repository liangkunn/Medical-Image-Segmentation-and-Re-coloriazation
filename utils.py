## Standard Library
import os
import json

## External Libraries
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as functional
from torch.utils.data import Dataset, DataLoader
from skimage import io
import matplotlib.pyplot as plt


def add_conv_stage(dim_in,
                   dim_out,
                   kernel_size=3,
                   stride=1,
                   padding=1,
                   bias=True,
                   useBN=True):
    """
    
    """
    # Use batch normalization
    if useBN:
        return nn.Sequential(
          nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
          nn.BatchNorm2d(dim_out),
          nn.LeakyReLU(0.1),
          nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
          nn.BatchNorm2d(dim_out),
          nn.LeakyReLU(0.1)
        )
    # No batch normalization
    else:
        return nn.Sequential(
          nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
          nn.ReLU(),
          nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
          nn.ReLU()
        )

## Upsampling
def upsample(ch_coarse,
             ch_fine):
    """
    
    """
    return nn.Sequential(
                    nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
                    nn.ReLU())


def dice_score_image(prediction, target, n_classes):
    '''
      computer the mean dice score for a single image

      Reminders: A false positive is a result that indicates a given condition exists, when it does not
               A false negative is a test result that indicates that a condition does not hold, while in fact it does
      Args:
          prediction (tensor): predictied labels of the image
          target (tensor): ground truth of the image
          n_classes (int): number of classes
      
          prediciton here is an image, with shape B * 1 * H * W where B=1, H=image height, W=image weight
          target here is ground truth of the image also with shape B * 1 * H * W where B=1
    
      Returns:
          m_dice (float): Mean dice score over classes
    '''
    ## Should test image one by one
    assert prediction.shape[0] == 1 #This line can not be deleted
    
    dice_classes = np.zeros(n_classes)

    epsilon = 0.01 #  prevent undesired behavior in muliplication/division

    for cl in range(n_classes):
        target_img = target[:,cl] #  obtain the corresponding slice of the target mask

        #  Computes the element-wise logical AND of the given input tensors
        TP = torch.logical_and(prediction == cl, target_img == 1).sum() 
        FP = torch.logical_and(prediction == cl, target_img == 0).sum()
        FN = torch.logical_and(prediction != cl, target_img == 1).sum()
        
        # When there is no ground truth of the class in this image
        # Give 1 dice score if False Positive pixel number is 0, 
        # give 0 dice score if False Positive pixel number is not 0 (> 0).
        
        #  no ground truth of the class in the image means all 0 in target_img
        if (TP + FN) == 0: 
            if FP == 0:
                dice_classes[cl] = 1
            else:
                dice_classes[cl] = 0
        else:
            #  calculate dice score
            dice_classes[cl] = (2*TP + epsilon) / (2*TP + FP + FN)

    return dice_classes.mean()



def dice_score_dataset(model, dataloader, num_classes, use_gpu=False):
    """
    Compute the mean dice score on a set of data.
    
    Note that multiclass dice score can be defined as the mean over classes of binary
    dice score. Dice score is computed per image. Mean dice score over the dataset is the dice
    score averaged across all images.
    
    Reminders: A false positive is a result that indicates a given condition exists, when it does not
               A false negative is a test result that indicates that a condition does not hold, while in fact it does
     
    Args:
        model (UNET class): Your trained model
        dataloader (DataLoader): Dataset for evaluation
        num_classes (int): Number of classes
    
    Returns:
        m_dice (float): Mean dice score over the input dataset
    """
    ## Number of Batches and Cache over Dataset 
    n_batches = len(dataloader)
    scores = np.zeros(n_batches)
    ## Evaluate
    model.eval()
    idx = 0 # idx here indicating how many batches we have
    for data in dataloader: #  data here is data of each batch
        ## Format Data
        img, target = data
        if use_gpu:
            img = img.cuda()
            target = target.cuda()
        ## Make Predictions
        out = model(img) #  shape [1, 8, 256, 320] 
        n_classes = out.shape[1] 
        prediction = torch.argmax(out, dim = 1) # shape: [1, 256, 320], each pixel is a class number
        scores[idx] = dice_score_image(prediction, target, n_classes)
        idx += 1
    ## Average Dice Score Over Images
    m_dice = scores.mean()
    return m_dice
