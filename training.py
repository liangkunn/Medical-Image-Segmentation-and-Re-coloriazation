
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
from DICEloss import DICELoss



def train_segmentation(train_dataloader, validation_dataloader, model, num_epochs, optimizer):
  print(" ======> Start Training...")
  train_losses = []
  val_losses = []
  loss = DICELoss()
  for epoch in range(num_epochs):
      ########################### Training #####################################
      # print("\nEPOCH " +str(epoch+1)+" of "+str(num_epochs)+"\n")

      # TODO: Design your own training section
      # following comments are added by student
      model.train()
      train_loss_epoch = 0
      val_loss_epoch = 0
      # iterate through batches
      for i, (images, labels) in enumerate(train_dataloader):

          if torch.cuda.is_available():
              images = images.cuda()
              labels = labels.cuda() 
          # zero grad for each batch    
          optimizer.zero_grad()
          
          # making predictions
          train_out = model(images)

          # compute loss
          train_loss = loss.forward(train_out, labels) 
          train_loss_epoch += train_loss.item()

          # backward propagation
          train_loss.backward()
          
          # update optimization location
          optimizer.step()

      # log the average train_loss after each trainig batch
      train_losses.append(train_loss_epoch/len(train_dataloader))

      ########################### Validation #####################################
      # TODO: Design your own validation section
      for i, (images, labels) in enumerate(validation_dataloader):
          if torch.cuda.is_available():
              images = images.cuda()
              labels = labels.cuda()

          # making predictions
          val_out = model(images)

          # compute loss
          val_loss = loss.forward(val_out, labels) 
          val_loss_epoch += val_loss.item()

      # for logging
      val_losses.append(val_loss_epoch/len(validation_dataloader))
      
      print("{}/{} Epochs | Train Loss={:.4f} | Val_loss={:.4f}".format(epoch+1, num_epochs, train_loss_epoch/len(train_dataloader), val_loss_epoch/len(validation_dataloader)))

def train_colorization(train_dataloader_cor, validation_dataloader_cor, model_cor, num_epochs, optimizer):
      
  print(" ======> Start Training...")
  train_losses_cor = []
  val_losses_cor = []

  # loss function here is MSE
  loss_cor = nn.MSELoss()

  for epoch in range(num_epochs):
      ########################### Training #####################################
      # print("\nEPOCH " +str(epoch+1)+" of "+str(num_epochs)+"\n")

      # TODO: Design your own training section
      # following comments are added by student
      model_cor.train()
      train_loss_epoch = 0
      val_loss_epoch = 0
      # iterate through batches
      for i, (images, labels) in enumerate(train_dataloader_cor):
          
          # images shape here N*1*height*width 
          # concatenate images to 3 channel input because UNET input channel should be 3
          images = torch.cat([images, images, images], 1) # images shape now: N*3*height*width
          
          if torch.cuda.is_available():
              images = images.cuda()
              labels = labels.cuda() 
          # zero grad for each batch    
          optimizer.zero_grad()
          
          # making predictions
          train_out = model_cor(images)

          # compute loss
          train_loss = loss_cor.forward(train_out, labels) 
          train_loss_epoch += train_loss.item()

          # backward propagation
          train_loss.backward()
          
          # update optimization location
          optimizer.step()

      # log the average train_loss after each trainig batch
      train_losses_cor.append(train_loss_epoch/len(train_dataloader_cor))

      ########################### Validation #####################################
      # TODO: Design your own validation section
      for i, (images, labels) in enumerate(validation_dataloader_cor):

          images = torch.cat([images, images, images], 1)

          if torch.cuda.is_available():
              images = images.cuda()
              labels = labels.cuda()

          # making predictions
          val_out = model_cor(images)

          # compute loss
          val_loss = loss_cor.forward(val_out, labels) 
          val_loss_epoch += val_loss.item()

      # for logging
      val_losses_cor.append(val_loss_epoch/len(validation_dataloader_cor))
      
      print("{}/{} Epochs | Train Loss={:.4f} | Val_loss={:.4f}".format(epoch+1, num_epochs, train_loss_epoch/len(train_dataloader_cor), val_loss_epoch/len(validation_dataloader_cor)))
