from venv import create
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import cv2
from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import sys
import os

from config.train_config import parse_train_configs
from models import CustomCNN, InceptionModule, ResNet

from utils.model_utils import create_model, create_optimizer
from utils.train_utils import train_model, plot_performance
from utils.eval_utils import eval_model
from utils.dataset_utils import *

def main():
    configs = parse_train_configs()
    model = create_model(configs).to(device=configs.device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(configs, model)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=configs.step_size, gamma=configs.gamma)

    model, losses, accuracies = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=configs.num_epochs)
    torch.save(model.state_dict(),"weights.pth")    

    plot_performance('Accuracy', accuracies)
    plot_performance('Loss', losses)

    torch.cuda.empty_cache() 
    eval_model(model, criterion, optimizer, exp_lr_scheduler)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
   