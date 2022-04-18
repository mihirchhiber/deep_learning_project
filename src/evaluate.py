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

from utils.model_utils import create_model, create_optimizer
from utils.train_utils import train_model, plot_performance
from utils.eval_utils import eval_model
from utils.preprocessing_utils import create_df, create_dataloaders

def main():
    # torch.cuda.empty_cache() 
    # eval_model(model, criterion, optimizer, exp_lr_scheduler)
    pass

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)