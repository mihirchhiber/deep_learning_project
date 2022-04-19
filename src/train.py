import torch.nn as nn
from torch.optim import lr_scheduler
import sys

from config.train_config import parse_train_configs

from utils.model_utils import create_model, create_optimizer
from utils.train_utils import train_model, plot_performance
from utils.preprocessing_utils import create_df, create_dataloaders

def main():
    configs = parse_train_configs()
    df = create_df(configs, "features_30_sec.csv")
    print(df)
    dataloaders, dataset_sizes = create_dataloaders(configs, df)

    model = create_model(configs).to(device=configs.device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(configs, model)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=configs.step_size, gamma=configs.gamma)

    model, losses, accuracies = train_model(model, dataloaders, dataset_sizes, configs,
                                    criterion, optimizer, exp_lr_scheduler, num_epochs=configs.num_epochs,
                                    patience=configs.patience)    

    plot_performance('Accuracy', accuracies)
    plot_performance('Loss', losses)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
   