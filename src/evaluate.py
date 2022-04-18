import torch
import sys

from config.eval_config import parse_eval_configs

from utils.model_utils import create_model
from utils.eval_utils import eval_model
from utils.preprocessing_utils import create_df, create_dataloaders

def main():
    configs = parse_eval_configs()
    df = create_df(configs, "features_30_sec.csv")
    print(df)
    dataloaders, _ = create_dataloaders(configs, df)

    model = create_model(configs).to(device=configs.device)
    model.load_state_dict(torch.load(f"{configs.checkpoints_dir}/cnn_weights.pth"))

    eval_model(model, dataloaders, configs)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)