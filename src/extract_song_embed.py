import sys
import torch

from config.song_embed_config import parse_eval_configs

from utils.model_utils import create_model
from utils.extract_song_embed import extract_song_embed_from_model
from utils.preprocessing_utils import create_df, create_dataloaders_song_embedding


def main():
    configs = parse_eval_configs()
    df = create_df(configs, "features_30_sec.csv")
    print(df)
    dataloaders = create_dataloaders_song_embedding(configs, df)

    model = create_model(configs).to(device=configs.device)
    
    model.load_state_dict(torch.load(f"{configs.checkpoints_dir}/cnn_weights_best.pth", map_location=configs.device))

    extract_song_embed_from_model(model, dataloaders, configs)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
