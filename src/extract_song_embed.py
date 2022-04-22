import sys

from config.song_embed_config import parse_eval_configs

from utils.model_utils import create_model
from utils.extract_song_embed import extract_song_embed_from_model
from utils.preprocessing_utils import create_df, create_dataloaders


def main():
    configs = parse_eval_configs()
    df = create_df(configs, "features_30_sec.csv")
    print(df)
    dataloaders, _ = create_dataloaders(configs, df)

    model = create_model(configs).to(device=configs.device)

    extract_song_embed_from_model(model, dataloaders, configs)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)