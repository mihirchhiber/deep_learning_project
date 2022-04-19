import torch
import sys

from config.eval_config import parse_eval_configs

from utils.model_utils import create_model
from utils.inference_utils import audio_to_spec, inference, upload_spec
from utils.preprocessing_utils import create_df, create_dataloaders

def main():
    configs = parse_eval_configs()
    df = create_df(configs, "features_30_sec.csv")
    print(df)
    # dataloaders, _ = create_dataloaders(configs, df)

    model = create_model(configs).to(device=configs.device)
    model.load_state_dict(torch.load(f"{configs.checkpoints_dir}/{configs.arch}_weights.pth"))

    path = f"{configs.dataset_dir}/prediction_sounds/jazz.00085.wav"
    song_spec = audio_to_spec(path, configs)
    inputs = upload_spec(f"{configs.dataset_dir}/prediction/{song_spec}")

    genre = inference(model, inputs, configs)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)