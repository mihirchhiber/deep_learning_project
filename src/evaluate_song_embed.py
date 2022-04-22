import sys
import pickle

from config.song_embed_config import parse_eval_configs

from utils.eval_utils import eval_embed

def main():
    
    with open('checkpoints/song_embeddings/song_embed.pkl', 'rb') as f:
        song_embed = pickle.load(f)
    with open('checkpoints/song_embeddings/song_name.pkl', 'rb') as f:
        song_name = pickle.load(f)

    eval_embed(song_name, song_embed)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)