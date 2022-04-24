import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pickle
import numpy as np

def extract_song_embed_from_model(model, dataloaders, configs):

    model.eval() # Set model to evaluate mode

    my_model = torch.nn.Sequential(*(list(model.modules())[1:])[0]) # strips off last linear layer and dropout layer

    def song_encoder(model):
                    
        model.eval()   # Set model to evaluate mode

        song_name = []
        song_encoded = []

        # Iterate over data.
        for inputs, labels in dataloaders:
            
            inputs = inputs.to(configs.device)

            outputs = model(inputs.float())
            temp = outputs.cpu().detach().numpy()
            song_name += list(labels)
            for i in temp:
                song_encoded.append(i/np.linalg.norm(i))

        return song_name, song_encoded

    song_name, song_encoded = song_encoder(my_model)
    song_embed = np.array(song_encoded)
    with open(f'{configs.checkpoints_dir}/song_embeddings/song_embed.pkl', 'wb') as f:
        pickle.dump(song_embed, f)
    with open(f'{configs.checkpoints_dir}/song_embeddings/song_name.pkl', 'wb') as f:
        pickle.dump(song_name, f)

    # return song_name, song_embed