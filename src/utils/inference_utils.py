import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import cv2
import numpy as np

import librosa
import soundfile as sf
import librosa, librosa.display

def inference(model, inputs, configs):

    labels = {
                0: "blues",
                1: "classical",
                2: "country",
                3: "disco",
                4: "hiphop",
                5: "jazz",
                6: "metal",
                7: "pop",    
                8: "reggae",
                9: "rock"
             }

    model.eval() # Set model to evaluate mode
        
    if configs.arch in ['rnn', 'gru', 'lstm']:
        # Reshape inputs to (batch_size, seq_length, input_size)
        inputs = inputs.reshape(-1, 339, 221).to(device=configs.device)
    else:
        inputs = inputs.to(device=configs.device)

    with torch.no_grad():
        output = model(inputs.float())
        genre_idx = int(torch.argmax(output))

    print(f"Genre of this song is: {labels[genre_idx]}")
    return output
    
def upload_spec(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    print(f"Original image size: {image.shape}")

    image = cv2.resize(image, (339,221))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.expand_dims(image, axis=-1)
    print(f"Reshaped image size: {image.shape}")

    inputs = np.transpose(image, (2,0,1))
    inputs = np.expand_dims(inputs, axis=0)
    inputs = torch.Tensor(inputs)
    
    return inputs

def audio_to_spec(path, configs):

    audio, sr = librosa.load(path, offset=20,duration=30)

    songname = path.split("/")[-1][:-4]

    # Get number of samples for 2 seconds; replace 2 by any number
    buffer = 30 * sr

    samples_total = len(audio)
    samples_wrote = 0
    counter = 1

    #check if the buffer is not exceeding total samples 
    if buffer > (samples_total - samples_wrote):
        buffer = samples_total - samples_wrote

    block = audio[samples_wrote : (samples_wrote + buffer)]
    out_filename = f"{configs.dataset_dir}/prediction_sounds/{songname}_cut.wav"

    # Write 30 second segment
    sf.write(out_filename, block, sr)
    counter += 1
    samples_wrote += buffer

    y, sr = librosa.load(out_filename)

    mel_signal = librosa.feature.melspectrogram(y=y, sr=sr)
    spectrogram = np.abs(mel_signal)
    power_to_db = librosa.power_to_db(spectrogram, ref=np.max)
    librosa.display.specshow(power_to_db, sr=sr, x_axis='time', y_axis='mel', cmap='magma')

    plt.axis('off')
    plt.savefig(f"{configs.dataset_dir}/prediction_mspec/{songname}_mspec.png", bbox_inches='tight', transparent=True, pad_inches=0)
    # plt.colorbar(label='dB')
    # plt.title('Mel-Spectrogram (dB)', fontdict=dict(size=18))
    # plt.xlabel('Time', fontdict=dict(size=15))
    # plt.ylabel('Frequency', fontdict=dict(size=15))
    # plt.show()
    return f"{songname}_mspec.png"

def songRecomendation(song_name, song_embed, new_song, model, configs, k=5):

    model.eval() # Set model to evaluate mode

    model = torch.nn.Sequential(*(list(model.modules())[1:])[0]) # strips off last linear layer and dropout layer

    new_song = new_song.to(configs.device)

    output = model(new_song.float())

    ls = np.dot(song_embed, output/np.linalg.norm(output)) #normalize query and matrix mult
    ls = sorted(range(len(ls)), key=lambda i: ls[i])[-k:]

    print("The suggested songs are : ")
    ls = [song_name[i] for i in ls]
    for i in range(len(ls)):
        print(str(i) + ') ' + ls[i])
    return ls