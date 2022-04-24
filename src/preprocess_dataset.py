import sys
import librosa
import soundfile as sf
import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt
import cv2
from config.preprocess_dataset_config import parse_eval_configs
%matplotlib inline

def main():
    configs = parse_train_configs()
    
    loc = f"{configs.dataset_dir}/genres_original"
    for i in os.listdir(loc)[3:4]:
        for j in os.listdir(loc + "/" + i)[15:]:
            filename = loc + '/' + i + '/' + j
            audio, sr = librosa.load(filename)
            mel_signal = librosa.feature.melspectrogram(y=audio, sr=sr)
            spectrogram = np.abs(mel_signal)
            power_to_db = librosa.power_to_db(spectrogram, ref=np.max)
            librosa.display.specshow(power_to_db, sr=sr, x_axis='time', y_axis='mel', cmap='magma')
            plt.axis("off")
            plt.savefig(filename[:-4].replace('.','') + '.png')
            image = cv2.imread(filename[:-4].replace('.','') + '.png',cv2.IMREAD_COLOR)
            image = image[34:-35]
            image = image[:,52:-42]
            cv2.imwrite(filename[:-4].replace('.','') + '.png', image)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
   