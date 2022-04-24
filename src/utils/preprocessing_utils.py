import pandas as pd
import sys
import torch
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

sys.path.append('../')

from config.train_config import parse_train_configs

class GenreDataset(Dataset):
    """Genre dataset."""

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv = csv_file
        self.transform = transform

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.csv.iloc[idx, 0]
        image = cv2.imread(img_name,cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.expand_dims(image, axis=-1)
        details = self.csv.iloc[idx, 1:]
        sample = {'image': image, 'label': details[0]}

        if self.transform:
            sample = self.transform(sample)

        return sample

class PreProcessing(object):

    def __init__(self):
        self.scaler = MinMaxScaler()

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        h, w = image.shape[:2]

        ### ADD PREPROCESSING CODE HERE

        # Normalize if is RNN <Testing out>
        # image_2d = image[:,:,-1]
        # image_2d = self.scaler.fit_transform(image_2d)
        # image = np.expand_dims(image_2d, axis=2)
        
        return [torch.Tensor(image.transpose(2,0,1)), label]

def create_df(configs, csv):
    df = pd.read_csv(f"{configs.dataset_dir}/{csv}")
    df = df[['filename','label']]
    df = df[df['filename'] != "jazz.00054.wav"]
    df = df.reset_index()
    df.pop('index')

    class_name = {}
    n = 0
    for i in df['label'].unique():
        class_name[i] = n
        n+=1

    df['label'] = df['label'].map(class_name)

    for i in range(len(df)):
        temp = df['filename'][i].split(".")
        df['filename'][i] = configs.dataset_dir + "/images_original/" + temp[0] + "/" + temp[0] + temp[1] + ".png"
    
    return df

def create_dataloaders(configs, df):
    train, test = train_test_split(df, test_size=configs.test_size, random_state=42, stratify = df['label'])
    test, val = train_test_split(test, test_size=0.50, random_state=42, stratify = test['label'])
    dataset_sizes = {'train': len(train), 'test': len(test), 'val': len(val)}
    print(f"Dataset sizes: {dataset_sizes}")

    train_transformed_dataset = GenreDataset(csv_file=train,
                                               transform=transforms.Compose([
                                               PreProcessing()
                                           ]))
    test_transformed_dataset = GenreDataset(csv_file=test,
                                                transform=transforms.Compose([
                                                PreProcessing()
                                            ]))
    val_transformed_dataset = GenreDataset(csv_file=val,
                                                transform=transforms.Compose([
                                                PreProcessing()
                                            ]))

    dataloaders = {'train' : DataLoader(train_transformed_dataset, batch_size=configs.trg_batch_size,
                        shuffle=True, num_workers=0),
                    'test' : DataLoader(test_transformed_dataset, batch_size=configs.test_batch_size,
                                shuffle=True, num_workers=0),
                    'val' : DataLoader(val_transformed_dataset, batch_size=configs.val_batch_size,
                                shuffle=True, num_workers=0)}

    return dataloaders, dataset_sizes

class GenreDataset_embed(Dataset):
    """Genre dataset."""

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv = csv_file
        self.transform = transform

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.csv.iloc[idx, 0]
        image = cv2.imread(img_name,cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.expand_dims(image, axis=-1)
        details = self.csv.iloc[idx, 1:]
        sample = {'image': image, 'label': img_name}

        if self.transform:
            sample = self.transform(sample)

        return sample

def create_dataloaders_song_embedding(configs, df):

    transformed_dataset = GenreDataset_embed(csv_file=df,
                                               transform=transforms.Compose([
                                               PreProcessing()
                                           ]))

    dataloader = DataLoader(transformed_dataset, batch_size=configs.trg_batch_size,
                        shuffle=False, num_workers=0)

    return dataloader

if __name__ == "__main__":
    configs = parse_train_configs()
    df = create_df(configs, "features_30_sec.csv")
    print(df)
    dataloaders, dataset_sizes = create_dataloaders(configs, df)