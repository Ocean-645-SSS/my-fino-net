"""
FinoNet RGB-D-A: Multi-modal Deep Learning for Action Recognition
Combines RGB, Depth, and Audio modalities using ConvLSTM and CNN architectures
"""

import os
import glob
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from convlstm import ConvLSTM

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.model_selection import train_test_split

import librosa
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
from keras_preprocessing import sequence

# ==================== Configuration Parameters ====================

param_dict = {
    'img_path': "/root/autodl-tmp/failnet_dataset/rgb_imgs/",
    'depth_path': "/root/autodl-tmp/failnet_dataset/depth_imgs/",
    'audio_path': "/root/autodl-tmp/failnet_dataset/audio/",
    'img_size': 224,

    'sampling_mode': "depth",
    'n_sample_frames': 8,
    'depth_thresh': 500,

    'lr': 0.000001,
    'n_epoch': 1000,
    'batch_size': 16,

    'batch_norm': True,
    'normalize': False,

    'dropout': True,
    'dropout_p': 0.4,

    'jitter': True,
    'j_brightness': 0.2,
    'j_contrast': 0.2,
    'j_saturation': 0.2,
    'j_hue': 0.2,

    'random_crop': False,
    'flip': True,

    "freeze_rgbd": False,
    "freeze_arm": True,
}

device = "cuda" if torch.cuda.is_available() else "cpu"


# ==================== Data Loading and Preprocessing ====================

def load_data():
    """Load and organize all data from directories"""
    data = []
    id_counter = 0

    action_list = next(os.walk(param_dict['img_path']))[1]

    for action in action_list:
        # Read Label file
        annotation_df = pd.read_csv(os.path.join(param_dict["img_path"], action, "annotation.txt"))
        #annotation_df["name"] = np.array([int(item[0]) for item in annotation_df["name"].str.split("_").tolist()])
        annotation_df["name"] = annotation_df["name"].astype(int)

        # Get bags list of the current action
        action_path = os.path.join(param_dict['img_path'], action)
        bags = next(os.walk(action_path))[1]

        for bag in bags:
            bag_label = int(annotation_df[annotation_df["name"] == int(bag)]["label"]).iloc[0]
            #todo:这里要修改
            fail_timestamp = 0

            bag_path = os.path.join(action_path, bag)

            bag_imgs_path = glob.glob(os.path.join(bag_path, "*.png"))
            bag_imgs_path.sort()
            bag_stamps_path = glob.glob(os.path.join(bag_path, "*.txt"))
            bag_stamps_path.sort()

            bag_content = []

            for img_path, stamp_path in zip(bag_imgs_path, bag_stamps_path):
                # Read time stamp of the frame from file
                timestamp = float(open(stamp_path, "r").read())

                depth_img_path = os.path.join(param_dict['depth_path'],
                                              action,
                                              bag,
                                              os.path.basename(img_path).split(".")[0] + ".tiff")

                d_img = Image.open(depth_img_path)
                d_mean = np.mean(d_img)
                d_img.close()

                bag_content.append({
                    'img_id': os.path.basename(img_path),
                    'timestamp': timestamp,
                    'img_path': img_path,
                    'img_label': bag_label & int((timestamp >= fail_timestamp)),
                    'depth_path': depth_img_path,
                    'depth_avg': d_mean,
                })

            if os.path.isfile(os.path.join(param_dict['audio_path'], action, bag + ".wav")):
                data.append({
                    'unique_id': id_counter,
                    'action': action,
                    'bag_no': int(bag),
                    'bag_label': int(bag_label),
                    'bag_content': bag_content,
                    'audio_path': os.path.join(param_dict['audio_path'], action, bag + ".wav"),
                })
                id_counter += 1

    return pd.DataFrame.from_dict(data)


def get_train_test_split(df):
    """Split data into train and test sets stratified by action"""
    tr_succ, tr_fail, te_succ, te_fail = [], [], [], []

    for action in df["action"].unique():
        succ = df.loc[(df['action'] == action) & (df['bag_label'] == 0)]["unique_id"].values
        fail = df.loc[(df['action'] == action) & (df['bag_label'] == 1)]["unique_id"].values

        X_tr, X_te = train_test_split(succ, test_size=0.3, shuffle=True, random_state=42)
        tr_succ = np.concatenate((tr_succ, X_tr), axis=0)
        te_succ = np.concatenate((te_succ, X_te), axis=0)

        X_tr, X_te = train_test_split(fail, test_size=0.3, shuffle=True, random_state=42)
        tr_fail = np.concatenate((tr_fail, X_tr), axis=0)
        te_fail = np.concatenate((te_fail, X_te), axis=0)

    train_idx = np.concatenate((tr_succ, tr_fail), axis=0)
    test_idx = np.concatenate((te_succ, te_fail), axis=0)

    return train_idx, test_idx


# ==================== Dataset Class ====================

# Image transformations
tf_list = []
if param_dict["random_crop"]:
    tf_list.append(transforms.RandomCrop((224, 224)))
tf_list.append(transforms.ToTensor())
if param_dict["normalize"]:
    tf_list.append(transforms.Normalize([0.47264798, 0.47641314, 0.46798028],
                                        [0.07805742, 0.0770264, 0.08050214]))
tf = transforms.Compose(tf_list)


class DatasetFD(Dataset):
    """Dataset class for RGB-D and Audio data"""

    def __init__(self, data, indices, n_sample_frames, sampling_mode):
        self.data = data
        self.n_sample_frames = n_sample_frames
        self.sampling_mode = sampling_mode
        self.data = self.data.iloc[indices]

    def __len__(self):
        return len(self.data)

    def _get_frame_indices(self, index):
        frames = self.data.iloc[index]["bag_content"]
        n_frames = len(frames)

        if self.sampling_mode == "depth":
            df_depth = pd.DataFrame(frames)

            def _get_indices(df_depth):
                a = df_depth["depth_avg"].values > param_dict["depth_thresh"]
                frame_idx = np.where(a == True)[0]
                n_frames = len(frame_idx)

                segments = np.split(frame_idx, [int(n_frames / 3), int(n_frames / 3) * 2])
                first_segment_idx = np.sort(np.random.permutation(segments[0])[:4])
                last_segment_idx = np.sort(np.random.permutation(segments[2])[:4])
                frame_idx = np.concatenate((first_segment_idx, last_segment_idx), 0)

                return frame_idx

            frame_idx = _get_indices(df_depth)

        return frame_idx

    def load_img(self, img_path, vertical_flip, color_transform):
        if param_dict["random_crop"]:
            border = (210, 150, 454, 394)  # left, up, right, bottom
        else:
            border = (220, 160, 444, 384)  # left, up, right, bottom [224x224]

        img = Image.open(img_path).crop(border)

        if "depth" in img_path:
            img = img / np.max(img)
            img = Image.fromarray(np.uint8(img * 255))

        if vertical_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        if color_transform is not None:
            img = color_transform(img)

        imgarr = tf(img)
        img.close()
        return imgarr

    def load_img_arm(self, img_path, vertical_flip, color_transform):
        """Processing for arm images"""
        img = Image.open(img_path).crop((96, 16, 544, 464))
        img = img.resize((224, 224))

        if vertical_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        if color_transform is not None:
            img = color_transform(img)

        imgarr = tf(img)
        img.close()
        return imgarr

    def load_mfcc(self, wav_path):
        sample_rate, samples1 = wavfile.read(wav_path)

        if samples1.ndim > 1:
            mfcc = librosa.feature.mfcc(y=samples1[:, 0].astype(np.float32), sr=sample_rate).T
        else:
            mfcc = librosa.feature.mfcc(y=samples1.astype(np.float32), sr=sample_rate).T

        mfcc = sequence.pad_sequences([mfcc], maxlen=3500)[0]
        mfcc = torch.Tensor(mfcc).float()
        return mfcc

    def __getitem__(self, index):
        frames = self.data.iloc[index]["bag_content"]
        label = self.data.iloc[index]["bag_label"]

        frame_idx = self._get_frame_indices(index)

        if param_dict["flip"]:
            vert_flip = np.random.randint(2)
        else:
            vert_flip = False

        if param_dict["jitter"]:
            color_jitter = transforms.ColorJitter(brightness=param_dict["j_brightness"],
                                                  contrast=param_dict["j_contrast"],
                                                  saturation=param_dict["j_saturation"],
                                                  hue=param_dict["j_hue"])
            color_transform = transforms.ColorJitter.get_params(
                color_jitter.brightness, color_jitter.contrast, color_jitter.saturation,
                color_jitter.hue)
        else:
            color_transform = None

        img_batch = torch.stack([self.load_img(frames[idx]["img_path"], vert_flip, color_transform)
                                 for idx in frame_idx])
        depthimg_batch = torch.stack([self.load_img(frames[idx]["depth_path"], vert_flip, color_transform)
                                      for idx in frame_idx])

        img_batch = torch.cat((img_batch, depthimg_batch), axis=1)

        # Audio
        wav_path = self.data.iloc[index]["audio_path"]
        audio_batch = self.load_mfcc(wav_path)

        return self.data.iloc[index]["unique_id"], frame_idx, img_batch, audio_batch, label


# ==================== Model Definitions ====================

class FinoNetRGBD(nn.Module):
    """RGB-D branch with ConvLSTM layers"""

    def __init__(self):
        super(FinoNetRGBD, self).__init__()

        self.num_filters1 = 64
        self.num_filters2 = 128
        self.num_filters3 = 128

        self.bn1 = nn.BatchNorm2d(self.num_filters1)
        self.bn11 = nn.BatchNorm2d(self.num_filters1)
        self.bn2 = nn.BatchNorm2d(self.num_filters2)
        self.bn22 = nn.BatchNorm2d(self.num_filters2)
        self.bn3 = nn.BatchNorm2d(self.num_filters3)
        self.bn33 = nn.BatchNorm2d(self.num_filters3)

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=self.num_filters1, kernel_size=3)
        self.conv11 = nn.Conv2d(in_channels=self.num_filters1, out_channels=self.num_filters1, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.relu11 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.convlstm1 = ConvLSTM(input_dim=self.num_filters1,
                                  hidden_dim=[self.num_filters1],
                                  kernel_size=(3, 3),
                                  num_layers=1,
                                  batch_first=True,
                                  bias=True,
                                  return_all_layers=False)

        self.conv2 = nn.Conv2d(in_channels=self.num_filters1, out_channels=self.num_filters2, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.conv22 = nn.Conv2d(in_channels=self.num_filters2, out_channels=self.num_filters2, kernel_size=3)
        self.relu22 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.convlstm2 = ConvLSTM(input_dim=self.num_filters2,
                                  hidden_dim=[self.num_filters2],
                                  kernel_size=(3, 3),
                                  num_layers=1,
                                  batch_first=True,
                                  bias=True,
                                  return_all_layers=False)

        self.conv3 = nn.Conv2d(in_channels=self.num_filters2, out_channels=self.num_filters3, kernel_size=3)
        self.relu3 = nn.ReLU()
        self.conv33 = nn.Conv2d(in_channels=self.num_filters3, out_channels=self.num_filters3, kernel_size=3)
        self.relu33 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.convlstm3 = ConvLSTM(input_dim=self.num_filters3,
                                  hidden_dim=[self.num_filters3],
                                  kernel_size=(3, 3),
                                  num_layers=1,
                                  batch_first=True,
                                  bias=True,
                                  return_all_layers=False)

    def forward(self, x):
        # BLOCK 1
        batch_size, frame_size, channel, height, width = x.size()
        x_in1 = x.view(batch_size * frame_size, channel, height, width)
        x = self.conv1(x_in1)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv11(x)
        x = self.bn11(x)
        x = self.relu11(x)
        x = self.pool1(x)
        x = F.dropout(x, param_dict["dropout_p"])

        x = x.view(batch_size, frame_size, self.num_filters1, x.size()[-1], x.size()[-1])
        x = self.convlstm1(x)[0][0]
        batch_size, frame_size, channel, height, width = x.size()
        x = x.view(batch_size * frame_size, channel, height, width)
        x_in2 = F.dropout(x, param_dict["dropout_p"])

        # BLOCK 2
        x = self.conv2(x_in2)
        x = self.bn2(x)
        x = self.relu2(x)
        x = F.dropout(x, param_dict["dropout_p"])

        x = self.conv22(x)
        x = self.bn22(x)
        x = self.relu22(x)
        x = self.pool2(x)
        x = F.dropout(x, param_dict["dropout_p"])

        x = x.view(batch_size, frame_size, self.num_filters2, x.size()[-1], x.size()[-1])
        x = self.convlstm2(x)[0][0]
        batch_size, frame_size, channel, height, width = x.size()
        x = x.view(batch_size * frame_size, channel, height, width)
        x_in3 = F.dropout(x, param_dict["dropout_p"])

        # BLOCK 3
        x = self.conv3(x_in3)
        x = self.bn3(x)
        x = self.relu3(x)
        x = F.dropout(x, param_dict["dropout_p"])

        x = self.conv33(x)
        x = self.bn33(x)
        x = self.relu33(x)
        x = self.pool3(x)
        x = F.dropout(x, param_dict["dropout_p"])

        x = x.view(batch_size, frame_size, self.num_filters3, x.size()[-1], x.size()[-1])
        x = self.convlstm3(x)[0][0]

        batch_size, frame_size, channel, height, width = x.size()
        x = x.view(batch_size * frame_size, channel, height, width)
        x = x.view(batch_size, frame_size, self.num_filters3, x.size()[-1], x.size()[-1])

        x = x[:, -1, :]  # get last frame features
        x = x.view(batch_size, -1)
        return x


class AudioCNN(nn.Module):
    """Audio branch with CNN"""

    def __init__(self):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, (32, 20))
        self.conv2 = nn.Conv1d(64, 64, (32, 1))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_2blocks(x)
        return x

    def conv_2blocks(self, input):
        conv_out = self.conv1(input)
        conv_out = F.relu(conv_out)
        conv_out = self.conv2(conv_out)
        activation = F.relu(conv_out.squeeze(3))
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)
        return max_out


class FinonetRGBDA(nn.Module):
    """Multi-modal fusion network combining RGB-D and Audio"""

    def __init__(self):
        super(FinonetRGBDA, self).__init__()

        self.rgbd = FinoNetRGBD()
        #self.rgbd.load_state_dict(torch.load("finonet-rgbd.pth"), strict=False)

        self.audio = AudioCNN()

        if param_dict["freeze_rgbd"]:
            self.rgbd.module.conv1.requires_grad = False
            self.rgbd.module.conv11.requires_grad = False
            self.rgbd.module.conv2.requires_grad = False
            self.rgbd.module.conv22.requires_grad = False
            self.rgbd.module.conv3.requires_grad = False
            self.rgbd.module.conv33.requires_grad = False
            self.rgbd.module.convlstm1.cell_list[0].requires_grad = False
            self.rgbd.module.convlstm2.cell_list[0].requires_grad = False

        self.HIDDEN_SIZE = 128 * 24 * 24
        self.AUDIO_SIZE = 64

        self.linear1 = nn.Linear(self.HIDDEN_SIZE + self.AUDIO_SIZE, 1024)
        self.linear2 = nn.Linear(1024, 2)

        if param_dict["dropout"]:
            self.dropout = nn.Dropout(p=param_dict["dropout_p"])
            self.dropout2 = nn.Dropout(p=param_dict["dropout_p"])

        self.mp = nn.MaxPool1d(2)

    def forward(self, x, x_audio):
        x = self.rgbd(x)
        x_audio = self.audio(x_audio)
        x = torch.cat((x, x_audio), 1)

        if param_dict["dropout"]:
            x = F.relu(self.linear1(x))
            x = self.linear2(self.dropout2(x))
            return x
        else:
            return self.linear(x)


# ==================== Training Functions ====================

def test_epoch(model, test_loader, test_dataset, epoch):
    """Run one test epoch"""
    model.eval()

    test_loss = 0
    all_labels, all_preds = [], []

    for idx, frame_idx, imgs, audio, labels in test_loader:
        all_labels = np.concatenate((all_labels, labels.cpu().data.numpy()), axis=0)

        imgs = imgs.to(device)
        audio = audio.to(device)
        labels = labels.to(device)

        output = model(imgs, audio)
        loss = F.cross_entropy(output, labels)

        test_loss += loss.item()
        values, indices = torch.max(torch.softmax(output, dim=1), 1)
        all_preds = np.concatenate((all_preds, indices.cpu().data.numpy()), axis=0)

    test_loss = test_loss / len(test_dataset)
    test_acc = f1_score(all_labels, all_preds, average='weighted')

    print("[Test] Epoch: {}, Loss: {} Acc: {}".format(epoch, test_loss, test_acc))
    return test_acc


def train_epoch(model, train_loader, train_dataset, optimizer, epoch):
    """Run one training epoch"""
    model.train()

    epoch_loss = 0
    all_labels, all_preds = [], []

    for idx, frame_idx, imgs, audio, labels in train_loader:
        imgs = imgs.to(device)
        audio = audio.to(device)
        labels = labels.to(device)

        all_labels = np.concatenate((all_labels, labels.cpu().data.numpy()), axis=0)

        optimizer.zero_grad()

        output = model(imgs, audio)
        loss = F.cross_entropy(output, labels)

        values, indices = torch.max(torch.softmax(output, dim=1), 1)
        all_preds = np.concatenate((all_preds, indices.cpu().data.numpy()), axis=0)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss = epoch_loss / len(train_dataset)
    tr_acc = f1_score(all_labels, all_preds, average='weighted')

    print("[Train] Epoch: {}, Loss: {} Acc: {}".format(epoch, epoch_loss, tr_acc))
    return tr_acc


# ==================== Main Training Loop ====================

def main():
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} samples")

    # Split data
    train_idx, test_idx = get_train_test_split(df)
    print(f"Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")

    # Create data loaders
    train_dataset = DatasetFD(df, train_idx, param_dict["n_sample_frames"], param_dict["sampling_mode"])
    train_loader = DataLoader(train_dataset, batch_size=param_dict["batch_size"], shuffle=True, num_workers=16)

    test_dataset = DatasetFD(df, test_idx, param_dict["n_sample_frames"], param_dict["sampling_mode"])
    test_loader = DataLoader(test_dataset, batch_size=param_dict["batch_size"], shuffle=False, num_workers=16)

    # Initialize model
    model = FinonetRGBDA()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=param_dict["lr"])

    # Training loop
    for epoch in range(param_dict["n_epoch"]):
        train_epoch(model, train_loader, train_dataset, optimizer, epoch)
        test_epoch(model, test_loader, test_dataset, epoch)


if __name__ == "__main__":
    main()