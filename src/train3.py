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

# 修改为直接扫描文件夹
def load_data():
    all_data = []
    actions = ['place', 'pour', 'push', 'put_in', 'put_on']

    for action in actions:
        file_path = os.path.join(param_dict['img_path'], action, "annotation.txt")

        if os.path.exists(file_path):
            df_temp = pd.read_csv(file_path)
            for _, row in df_temp.iterrows():
                folder_rel_path = os.path.join(action, str(int(row['name'])))

                #定位到图片文件夹
                rgb_dir = os.path.join(param_dict['img_path'], folder_rel_path)

                # 检查文件夹是否存在
                if not os.path.exists(rgb_dir):
                    # print(f"跳过不存在的文件夹: {rgb_dir}")
                    continue

                # 检查里面是否有图片,同时支持 .jpg 和 .png）
                img_files = glob.glob(os.path.join(rgb_dir, "*.jpg")) + glob.glob(os.path.join(rgb_dir, "*.png"))
                if len(img_files) == 0:
                    # print(f"跳过空文件夹: {rgb_dir}")
                    continue

                all_data.append({
                    'folder_rel_path': folder_rel_path,
                    'action': action,
                    'label': int(row['label'])
                })
            print(f"成功加载【{action}】的标注文件！")
        else:
            print(f"路径错误，未找到：{file_path}")

    final_df = pd.DataFrame(all_data)
    print(f"\n数据加载完成！过滤后共计 {len(final_df)} 条有效视频片段样本。")
    return final_df


def get_train_test_split(df):
    from sklearn.model_selection import train_test_split
    # 获取索引
    indices = np.arange(len(df))
    # 按照 label 进行分层抽样，确保训练集和测试集都有成功和失败样本
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=df['label'])
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
    def __init__(self, df, indices, n_sample_frames=8, sampling_mode="depth"):
        self.df = df.iloc[indices].reset_index(drop=True)
        self.n_sample_frames = n_sample_frames
        self.sampling_mode = sampling_mode
        self.img_size = param_dict['img_size']

        # 标准图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])

    def _get_sampling_indices(self, total_frames):
        """
        [改进点1] 均匀采样：解决 start 缺失问题
        把视频全时长等分为n份，确保覆盖动作始末
        """
        if total_frames <= 0:
            return [0] * self.n_sample_frames
        # np.linspace 确保索引在 [0, total-1] 之间均匀分布
        indices = np.linspace(0, total_frames - 1, self.n_sample_frames, dtype=int)
        return indices

    def _load_mfcc(self, audio_path):
        """
        [改进点2] 音频预处理逻辑：解决 RuntimeError 维度报错
        """
        try:
            # 使用 librosa 加载，统一采样率 16k
            y, sr = librosa.load(audio_path, sr=16000)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

            # 定长对齐：确保输出维度始终为 (13, 128)
            target_len = 128
            if mfcc.shape[1] < target_len:
                mfcc = np.pad(mfcc, ((0, 0), (0, target_len - mfcc.shape[1])), mode='constant')
            else:
                mfcc = mfcc[:, :target_len]
            return torch.FloatTensor(mfcc)
        except Exception as e:
            # 如果音频损坏，返回静音特征张量
            return torch.zeros((13, 128))

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            # 获取文件夹列表
            action_folders = [
                'place', 'pour', 'push', 'put_in', 'put_on'
            ]
            folder_name = action_folders[int(row['name']) % len(action_folders)]
            label = int(row['label'])

            # 1. 获取 RGB 图像列表
            rgb_dir = os.path.join(param_dict['img_path'], folder_name)
            rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.jpg")) +
                               glob.glob(os.path.join(rgb_dir, "*.png")))

            if len(rgb_files) == 0:
                raise FileNotFoundError(f"No images in {rgb_dir}")

            # 2. 调用均匀采样8帧
            sample_indices = self._get_sampling_indices(len(rgb_files))

            # 3. 读取并拼接 RGB-D 帧 (4通道)
            frames = []
            for i in sample_indices:
                # RGB
                img = Image.open(rgb_files[i]).convert('RGB')

                # Depth (尝试找对应文件，找不到补黑图)
                depth_file = rgb_files[i].replace('rgb_imgs', 'depth_imgs')
                if os.path.exists(depth_file):
                    d_img = Image.open(depth_file).convert('L')
                else:
                    d_img = Image.new('L', (self.img_size, self.img_size))

                # 拼接成 (4, H, W)
                combined = torch.cat([self.transform(img), self.transform(d_img)], dim=0)
                frames.append(combined)

            # 4. 加载音频
            audio_file = os.path.join(param_dict['audio_path'], rel_path + ".wav")
            mfcc_tensor = self._load_mfcc(audio_file)

            # 返回 (T, C, H, W) 形状的视频张量、音频张量和标签
            return torch.stack(frames), mfcc_tensor, torch.tensor(label, dtype=torch.long)

        except Exception as e:
            # 一旦出错，随机换一个 index 重新加载，直到成功
            print(f"Warning: Loading error at index {idx}, swapping... Error: {e}")
            new_idx = np.random.randint(0, len(self.df))
            return self.__getitem__(new_idx)

    def __len__(self):
        return len(self.df)


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