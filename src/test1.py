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
from scipy.io import wavfile
from keras_preprocessing import sequence

# ==========================================
# 1. 配置参数
# ==========================================
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
    'batch_size': 2,

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


# ==========================================
# 2. 数据预处理与读取
# ==========================================
def prepare_data_list():
    data = []
    id_counter = 0

    if not os.path.exists(param_dict['img_path']):
        print(f"警告: 路径 {param_dict['img_path']} 不存在。")
        return pd.DataFrame()

    action_list = next(os.walk(param_dict['img_path']))[1]

    for action in action_list:
        # 读取标签文件
        anno_path = os.path.join(param_dict["img_path"], action, "annotation.txt")
        if not os.path.exists(anno_path): continue

        annotation_df = pd.read_csv(anno_path)
        # 修改1:annotation_df["name"] = np.array([int(item[0]) for item in annotation_df["name"].str.split("_").tolist()])
        annotation_df["name"] = annotation_df["name"].astype(int)

        action_path = os.path.join(param_dict['img_path'], action)
        bags = next(os.walk(action_path))[1]

        for bag in bags:
            # 修改4：
            bag=int(bag)
            matching_row = annotation_df[annotation_df["name"] == int(bag)]
            if matching_row.empty: continue
            # 修改4：
            row = matching_row.iloc[0]

            bag_label = int(matching_row["label"])
            fail_timestamp = float(matching_row["start"])

            bag_path = os.path.join(action_path, bag)
            bag_imgs_path = sorted(glob.glob(os.path.join(bag_path, "*.png")))
            bag_stamps_path = sorted(glob.glob(os.path.join(bag_path, "*.txt")))

            bag_content = []
            for img_path, stamp_path in zip(bag_imgs_path, bag_stamps_path):
                with open(stamp_path, "r") as f:
                    timestamp = float(f.read())

                depth_img_path = os.path.join(param_dict['depth_path'],
                                              action,
                                              bag,
                                              os.path.basename(img_path).split(".")[0] + ".tiff")

                if os.path.exists(depth_img_path):
                    d_img = Image.open(depth_img_path)
                    d_mean = np.mean(d_img)
                    d_img.close()
                else:
                    d_mean = 0

                bag_content.append({
                    'img_id': os.path.basename(img_path),
                    'timestamp': timestamp,
                    'img_path': img_path,
                    'img_label': bag_label & int((timestamp >= fail_timestamp)),
                    'depth_path': depth_img_path,
                    'depth_avg': d_mean,
                })

            audio_file = os.path.join(param_dict['audio_path'], action, bag + ".wav")
            if os.path.isfile(audio_file):
                data.append({
                    'unique_id': id_counter,
                    'action': action,
                    'bag_no': int(bag),
                    'bag_label': int(bag_label),
                    'bag_content': bag_content,
                    'audio_path': audio_file,
                })
                id_counter += 1
    return pd.DataFrame.from_dict(data)


def get_all_data_splits(df):
    tr_succ, tr_fail, te_succ, te_fail = [], [], [], []
    for action in df["action"].unique():
        succ = df.loc[(df['action'] == action) & (df['bag_label'] == 0)]["unique_id"].values
        fail = df.loc[(df['action'] == action) & (df['bag_label'] == 1)]["unique_id"].values

        if len(succ) > 1:
            X_tr, X_te = train_test_split(succ, test_size=0.3, shuffle=True, random_state=42)
            tr_succ = np.concatenate((tr_succ, X_tr), axis=0)
            te_succ = np.concatenate((te_succ, X_te), axis=0)

        if len(fail) > 1:
            X_tr, X_te = train_test_split(fail, test_size=0.3, shuffle=True, random_state=42)
            tr_fail = np.concatenate((tr_fail, X_tr), axis=0)
            te_fail = np.concatenate((te_fail, X_te), axis=0)

    return tr_succ.astype(int), tr_fail.astype(int), te_succ.astype(int), te_fail.astype(int)


# ==========================================
# 3. Dataset 类定义
# ==========================================
tf_list = []
if param_dict["random_crop"]:
    tf_list.append(transforms.RandomCrop((224, 224)))
tf_list.append(transforms.ToTensor())
if param_dict["normalize"]:
    tf_list.append(transforms.Normalize([0.47264798, 0.47641314, 0.46798028], [0.07805742, 0.0770264, 0.08050214]))
tf_comp = transforms.Compose(tf_list)


class DatasetFD(Dataset):
    def __init__(self, data, indices, n_sample_frames, sampling_mode):
        self.data = data.iloc[indices].reset_index(drop=True)
        self.n_sample_frames = n_sample_frames
        self.sampling_mode = sampling_mode

    def __len__(self):
        return len(self.data)

    def _get_frame_indices(self, index):
        frames = self.data.iloc[index]["bag_content"]
        if self.sampling_mode == "depth":
            df_depth = pd.DataFrame(frames)
            a = df_depth["depth_avg"].values > param_dict["depth_thresh"]
            frame_idx = np.where(a == True)[0]
            n_f = len(frame_idx)
            if n_f < 8:  # 防止样本太少
                return np.linspace(0, len(frames) - 1, 8).astype(int)
            segments = np.split(frame_idx, [int(n_f / 3), int(n_f / 3) * 2])
            first_segment_idx = np.sort(np.random.permutation(segments[0])[:4])
            last_segment_idx = np.sort(np.random.permutation(segments[2])[:4])
            return np.concatenate((first_segment_idx, last_segment_idx), 0)
        return np.arange(8)

    def load_img(self, img_path, vertical_flip, color_transform):
        border = (210, 150, 454, 394) if param_dict["random_crop"] else (220, 160, 444, 384)
        img = Image.open(img_path).crop(border)

        if "depth" in img_path:
            img_arr = np.array(img).astype(float)
            img_max = np.max(img_arr)
            if img_max > 0: img_arr /= img_max
            img = Image.fromarray(np.uint8(img_arr * 255))

        if vertical_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if color_transform is not None:
            img = color_transform(img)
        imgarr = tf_comp(img)
        img.close()
        return imgarr

    def load_mfcc(self, wav_path):
        sample_rate, samples1 = wavfile.read(wav_path)
        y = samples1[:, 0].astype(np.float32) if samples1.ndim > 1 else samples1.astype(np.float32)
        mfcc = librosa.feature.mfcc(y=y, sr=sample_rate).T
        mfcc = sequence.pad_sequences([mfcc], maxlen=3500)[0]
        return torch.Tensor(mfcc).float()

    def __getitem__(self, index):
        frames = self.data.iloc[index]["bag_content"]
        label = self.data.iloc[index]["bag_label"]
        frame_idx = self._get_frame_indices(index)
        vert_flip = np.random.randint(2) if param_dict["flip"] else False

        color_transform = None
        if param_dict["jitter"]:
            cj = transforms.ColorJitter(brightness=param_dict["j_brightness"], contrast=param_dict["j_contrast"],
                                        saturation=param_dict["j_saturation"], hue=param_dict["j_hue"])
            color_transform = transforms.ColorJitter.get_params(cj.brightness, cj.contrast, cj.saturation, cj.hue)

        img_batch = torch.stack(
            [self.load_img(frames[idx]["img_path"], vert_flip, color_transform) for idx in frame_idx])
        depthimg_batch = torch.stack(
            [self.load_img(frames[idx]["depth_path"], vert_flip, color_transform) for idx in frame_idx])
        img_batch = torch.cat((img_batch, depthimg_batch), axis=1)

        audio_batch = self.load_mfcc(self.data.iloc[index]["audio_path"])
        return self.data.iloc[index]["unique_id"], frame_idx, img_batch, audio_batch, label


# ==========================================
# 4. 模型架构定义
# ==========================================
class FinoNetRGBD(nn.Module):
    def __init__(self):
        super(FinoNetRGBD, self).__init__()
        self.num_filters1, self.num_filters2, self.num_filters3 = 64, 128, 128

        self.bn1, self.bn11 = nn.BatchNorm2d(64), nn.BatchNorm2d(64)
        self.bn2, self.bn22 = nn.BatchNorm2d(128), nn.BatchNorm2d(128)
        self.bn3, self.bn33 = nn.BatchNorm2d(128), nn.BatchNorm2d(128)

        self.conv1 = nn.Conv2d(4, 64, kernel_size=3)
        self.conv11 = nn.Conv2d(64, 64, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.convlstm1 = ConvLSTM(input_dim=64, hidden_dim=[64], kernel_size=(3, 3), num_layers=1, batch_first=True)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.convlstm2 = ConvLSTM(input_dim=128, hidden_dim=[128], kernel_size=(3, 3), num_layers=1, batch_first=True)

        self.conv3 = nn.Conv2d(128, 128, kernel_size=3)
        self.conv33 = nn.Conv2d(128, 128, kernel_size=3)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.convlstm3 = ConvLSTM(input_dim=128, hidden_dim=[128], kernel_size=(3, 3), num_layers=1, batch_first=True)

    def forward(self, x):
        b, f, c, h, w = x.size()
        x = x.view(b * f, c, h, w)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn11(self.conv11(x)))
        x = self.pool1(x)
        x = F.dropout(x, param_dict["dropout_p"])

        x = x.view(b, f, self.num_filters1, x.size()[-2], x.size()[-1])
        x = self.convlstm1(x)[0][0]
        x = x.view(b * f, self.num_filters1, x.size()[-2], x.size()[-1])
        x = F.dropout(x, param_dict["dropout_p"])

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn22(self.conv22(x)))
        x = self.pool2(x)
        x = F.dropout(x, param_dict["dropout_p"])

        x = x.view(b, f, self.num_filters2, x.size()[-2], x.size()[-1])
        x = self.convlstm2(x)[0][0]
        x = x.view(b * f, self.num_filters2, x.size()[-2], x.size()[-1])
        x = F.dropout(x, param_dict["dropout_p"])

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn33(self.conv33(x)))
        x = self.pool3(x)
        x = F.dropout(x, param_dict["dropout_p"])

        x = x.view(b, f, self.num_filters3, x.size()[-2], x.size()[-1])
        x = self.convlstm3(x)[0][0]
        x = x[:, -1, :]  # 获取最后一帧
        return x.view(b, -1)


class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, (32, 20))
        self.conv2 = nn.Conv1d(64, 64, (32, 1))

    def forward(self, x):
        x = x.unsqueeze(1)
        conv_out = F.relu(self.conv1(x))
        conv_out = self.conv2(conv_out)
        activation = F.relu(conv_out.squeeze(3))
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)
        return max_out


class FinonetRGBDA(nn.Module):
    def __init__(self):
        super(FinonetRGBDA, self).__init__()
        self.rgbd = FinoNetRGBD()

        if os.path.exists("finonet-rgbd.pth"):
            self.rgbd.load_state_dict(torch.load("finonet-rgbd.pth"), strict=False)
        else:
            print("提示: 未找到 'finonet-rgbd.pth'，将从随机初始化开始。")

        self.audio = AudioCNN()

        if param_dict["freeze_rgbd"]:
            for param in self.rgbd.parameters():
                param.requires_grad = False

        self.HIDDEN_SIZE = 128 * 24 * 24
        self.AUDIO_SIZE = 64
        self.linear1 = nn.Linear(self.HIDDEN_SIZE + self.AUDIO_SIZE, 1024)
        self.linear2 = nn.Linear(1024, 2)
        self.dropout = nn.Dropout(p=param_dict["dropout_p"])

    def forward(self, x, x_audio):
        x = self.rgbd(x)
        x_audio = self.audio(x_audio)
        x = torch.cat((x, x_audio), 1)
        x = F.relu(self.linear1(x))
        x = self.linear2(self.dropout(x))
        return x


# ==========================================
# 5. 训练与测试函数
# ==========================================
def train_epoch(model, loader, optimizer, epoch):
    model.train()
    epoch_loss = 0
    all_labels, all_preds = [], []
    for _, _, imgs, audio, labels in loader:
        imgs, audio, labels = imgs.to(device), audio.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(imgs, audio)
        loss = F.cross_entropy(output, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, indices = torch.max(output, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(indices.cpu().numpy())

    acc = f1_score(all_labels, all_preds, average='weighted')
    print(f"[Train] Epoch: {epoch}, Loss: {epoch_loss / len(loader.dataset):.6f} F1: {acc:.4f}")


def test_epoch(model, loader, epoch):
    model.eval()
    test_loss = 0
    all_labels, all_preds = [], []
    with torch.no_grad():
        for _, _, imgs, audio, labels in loader:
            imgs, audio, labels = imgs.to(device), audio.to(device), labels.to(device)
            output = model(imgs, audio)
            loss = F.cross_entropy(output, labels)
            test_loss += loss.item()
            _, indices = torch.max(output, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(indices.cpu().numpy())

    acc = f1_score(all_labels, all_preds, average='weighted')
    print(f"[Test] Epoch: {epoch}, Loss: {test_loss / len(loader.dataset):.6f} F1: {acc:.4f}")


# ==========================================
# 6. 主程序运行
# ==========================================
if __name__ == "__main__":
    print("正在加载数据...")
    df = prepare_data_list()

    if df.empty:
        print("错误: 未加载到有效数据，请检查路径。")
    else:
        tr_succ, tr_fail, te_succ, te_fail = get_all_data_splits(df)
        train_idx = np.concatenate((tr_succ, tr_fail))
        test_idx = np.concatenate((te_succ, te_fail))

        train_dataset = DatasetFD(df, train_idx, param_dict["n_sample_frames"], param_dict["sampling_mode"])
        train_loader = DataLoader(train_dataset, batch_size=param_dict["batch_size"], shuffle=True, num_workers=4)

        test_dataset = DatasetFD(df, test_idx, param_dict["n_sample_frames"], param_dict["sampling_mode"])
        test_loader = DataLoader(test_dataset, batch_size=param_dict["batch_size"], shuffle=False, num_workers=4)

        print(f"训练样本数: {len(train_dataset)}, 测试样本数: {len(test_dataset)}")

        model = FinonetRGBDA().to(device)
        optimizer = optim.Adam(model.parameters(), lr=param_dict["lr"])

        for epoch in range(param_dict["n_epoch"]):
            train_epoch(model, train_loader, optimizer, epoch)
            test_epoch(model, test_loader, epoch)