import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
import librosa
from scipy.io import wavfile
from keras_preprocessing import sequence
from convlstm import ConvLSTM

# ========== 1. 参数设置 ==========
param_dict = {
    'img_path': "/content/failnet_dataset/rgb_imgs/",
    'depth_path': "/content/failnet_dataset/depth_imgs/",
    'audio_path': "/content/failnet_dataset/audio",
    'img_size': 224,
    'sampling_mode': "depth",
    'n_sample_frames': 8,
    'depth_thresh': 500,
    'lr': 5e-5,
    'n_epoch': 25,
    'batch_size': 2,
    'dropout_p': 0.4,
    'jitter': True,
    'flip': True,
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# ========== 2. 数据读取 ==========
print("正在读取数据...")
data = []
id_counter = 0

# 获取所有动作类型
action_list = next(os.walk(param_dict['img_path']))[1]
print("找到的动作类型:", action_list)

for action in action_list:
    # 读取标签文件
    annotation_path = os.path.join(param_dict['img_path'], action, "annotation.txt")
    print(f"读取: {annotation_path}")

    annotation_df = pd.read_csv(annotation_path)

    # 修改1：name列已经是数字，直接转成整数
    annotation_df["name"] = annotation_df["name"].astype(int)

    action_path = os.path.join(param_dict['img_path'], action)
    bags = next(os.walk(action_path))[1]

    for bag in bags:
        bag_label = int(annotation_df[annotation_df["name"] == int(bag)]["label"])

        # 修改2：没有start列，设为0
        fail_timestamp = 0

        bag_path = os.path.join(action_path, bag)
        bag_imgs_path = glob.glob(os.path.join(bag_path, "*.png"))
        bag_imgs_path.sort()
        bag_stamps_path = glob.glob(os.path.join(bag_path, "*.txt"))
        bag_stamps_path.sort()

        bag_content = []
        for img_path, stamp_path in zip(bag_imgs_path, bag_stamps_path):
            timestamp = float(open(stamp_path, "r").read())

            depth_img_path = os.path.join(param_dict['depth_path'], action, bag,
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

df = pd.DataFrame.from_dict(data)
print(f"共读取 {len(df)} 个样本")

# ========== 3. 数据集划分 ==========
from sklearn.model_selection import train_test_split

tr_succ, tr_fail, te_succ, te_fail = [], [], [], []
for action in df["action"].unique():
    succ = df.loc[(df['action'] == action) & (df['bag_label'] == 0)]["unique_id"].values
    fail = df.loc[(df['action'] == action) & (df['bag_label'] == 1)]["unique_id"].values

    if len(succ) > 0:
        X_tr, X_te = train_test_split(succ, test_size=0.3, shuffle=True, random_state=42)
        tr_succ.extend(X_tr)
        te_succ.extend(X_te)

    if len(fail) > 0:
        X_tr, X_te = train_test_split(fail, test_size=0.3, shuffle=True, random_state=42)
        tr_fail.extend(X_tr)
        te_fail.extend(X_te)

train_idx = tr_succ + tr_fail
test_idx = te_succ + te_fail
print(f"训练集: {len(train_idx)} 样本, 测试集: {len(test_idx)} 样本")

# ========== 4. 数据加载器 ==========
# todo:修改前：tf_list = [transforms.ToTensor()]
tf_list = [
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
]
# todo:修改前 tf = transforms.Compose(tf_list)
rgb_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
depth_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
])


class DatasetFD(Dataset):
    def __init__(self, data, indices, n_sample_frames):
        self.data = data.iloc[indices].reset_index(drop=True)
        self.n_sample_frames = n_sample_frames

    def __len__(self):
        return len(self.data)

    def _get_frame_indices(self, frames):
        df_depth = pd.DataFrame(frames)
        a = df_depth["depth_avg"].values > param_dict["depth_thresh"]
        frame_idx = np.where(a == True)[0]
        n_frames = len(frame_idx)

        if n_frames < 8:
            return np.arange(min(8, len(frames)))

        segments = np.split(frame_idx, [int(n_frames / 3), int(n_frames / 3) * 2])
        first = np.sort(np.random.choice(segments[0], min(4, len(segments[0])), replace=False))
        last = np.sort(np.random.choice(segments[2], min(4, len(segments[2])), replace=False))
        return np.concatenate([first, last])[:8]

    def load_img(self, img_path):
        img = Image.open(img_path).crop((220, 160, 444, 384))
        if "depth" in img_path:
            img = Image.fromarray(np.uint8(np.array(img) / np.max(np.array(img) + 1e-6) * 255))
        # return tf(img)

    def load_mfcc(self, wav_path):
        sr, samples = wavfile.read(wav_path)
        if samples.ndim > 1:
            mfcc = librosa.feature.mfcc(y=samples[:, 0].astype(np.float32), sr=sr).T
        else:
            mfcc = librosa.feature.mfcc(y=samples.astype(np.float32), sr=sr).T
        mfcc = sequence.pad_sequences([mfcc], maxlen=3500)[0]
        return torch.Tensor(mfcc).float()

    def __getitem__(self, idx):
        frames = self.data.iloc[idx]["bag_content"]
        frame_idx = self._get_frame_indices(frames)

        vert_flip = np.random.randint(2) if param_dict["flip"] else 0

        img_batch = []
        for i in frame_idx:
            # img = self.load_img(frames[i]["img_path"])
            # depth = self.load_img(frames[i]["depth_path"])
            # 修改过的：
            rgb_img = Image.open(frames[i]["img_path"]).crop((220, 160, 444, 384)).convert("RGB")

            depth_img = Image.open(frames[i]["depth_path"]).crop((220, 160, 444, 384)).convert("L")
            depth_img = Image.fromarray(
                np.uint8(np.array(depth_img) / (np.max(np.array(depth_img)) + 1e-6) * 255)
            )
            img = rgb_tf(rgb_img)
            depth = depth_tf(depth_img)

            if vert_flip:
                img = torch.flip(img, [2])
                depth = torch.flip(depth, [2])
            img_batch.append(torch.cat([img, depth], dim=0))

        while len(img_batch) < 8:
            img_batch.append(torch.zeros_like(img_batch[0]))

        img_batch = torch.stack(img_batch[:8])
        audio = self.load_mfcc(self.data.iloc[idx]["audio_path"])

        return img_batch, audio, self.data.iloc[idx]["bag_label"]


train_dataset = DatasetFD(df, train_idx, param_dict["n_sample_frames"])
test_dataset = DatasetFD(df, test_idx, param_dict["n_sample_frames"])
train_loader = DataLoader(train_dataset, batch_size=param_dict["batch_size"], shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=param_dict["batch_size"], shuffle=False, num_workers=0)


# ========== 5. 模型定义 ==========
class FinoNetRGBD(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_filters1, self.num_filters2, self.num_filters3 = 64, 128, 128

        self.conv1 = nn.Conv2d(4, 64, 3)
        self.conv11 = nn.Conv2d(64, 64, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn11 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2)
        self.convlstm1 = ConvLSTM(64, [64], (3, 3), 1, batch_first=True)

        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv22 = nn.Conv2d(128, 128, 3)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn22 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2)
        self.convlstm2 = ConvLSTM(128, [128], (3, 3), 1, batch_first=True)

        self.conv3 = nn.Conv2d(128, 128, 3)
        self.conv33 = nn.Conv2d(128, 128, 3)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn33 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)
        self.convlstm3 = ConvLSTM(128, [128], (3, 3), 1, batch_first=True)

        self.relu = nn.ReLU()

    def forward(self, x):
        b, t, c, h, w = x.shape

        x = x.view(b * t, c, h, w)
        x = self.pool1(self.relu(self.bn11(self.conv11(self.relu(self.bn1(self.conv1(x)))))))
        x = F.dropout(x, param_dict["dropout_p"])
        x = x.view(b, t, 64, x.shape[-2], x.shape[-1])
        x = self.convlstm1(x)[0][0]

        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        x = self.pool2(self.relu(self.bn22(self.conv22(self.relu(self.bn2(self.conv2(x)))))))
        x = F.dropout(x, param_dict["dropout_p"])
        x = x.view(b, t, 128, x.shape[-2], x.shape[-1])
        x = self.convlstm2(x)[0][0]

        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        x = self.pool3(self.relu(self.bn33(self.conv33(self.relu(self.bn3(self.conv3(x)))))))
        x = F.dropout(x, param_dict["dropout_p"])
        x = x.view(b, t, 128, x.shape[-2], x.shape[-1])
        x = self.convlstm3(x)[0][0]

        return x[:, -1, :].view(b, -1)


class AudioCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, (32, 20))
        self.conv2 = nn.Conv1d(64, 64, 32) 

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = x.squeeze(3)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x


class FinonetRGBDA(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgbd = FinoNetRGBD()
        self.audio = AudioCNN()
        self.fc1 = nn.Linear(128 * 24 * 24 + 64, 1024)
        self.fc2 = nn.Linear(1024, 2)
        self.dropout = nn.Dropout(param_dict["dropout_p"])

    def forward(self, x, x_audio):
        vis = self.rgbd(x)
        aud = self.audio(x_audio)
        feat = torch.cat([vis, aud], dim=1)
        feat = F.relu(self.fc1(feat))
        feat = self.dropout(feat)
        return self.fc2(feat)


# ========== 6. 训练函数 ==========
def train_epoch(model, loader, optimizer, epoch):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for i, (imgs, audio, labels) in enumerate(loader):
        imgs, audio, labels = imgs.to(device), audio.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(imgs, audio)
        # todo:修改前：loss = F.cross_entropy(output, labels)
        loss = F.cross_entropy(output, labels, weight=weight)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = output.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if i % 10 == 0:
            print(f"  Batch {i}/{len(loader)}, Loss: {loss.item():.4f}")

    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"[Train] Epoch {epoch}, Loss: {total_loss / len(loader):.4f}, F1: {f1:.4f}")
    return f1


def test_epoch(model, loader, epoch):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, audio, labels in loader:
            imgs, audio, labels = imgs.to(device), audio.to(device), labels.to(device)
            output = model(imgs, audio)
            loss = F.cross_entropy(output, labels)
            total_loss += loss.item()
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"[Test] Epoch {epoch}, Loss: {total_loss / len(loader):.4f}, F1: {f1:.4f}")
    return f1


# ========== 7. 开始训练  ==========
print("\n" + "=" * 50)
print("开始训练")
print("=" * 50 + "\n")

model = FinonetRGBDA().to(device)
# optimizer = optim.Adam(model.parameters(), lr=param_dict["lr"])
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=5e-5,
    weight_decay=1e-4
)

# todo：新加的class_weight
num_success = 82
num_fail = 147
weight = torch.tensor([1.0/num_success, 1.0/num_fail])
weight = weight / weight.sum()
weight = weight.to(device)

best_f1 = 0
for epoch in range(param_dict["n_epoch"]):
    print(f"\n--- Epoch {epoch + 1}/{param_dict['n_epoch']} ---")
    train_f1 = train_epoch(model, train_loader, optimizer, epoch)
    test_f1 = test_epoch(model, test_loader, epoch)

    if test_f1 > best_f1:
        best_f1 = test_f1
        torch.save(model.state_dict(), "best_model.pth")
        print(f" 保存最佳模型，F1: {best_f1:.4f}")

print("\n" + "=" * 50)
print("训练完成！")
print(f"最佳测试 F1 分数: {best_f1:.4f}")
print("模型已保存为 best_model.pth")
print("=" * 50)