import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import string
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['Microsoft JhengHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# ===== [新增] 混合精度 AMP import =====
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

# 主流程
df = pd.read_csv('labels.csv')
df = df.dropna(subset=['filename', 'label'])
valid_chars = set(string.digits + string.ascii_uppercase + string.ascii_lowercase + '.')
df = df[df['label'].apply(lambda x: all(c in valid_chars for c in str(x)))]
df = df[df['filename'].str.strip() != '']
df.to_csv('labels_clean.csv', index=False)

# 1. 字元表與編碼
alphabet = string.digits + string.ascii_uppercase + string.ascii_lowercase + '.'
char_to_idx = {char: idx + 1 for idx, char in enumerate(alphabet)}  # 0 給 CTC blank
idx_to_char = {idx + 1: char for idx, char in enumerate(alphabet)}
num_classes = len(alphabet) + 1

# 2. 自訂 Dataset
class CaptchaDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.data = self.data.dropna(subset=['filename', 'label'])
        self.data = self.data[self.data['filename'].astype(str).str.strip() != '']
        valid_chars = set(string.digits + string.ascii_uppercase + string.ascii_lowercase + '.')
        self.data = self.data[self.data['label'].apply(lambda x: all(c in valid_chars for c in str(x)))]
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def encode_label(self, label):
        return [char_to_idx[c] for c in label]

    def __getitem__(self, idx):
        img_name = str(self.data.iloc[idx]['filename'])
        label = str(self.data.iloc[idx]['label'])
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        label_encoded = self.encode_label(label)
        label_length = len(label_encoded)
        return image, torch.tensor(label_encoded, dtype=torch.long), label_length, label

# 3. collate_fn 讓 batch 能處理不同長度標籤
def collate_fn(batch):
    images, labels, label_lengths, raw_labels = zip(*batch)
    images = torch.stack(images, 0)
    labels = torch.cat(labels)
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)
    return images, labels, label_lengths, raw_labels

# 4. 模型
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.rnn = nn.LSTM(128 * 10, 256, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256 * 2, num_classes)

    def forward(self, x):
        x = self.cnn(x)  # (B, 64, 10, 37)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2)  # (B, W, C, H)
        x = x.view(b, w, c * h)    # (B, W, C*H)
        x, _ = self.rnn(x)         # (B, W, 256)
        x = self.fc(x)             # (B, W, num_classes)
        x = x.log_softmax(2)
        return x

# CTC解碼（簡單貪婪解碼）
def ctc_greedy_decoder(output, idx_to_char):
    output = output.argmax(2)  # (B, W)
    decoded = []
    for seq in output:
        prev = -1
        seq_str = ''
        for idx in seq.cpu().numpy():
            if idx != prev and idx != 0:
                seq_str += idx_to_char.get(idx, '')
            prev = idx
        decoded.append(seq_str)
    return decoded

# 5. 訓練流程
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"目前使用裝置：{device}")
    csv_path = r'E:\作業\電腦模擬期末\labels.csv'
    img_dir = r'E:\作業\電腦模擬期末\archive'
    batch_size = 128   # ← 依顯卡調整，建議 64~128

    transform = transforms.Compose([
        transforms.Resize((40, 150)),
        transforms.ToTensor(),
    ])

    # ====== 這裡開始是 7:2:1 分割 ======
    dataset = CaptchaDataset(csv_path, img_dir, transform)
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size

    from torch.utils.data import random_split
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    model = CRNN(num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    ctc_loss = nn.CTCLoss(blank=0)

    num_epochs = 50
            # 計算準確率

    loss_history = []
    acc_history = []

    # ===== [AMP] 初始化 GradScaler =====
    scaler = GradScaler()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for images, labels, label_lengths, raw_labels in train_loader:  # <--- 這裡修正
            images = images.to(device)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)
            optimizer.zero_grad()
            # ===== [AMP] 混合精度運算 =====
            with autocast():
                logits = model(images)  # (B, W, num_classes)
                input_lengths = torch.full((images.size(0),), logits.size(1), dtype=torch.long).to(device)
                loss = ctc_loss(logits.permute(1, 0, 2), labels, input_lengths, label_lengths)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

            # 計算準確率（整個字串完全正確才算對）
            decoded = ctc_greedy_decoder(logits, idx_to_char)
            for pred, gt in zip(decoded, raw_labels):
                if pred == gt:
                    correct += 1
                total += 1

        avg_loss = total_loss / len(train_loader)  # <--- 這裡也修正
        acc = correct / total if total > 0 else 0
        loss_history.append(avg_loss)
        acc_history.append(acc)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Acc: {acc:.4f}")

    # 儲存模型
    torch.save(model.state_dict(), 'crnn_captcha.pth')
    print("模型已儲存為 crnn_captcha.pth")

    # 繪製 loss 和 acc 曲線
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('訓練損失')

    plt.subplot(1, 2, 2)
    plt.plot(acc_history, label='Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('訓練準確率')

    plt.tight_layout()
    plt.savefig('training_curve.png')
    plt.show()
    print("訓練曲線已儲存為 training_curve.png")


if __name__ == '__main__':
    train()
