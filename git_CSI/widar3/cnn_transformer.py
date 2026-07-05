from __future__ import print_function
import torch
import torch.nn as nn
import os
import numpy as np
import scipy.io as scio
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from sklearn.metrics import accuracy_score
from collections import Counter
import warnings
import math

# 屏蔽dropout过时警告
warnings.filterwarnings("ignore", category=UserWarning, message="dropout2d: Received a 2-D input to dropout2d")

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, input):
        return input.view(input.size(0), -1)

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
    def forward(self, x):
        return x.unsqueeze(1)

# 通道注意力（修复return语法）
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class MyDataset(Dataset):
    def __init__(self, root_path, class_list, location_list):
        self.bvp = []
        self.ges = []
        all_mat_paths = []
        for root, _, files in os.walk(root_path):
            for fname in files:
                if fname.lower().endswith(".mat"):
                    full_p = os.path.join(root, fname)
                    all_mat_paths.append(full_p)
        print("一共找到mat文件：", len(all_mat_paths))
        for mat_path in all_mat_paths:
            file_name = os.path.basename(mat_path)
            split_info = file_name.split("-")
            gesture_cls = None
            location_id = None
            try:
                gesture_cls = int(split_info[1])
                location_id = int(split_info[2])
            except ValueError:
                try:
                    gesture_cls = int(split_info[2])
                    location_id = int(split_info[3])
                except:
                    continue
            # 只保留前6类手势
            if gesture_cls not in class_list or location_id not in location_list:
                continue
            try:
                mat_data = scio.loadmat(mat_path)
            except Exception as e:
                print(f"损坏文件跳过：{file_name}，错误：{e}")
                continue
            if "velocity_spectrum_ro" not in mat_data:
                continue
            bvp_raw = mat_data["velocity_spectrum_ro"]
            if bvp_raw.size == 0:
                continue
            if bvp_raw.ndim == 3:
                bvp_seq = np.transpose(bvp_raw, [2, 0, 1])
            elif bvp_raw.ndim == 2:
                bvp_seq = np.expand_dims(bvp_raw, axis=0)
            else:
                continue
            if bvp_seq.size == 0:
                continue

            # =========【0~1归一化，唯一新增逻辑】==========
            bvp_seq = bvp_seq.astype(np.float32)
            minv = bvp_seq.min()
            maxv = bvp_seq.max()
            if maxv - minv > 1e-6:
                bvp_seq = (bvp_seq - minv) / (maxv - minv)
            # ==============================================

            # 原有噪声增强完全保留，不新增任何增强
            noise = np.random.normal(loc=0, scale=0.004, size=bvp_seq.shape).astype(np.float32)
            bvp_seq = bvp_seq + noise

            self.bvp.append(torch.from_numpy(bvp_seq).float())
            self.ges.append(gesture_cls - 1)
        self.ges = torch.LongTensor(self.ges)
        print("有效样本总数：", len(self.bvp))
        print("标签分布：", Counter(self.ges.tolist()))
    def __len__(self):
        return len(self.bvp)
    def __getitem__(self, idx):
        return self.bvp[idx], self.ges[idx]
    def get_filelist(self, path):
        Filelist = []
        for home, dirs, files in os.walk(path):
            for filename in files:
                Filelist.append(os.path.join(home, filename))
        return Filelist

# collate_fn无修改
def collate_fn(batch):
    batch.sort(key=lambda x: x[0].size(0), reverse=True)
    bvp_batch, batch_ges = zip(*batch)
    bvp_len = [x.size(0) for x in bvp_batch]
    batch_bvp = nn.utils.rnn.pad_sequence(bvp_batch, batch_first=True, padding_value=0.)
    batch_ges = torch.stack(batch_ges, dim=0).squeeze()
    return batch_bvp, batch_ges, bvp_len

# ========== 修改 train 函数，传递长度 ==========
def train(model, params, device, optimizer, scheduler, epoch, train_loader, cls_weight):
    cnn_encoder, rnn_decoder = model
    cnn_encoder.train()
    rnn_decoder.train()
    losses = []
    scores = []
    for i, (bvps, labels, bvp_len) in enumerate(train_loader):
        bvps, labels = bvps.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        feat_seq = cnn_encoder(bvps)
        output = rnn_decoder(feat_seq, bvp_len)   # 传入长度列表
        loss = F.cross_entropy(output, labels, weight=cls_weight, label_smoothing=0.05)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=4.0)
        optimizer.step()
        losses.append(loss.item())
        pred = torch.max(output, 1)[1]
        acc = accuracy_score(labels.cpu().numpy(), pred.cpu().numpy())
        scores.append(acc)
        if i % 100 == 0:
            print(f'Epoch {epoch}, Step {i}, Loss {loss.item():.3f}')
    return np.mean(losses), np.mean(scores)

def validation(model, device, test_loader):
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()
    test_loss = 0
    all_y = []
    all_pred = []
    with torch.no_grad():
        for (X, y, bvp_len) in test_loader:
            X, y = X.to(device), y.to(device)
            feat_seq = cnn_encoder(X)
            output = rnn_decoder(feat_seq, bvp_len)   # 传入长度列表
            loss = F.cross_entropy(output, y, reduction="sum")
            test_loss += loss.item()
            pred = torch.max(output, 1)[1]
            all_y.extend(y.cpu().tolist())
            all_pred.extend(pred.cpu().tolist())
    test_loss /= len(test_loader.dataset)
    acc = accuracy_score(all_y, all_pred)
    print(f"\nVal Loss:{test_loss:.4f} | Val Acc:{acc*100:.2f}%\n")
    return test_loss, acc

# ========== CNN编码器（与初版完全相同） ==========
class EncoderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        encoder = nn.Sequential()
        encoder.add_module('reshape', Reshape(-1, 1, 20, 20))
        encoder.add_module('conv0', nn.Conv2d(1, 32, 3, padding=1))
        encoder.add_module('relu0', nn.ReLU(True))
        encoder.add_module('ca0', ChannelAttention(32))
        encoder.add_module('pool0', nn.MaxPool2d(2))
        encoder.add_module('conv1', nn.Conv2d(32, 64, 3, padding=1))
        encoder.add_module('relu1', nn.ReLU(True))
        encoder.add_module('ca1', ChannelAttention(64))
        encoder.add_module('pool1', nn.MaxPool2d(2))
        encoder.add_module('flatten', Flatten())
        encoder.add_module('fc0', nn.Linear(1600, 128))
        encoder.add_module('relu2', nn.ReLU(True))
        encoder.add_module('drop0', nn.Dropout(0.1))
        encoder.add_module('fc1', nn.Linear(128, 64))
        encoder.add_module('relu3', nn.ReLU(True))
        self.encoder = encoder
    def forward(self, input):
        seq_out = []
        for t in range(input.size(1)):
            frame_feat = self.encoder(input[:, t, :])
            seq_out.append(frame_feat)
        seq_out = torch.stack(seq_out, dim=0).transpose(0, 1)
        return seq_out

# ========== 位置编码 ==========
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]

# ========== Transformer 解码器（替换 GRU） ==========
class DecoderTransformer(nn.Module):
    def __init__(self, class_num, d_model=256, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(64, d_model)          # 64 来自 CNN 输出
        self.pos_encoder = PositionalEncoding(d_model, max_len=500)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=False          # 我们使用 (seq, batch, feature) 格式
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, class_num)
        )

    def forward(self, input_seq, seq_lengths):
        """
        input_seq: (batch, seq_len, 64)
        seq_lengths: list of ints, 每个样本的实际长度
        """
        batch, seq_len, _ = input_seq.size()
        # 投影到 d_model
        x = self.input_proj(input_seq)          # (batch, seq_len, d_model)
        # 位置编码
        x = self.pos_encoder(x)                 # (batch, seq_len, d_model)
        # 转置为 (seq_len, batch, d_model) 供 Transformer
        x = x.permute(1, 0, 2)
        # 生成 padding mask，True 表示忽略该位置
        # mask 形状 (batch, seq_len)
        mask = torch.arange(seq_len, device=input_seq.device).expand(batch, seq_len) >= torch.tensor(seq_lengths, device=input_seq.device).unsqueeze(1)
        # 传入 transformer
        x = self.transformer(x, src_key_padding_mask=mask)   # (seq_len, batch, d_model)
        # 转置回 (batch, seq_len, d_model)
        x = x.permute(1, 0, 2)
        # 有效位置掩码（用于平均）
        valid_mask = ~mask   # (batch, seq_len)
        # 对有效部分求和
        sum_x = torch.sum(x * valid_mask.unsqueeze(-1).float(), dim=1)   # (batch, d_model)
        # 平均
        avg_x = sum_x / torch.tensor(seq_lengths, dtype=torch.float, device=input_seq.device).unsqueeze(1)  # (batch, d_model)
        logits = self.classifier(avg_x)
        return logits

def crnn_test():
    save_model_path = "./CRNN_6Class_Transformer/"
    os.makedirs(save_model_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch_total = 100
    warmup = 8
    class_list = [1,2,3,4,5,6]
    class_num = 6
    location_list = list(range(1,11))
    root = r"D:\userZHANG\Widar3.0test\data\BVP"

    dataset = MyDataset(root, class_list, location_list)
    full_len = len(dataset)
    train_size = int(0.8 * full_len)
    test_size = full_len - train_size
    all_labels = [item[1].item() for item in dataset]
    label_cnt = Counter(all_labels)
    max_cnt = max(label_cnt.values())
    weight_list = [np.sqrt(max_cnt / label_cnt[i]) for i in range(class_num)]
    cls_weight = torch.tensor(weight_list, dtype=torch.float32).to(device)
    print("类别损失权重：", np.round(weight_list, 2))

    train_ds, test_ds = torch.utils.data.random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_ds, batch_size=32, shuffle=True, num_workers=0, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        test_ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_fn
    )

    cnn = EncoderCNN().to(device)
    transformer = DecoderTransformer(class_num, d_model=256, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1).to(device)
    if torch.cuda.device_count() > 1:
        cnn = nn.DataParallel(cnn)
        transformer = nn.DataParallel(transformer)
    model = [cnn, transformer]
    params = list(cnn.parameters()) + list(transformer.parameters())

    lr = 1.5e-4
    opt = optim.Adam(params, lr=lr)
    warm_sch = LinearLR(opt, start_factor=0.001, end_factor=1.0, total_iters=warmup)
    cos_sch = CosineAnnealingLR(opt, T_max=epoch_total - warmup, eta_min=1e-6)

    best_acc = 0.0
    patience = 25
    no_improve = 0
    tr_loss_list, tr_acc_list = [], []
    val_loss_list, val_acc_list = [], []

    for epoch in range(epoch_total):
        tr_loss, tr_acc = train(model, params, device, opt, cos_sch, epoch, train_loader, cls_weight)
        val_loss, val_acc = validation(model, device, val_loader)
        if epoch < warmup:
            warm_sch.step()
        else:
            cos_sch.step()

        tr_loss_list.append(tr_loss)
        tr_acc_list.append(tr_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        np.save(os.path.join(save_model_path, "train_loss.npy"), np.array(tr_loss_list))
        np.save(os.path.join(save_model_path, "train_acc.npy"), np.array(tr_acc_list))
        np.save(os.path.join(save_model_path, "val_loss.npy"), np.array(val_loss_list))
        np.save(os.path.join(save_model_path, "val_acc.npy"), np.array(val_acc_list))

        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            torch.save(cnn.state_dict(), os.path.join(save_model_path, "best_cnn.pth"))
            torch.save(transformer.state_dict(), os.path.join(save_model_path, "best_transformer.pth"))
            print(f"✅ 新最优：{best_acc*100:.2f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"早停，最高准确率 {best_acc*100}")
                break
    print("训练结束，最优验证精度：", best_acc*100)

if __name__ == "__main__":
    crnn_test()