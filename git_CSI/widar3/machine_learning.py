from __future__ import print_function
import torch
import torch.nn as nn
import os
import numpy as np
import scipy.io as scio
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
import warnings
# 屏蔽dropout2d过时警告
warnings.filterwarnings("ignore", category=UserWarning, message="dropout2d: Received a 2-D input to dropout2d")



class Flatten(nn.Module):
    # 构造函数，没有什么要做的
    def __init__(self):
        # 调用父类构造函数
        super(Flatten, self).__init__()

    # 实现forward函数
    def forward(self, input):
        # 保存batch维度，后面的维度全部压平，例如输入是28*28的特征图，压平后为784的向量
        return input.view(input.size(0), -1)


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        # 如果使用view(3,2)或reshape(3,2)，得到的tensor并不是转置的效果，
        # 而是相当于将原tensor的元素按行取出，然后按行放入到新形状的tensor中
        return x.unsqueeze(1)   # return x.view(self.shape)


# 参考 https://www.jianshu.com/p/2d9927a70594
class MyDataset(Dataset):
    def __init__(self, root_path, class_list, location_list):
        self.bvp = []
        self.ges = []
        all_mat_paths = []
        # 递归遍历所有子文件夹
        for root, _, files in os.walk(root_path):
            for fname in files:
                if fname.lower().endswith(".mat"):
                    full_p = os.path.join(root, fname)
                    all_mat_paths.append(full_p)

        print("一共找到mat文件数量：", len(all_mat_paths)) # 加这句打印总数
        if len(all_mat_paths) == 0:
            print("警告：路径下一个mat都没找到！")

        for mat_path in all_mat_paths:
            file_name = os.path.basename(mat_path)
            split_info = file_name.split("-")
            try:
                gesture_cls = int(split_info[1])
                location_id = int(split_info[2])
            except ValueError:
                gesture_cls = int(split_info[2])
                location_id = int(split_info[3])

            # 临时注释过滤，先全部加载测试
            # if gesture_cls not in class_list or location_id not in location_list:
            #    continue

            # 读取mat，增加异常捕获，跳过损坏文件
            try:
                mat_data = scio.loadmat(mat_path)
            except Exception as e:
                print(f"损坏文件跳过：{file_name}，错误：{e}")
                continue

            # 检查是否存在目标键
            if "velocity_spectrum_ro" not in mat_data:
                print(f"文件无velocity_spectrum_ro，跳过：{file_name}")
                continue
            bvp_raw = mat_data["velocity_spectrum_ro"]

            # 维度容错转置不变
            if bvp_raw.ndim == 3:
                bvp_seq = np.transpose(bvp_raw, [2, 0, 1])
            elif bvp_raw.ndim == 2:
                bvp_seq = np.expand_dims(bvp_raw, axis=0)
            else:
                print(f"维度异常跳过：{file_name} shape={bvp_raw.shape}")
                continue

            # =========新增高斯噪声增强代码========
            # 均值0，标准差0.004 轻微噪声模拟环境干扰
            noise = np.random.normal(loc=0, scale=0.004, size=bvp_seq.shape).astype(np.float32)
            bvp_seq = bvp_seq + noise
            # ======================================

            self.bvp.append(torch.from_numpy(bvp_seq).float())
            self.ges.append(gesture_cls - 1)

        self.ges = torch.LongTensor(self.ges)
        print("最终符合条件样本总数：", len(self.bvp))
        print("====标签统计====")
        print("所有标签最小值：", self.ges.min().item())
        print("所有标签最大值：", self.ges.max().item())
        print("全部唯一标签：", torch.unique(self.ges))

    def __len__(self):
        return len(self.bvp)

    def __getitem__(self, idx):
        return self.bvp[idx], self.ges[idx]

    def get_filelist(self, path):
        Filelist = []
        for home, dirs, files in os.walk(path):
            for filename in files:
                # 文件名列表，包含完整路径
                Filelist.append(os.path.join(home, filename))
                # # 文件名列表，只包含文件名
                # Filelist.append(filename)
        return Filelist


# 重写collate_fn函数，其输入为一个batch的sample数据
# 参考https://blog.csdn.net/kejizuiqianfang/article/details/100835528
def collate_fn(batch):
    batch.sort(key=lambda item: len(item[0]), reverse=True)
    bvp_batch, batch_ges = zip(*batch)
    bvp_len = [len(x) for x in bvp_batch]
    batch_bvp = nn.utils.rnn.pad_sequence(bvp_batch, batch_first=True, padding_value=0)
    batch_ges = torch.stack(batch_ges, dim=0).squeeze()
    return batch_bvp, batch_ges, bvp_len


def train(model, train_loader, device, optimizer, epoch):
    # tensor可以直接取代Variable
    cnn_encoder, rnn_decoder = model
    cnn_encoder.train()
    rnn_decoder.train()

    losses = []
    scores = []
    for i, (bvps, labels, bvps_len) in enumerate(train_loader):  # for each training step
        bvps, labels = bvps.to(device), labels.to(device)
        optimizer.zero_grad()  # 梯度清零
        output = rnn_decoder(cnn_encoder(bvps))
        loss = F.cross_entropy(output, labels)  # 对于回归问题，常用的损失函数是均方误差（MSE，Mean Squared Error）
                                                # 对于分类问题，常用的损失函数为交叉熵（CE，Cross Entropy）
                                                # 交叉熵一般与one-hot和softmax在一起使用
        losses.append(loss.item())

        labels_pred = torch.max(output, 1)[1]  # 返回每一行中最大值的那个元素，且返回其索引
        step_score = accuracy_score(labels.cpu().data.squeeze().numpy(), labels_pred.cpu().data.squeeze().numpy())
        scores.append(step_score)

        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('Epoch : %d, Step: %d, Loss: %.3f' % (epoch, i, loss.item()))
    return losses, scores


def validation(model, device, optimizer, test_loader, epoch):
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for (X, y, X_len) in test_loader:
            # distribute data to device
            X, y = X.to(device), y.to(device).view(-1, )

            output = rnn_decoder(cnn_encoder(X))

            loss = F.cross_entropy(output, y, reduction='sum')
            test_loss += loss.item()                 # sum up batch loss
            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)

    test_loss /= len(test_loader.dataset)

    # compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss, 100 * test_score))

    # save Pytorch models of best record
    # torch.save(cnn_encoder.state_dict(), os.path.join(save_model_path, 'cnn_encoder_epoch{}.pth'.format(epoch + 1)))  # save spatial_encoder
    # torch.save(rnn_decoder.state_dict(), os.path.join(save_model_path, 'rnn_decoder_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
    # torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'optimizer_epoch{}.pth'.format(epoch + 1)))      # save optimizer
    print("Epoch {} model saved!".format(epoch + 1))

    return test_loss, test_score


def crnn_test():
    import os
    save_model_path = "./CRNN_ckpt/"
    os.makedirs(save_model_path, exist_ok=True)
    print('===================CRNN===========================')
    use_cuda = torch.cuda.is_available()  # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU

    # 训练
    epoch_num = 100
    class_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    class_num = len(class_list)
    location_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    """dirlist = ['20181109', '20181115', '20181204', '20181117', '20181118', '20181128',
               '20181130', '20181204', '20181208', '20181209', '20181211']
    full_dataset = MyDataset('BVP', dirlist, class_list, location_list)
    """
    root_data = r"D:\userZHANG\Widar3.0test\data\BVP"
    full_dataset = MyDataset(root_data, class_list, location_list)

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=0, collate_fn=collate_fn)
    valid_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True, num_workers=0, collate_fn=collate_fn)

    cnn_encoder = EncoderCNN().to(device)
    rnn_decoder = DecoderRNN(class_num).to(device)

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        cnn_encoder = nn.DataParallel(cnn_encoder)
        rnn_decoder = nn.DataParallel(rnn_decoder)

    crnn_params = list(cnn_encoder.parameters()) + list(rnn_decoder.parameters())

    # optimizer = optim.Adam(crnn_params, lr=1e-8, betas=(0.9, 0.99))  # 初始化优化器
    optimizer = optim.Adam(crnn_params, lr=2e-4)  # 初始化优化器

    # record training process
    epoch_train_losses = []
    epoch_train_scores = []
    epoch_test_losses = []
    epoch_test_scores = []

    # start training
    for epoch in range(epoch_num):  # train entire dataset 5 times
        train_losses, train_scores = train([cnn_encoder, rnn_decoder], train_loader, device, optimizer, epoch)
        epoch_test_loss, epoch_test_score = validation([cnn_encoder, rnn_decoder], device, optimizer, valid_loader, epoch)

        # save results
        epoch_train_losses.append(train_losses)
        epoch_train_scores.append(train_scores)
        epoch_test_losses.append(epoch_test_loss)
        epoch_test_scores.append(epoch_test_score)

        # save all train test results
        A = np.array(epoch_train_losses)
        B = np.array(epoch_train_scores)
        C = np.array(epoch_test_losses)
        D = np.array(epoch_test_scores)
        np.save('./CRNN_ckpt/CRNN_epoch_training_losses.npy', A)
        np.save('./CRNN_ckpt/CRNN_epoch_training_scores.npy', B)
        np.save('./CRNN_ckpt/CRNN_epoch_test_loss.npy', C)
        np.save('./CRNN_ckpt/CRNN_epoch_test_score.npy', D)


class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        # CNN
        encoder = nn.Sequential()
        encoder.add_module('reshape{}'.format(0), Reshape(-1, 1, 20, 20))
        encoder.add_module('conv{}'.format(0), nn.Conv2d(1, 32, (3, 3), stride=(1, 1), padding=1))
        encoder.add_module('relu{}'.format(0), nn.ReLU(True))
        encoder.add_module('pooling{}'.format(0), nn.MaxPool2d(2, 2))

        encoder.add_module('conv{}'.format(1), nn.Conv2d(32, 64, 3, padding=1))
        encoder.add_module('relu{}'.format(1), nn.ReLU(True))
        encoder.add_module('pooling{}'.format(1), nn.MaxPool2d(2, 2))

        encoder.add_module('flatten{}'.format(0), Flatten())
        encoder.add_module('full_connenct{}'.format(0), nn.Linear(1600, 128))
        encoder.add_module('relu{}'.format(1), nn.ReLU(True))
        encoder.add_module('dropout{}'.format(0), nn.Dropout(0.1))
        encoder.add_module('full_connenct{}'.format(1), nn.Linear(128, 64))
        encoder.add_module('relu{}'.format(2), nn.ReLU(True))
        self.encoder = encoder

    def forward(self, input):
        cnn_embed_seq = []
        for t in range(input.size(1)):
            cnn_embed_seq.append(self.encoder(input[:, t, :, :]))
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)

        return cnn_embed_seq


class DecoderRNN(nn.Module):
    def __init__(self, class_num):
        super(DecoderRNN, self).__init__()
        self.class_num = class_num
        # 2层GRU，hidden128，层间轻微dropout防过拟合
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        # 优化分类头：去掉Dropout2d，双层非线性提升拟合
        self.decoder = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.class_num)
        )

    def forward(self, input_seq):
        gru_out, _ = self.gru(input_seq)
        # 取最后时序步全局特征分类
        last_feat = gru_out[:, -1, :]
        logits = self.decoder(last_feat)
        return logits


if __name__ == '__main__':
    crnn_test()