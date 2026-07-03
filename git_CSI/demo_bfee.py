import os
import numpy as np
import matplotlib.pyplot as plt
# 正确导入：底层解析类 + 标准化函数
from Bfee import Bfee
from get_scale_csi import get_scale_csi

# 1. 配置CSI原始数据根目录
CSI_ROOT = r"../data/CSI"
# 测试的dat文件
target_dat = r"20181109/user1/csi_res.dat"
# 拼接完整路径
file_path = os.path.join(CSI_ROOT, target_dat)
print("待读取文件完整路径：", file_path)

# 2. 核心修正：用Bfee类读取Intel格式dat（你的采集设备）
bf_obj = Bfee.from_file_intel(file_path)
bf_list = bf_obj.dicts  # 所有CSI数据包字典列表
print(f"文件总CSI帧数：{len(bf_list)}")

if len(bf_list) < 20:
    print("警告：该样本帧数不足20，批量训练时会被过滤")

# 3. 标准化CSI + 压缩单发射天线维度
csi_frames = []
for pkt in bf_list:
    scaled_csi = get_scale_csi(pkt)
    csi_frames.append(np.squeeze(scaled_csi))
csi_np = np.array(csi_frames)
csi_amp = np.abs(csi_np)
print("CSI数组shape [帧数,接收天线,子载波]：", csi_np.shape)

# 4. 绘图：0号接收天线，第3个子载波
# 解决matplotlib中文乱码警告
plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False
plt.figure(figsize=(10,4))
plt.plot(csi_amp[:, 3, 0])
plt.title("CSI Amplitude Time Series (Bfee原生Intel解析)")
plt.xlabel("Frame Index")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()