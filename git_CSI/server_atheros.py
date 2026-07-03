# -*- coding:utf-8 -*-
import socket
import numpy as np
from Bfee import Bfee
import matplotlib.pyplot as plt
import threading
import scipy.io as scio
from scipy import signal
from sklearn.decomposition import PCA
from widar3.bvp import circshift1D
import os
import pywt

import copy

# plt.figure(1)
# csi_data = []
csi_timestamp = []
MAXCSI = 1000
MAXRCEIBUF = 39000

def hampel(x, window_size, n_sigmas):
    import numpy as np
    k = window_size // 2
    x_pad = np.pad(x, pad_width=k, mode='reflect')
    x_med = np.zeros_like(x)
    x_mad = np.zeros_like(x)
    for i in range(len(x)):
        win = x_pad[i:i+window_size]
        med = np.median(win)
        mad = np.median(np.abs(win - med))
        x_med[i] = med
        x_mad[i] = mad
    threshold = n_sigmas * 1.4826 * x_mad
    mask = np.abs(x - x_med) > threshold
    x_clean = x.copy()
    x_clean[mask] = x_med[mask]
    return x_clean

"""def sock():
    #   设置IP和端口
    global csi_data
    global csi_timestamp
    global port
    # host = socket.gethostname()  # sudo vi /etc/hosts replace the ip
    host = 'localhost'
    ip = socket.gethostbyname(host)

    print("hostname: " + host + ", host ip: " + ip + ', host port:' + str(port))
    #   bind绑定该端口
    try:
        mySocket.bind((host, port))
    except:
        mySocket.bind((host, port + 1))
        #   监听

    while True:
        print("程序开始")
        mySocket.listen(10)
        #   接收客户端连接
        print("等待连接....")
        client, address = mySocket.accept()
        print("新连接")
        print("client IP is %s" % address[0])
        print("client port is %d" % address[1])

        count = 1
        while True:
            try:
                msg = client.recv(2)
                ll = int.from_bytes(msg, byteorder='big', signed=False)
                print('datalen:', ll)

                msg2_len = 0
                msg2 = b''
                while msg2_len < ll:  # read buffer for many times
                    msg2_temp = client.recv(MAXRCEIBUF)
                    msg2 = msg2 + msg2_temp  # combine bytes
                    msg2_len = msg2_len + len(msg2_temp)
            except Exception as e:
                client.close()
                mySocket.close()
                print("连接错误: ", e)
                print("连接断开")
                setup_connect()
                print("重新建立套接字")
                continue
            if msg == b"":
                print("程序结束2\n")
                break
            else:
                with open('test/csi_res.dat', 'ab+') as f:
                    f.write(msg)
                    f.write(msg2)

            try:
                bfee = Bfee.from_stream_atheros(msg2, ll)
                csi_timestamp = bfee.all_timestamp
                # csi_data = bfee.all_csi

                if count == 1:
                    csi_temp = bfee.all_csi
                    csi_aver_rssi_temp = bfee.all_aver_rssi

                else:
                    try:
                        csi_temp = np.concatenate((csi_temp, bfee.all_csi), axis=0)
                        csi_aver_rssi_temp = np.concatenate((csi_aver_rssi_temp, bfee.all_aver_rssi), axis=0)
                    except:
                        # csi_temp = copy.copy(bfee.all_csi)
                        csi_temp = bfee.all_csi
                        csi_aver_rssi_temp = bfee.all_aver_rssi
                        count = 1

            except Exception as e:
                print("有错误2: ", e)
                continue

            if count % int(MAXCSI / 100) == 0:
                plt.clf()
                # plt.figure(1)
                # display(csi_temp, csi_aver_rssi_temp)

                csi = np.zeros([np.size(csi_temp, 0), 56 * 3], dtype=complex)
                for ii in range(3):
                    csi[:, ii * 56: (ii + 1) * 56] = csi_temp[:, :, ii, 0]

                # # # plt.figure(1)
                get_doppler_spectrum_stream(csi, 1, 3, 56, 56, 'stft')
                # plt.clf()
                # get_figures(csi, method='stft')

                csi_timestamp = []
                count = 1
            else:
                count = count + 1
            # print("count", count)
"""

"""def setup_connect():
    #   设置IP和端口
    host = socket.gethostname()
    #   bind绑定该端口
    try:
        mySocket.bind((host, port))
    except:
        mySocket.bind((host, port + 1))
    #   监听
    mySocket.listen(10)
    print("等待连接....")
    client, address = mySocket.accept()
    print("新连接")
    print("IP is %s" % address[0])
    print("port is %d\n" % address[1])
"""

def get_doppler_spectrum_stream(csi, rx_cnt, rx_acnt, num_tones, freq_bin_len, method):
    # 设置参数
    sample_rate = 1000
    half_rate = sample_rate / 2
    upper_order = 6
    upper_stop = 40
    lower_order = 6
    lower_stop = 2
    lu, ld = signal.butter(upper_order, upper_stop / sample_rate, 'low')
    hu, hd = signal.butter(lower_order, lower_stop / sample_rate, 'high')

    ii = 0
    # Select Antenna Pair[WiDance]
    csi_mean = np.mean(np.abs(csi), axis=0)
    csi_var = np.sqrt(np.var(np.abs(csi), axis=0))
    csi_mean_var_ratio = np.divide(csi_mean, csi_var)
    csi_mean_var_ratio_mean = np.mean(np.transpose(np.reshape(csi_mean_var_ratio, [rx_acnt, num_tones])), axis=0)
    idx = int(np.where(csi_mean_var_ratio_mean == np.max(csi_mean_var_ratio_mean))[0])
    csi_data_ref = np.tile(csi[:, idx * num_tones: (idx + 1) * num_tones], [1, rx_acnt])
    # Amp Adjust[IndoTrack]
    csi_data_adj = np.zeros((np.shape(csi)), dtype=complex)
    csi_data_ref_adj = np.zeros(np.shape(csi_data_ref), dtype=complex)
    alpha_sum = 0

    for jj in range(num_tones * rx_acnt):
        amp = np.abs(csi[:, jj])
        alpha = np.min(amp[amp != 0])
        alpha_sum = alpha_sum + alpha
        csi_data_adj[:, jj] = np.multiply(np.abs(csi[:, jj]) - alpha, np.exp(1j * np.angle(csi[:, jj])))

    beta = 1000 * alpha_sum / (num_tones * rx_acnt)
    for jj in range(num_tones * rx_acnt):
        csi_data_ref_adj[:, jj] = np.multiply(np.abs(csi[:, jj]) + beta, np.exp(1j * np.angle(csi_data_ref[:, jj])))

    # Conj Mult
    conj_mult = np.multiply(csi_data_adj, np.conj(csi_data_ref_adj))
    conj_mult = np.concatenate([conj_mult[:, : int(num_tones * idx)], conj_mult[:, int(num_tones * (idx + 1)): num_tones * 3]], axis=1)
    # Filter Out Static Component & High Frequency Component
    for jj in range(np.size(conj_mult, 1)):
        conj_mult[:, jj] = signal.lfilter(lu, ld, conj_mult[:, jj])
        conj_mult[:, jj] = signal.lfilter(hu, hd, conj_mult[:, jj])

    # PCA analysis
    # pca = PCA(n_components=60)
    # conj_mult_pca_real = pca.fit_transform(conj_mult.real)  # 不支持复数，暂时使用这种方法，和MATLAB不同
    # conj_mult_pca_imag = pca.fit_transform(conj_mult.imag)  # 不支持复数，暂时使用这种方法，和MATLAB不同
    # conj_mult_pca = np.zeros(np.shape(conj_mult_pca_real), dtype=complex)
    # for kk in range(np.size(conj_mult_pca_real, 0)):
    #     for qq in range(np.size(conj_mult_pca_real, 1)):
    #         conj_mult_pca[kk, qq] = complex(conj_mult_pca_real[kk, qq], conj_mult_pca_imag[kk, qq])

    # % TFA With CWT or STFT
    if method == 'stft':
        # window_size = int(np.round(sample_rate / 4 + 1))
        window_size = sample_rate - 1
        if np.mod(window_size, 2) == 0:
            window_size = window_size + 1
        # f, t, freq_time_prof = signal.spectrogram(conj_mult[:,0],window=signal.get_window('hamming',256),fs=1000,nperseg=256, noverlap=255,return_onesided=False,axis=0)  # 和MATLAB的spectragram()对应
        f, t, freq_time_prof = signal.stft(conj_mult[:, 0], window=signal.get_window('hann', window_size), fs=sample_rate, nperseg=window_size, noverlap=window_size - 1, return_onesided=False)  # 和MATLAB的tfrsp()对应
    # plt.figure(ii + 1)
    # plt.pcolormesh(np.arange(np.size(freq_time_prof, 1)), np.arange(np.size(freq_time_prof, 0)),
    #                circshift1D(np.abs(freq_time_prof), int(np.size(freq_time_prof, 0) / 2)))
    # plt.pause(0.1)
    # Select Concerned Freq

    # Spectrum Normalization By Sum For Each Snapshot
    freq_time_prof = np.divide(np.abs(freq_time_prof), np.tile(np.sum(np.abs(freq_time_prof), 0), np.size(freq_time_prof, 0)).reshape([np.size(freq_time_prof, 0), np.size(freq_time_prof, 1)]))

    # Frequency Bin(Corresponding to FFT Results)

    # Store Doppler Velocity Spectrum
    if ii == 0:
        doppler_spectrum = np.zeros([rx_cnt, np.size(freq_time_prof, 0), np.size(freq_time_prof, 1)])
    if np.size(freq_time_prof, 1) >= np.size(doppler_spectrum, 2):
        doppler_spectrum[ii, :, :] = freq_time_prof[:, : np.size(doppler_spectrum, 2)]
    else:
        doppler_spectrum[ii, :, :] = np.concatenate([freq_time_prof, np.zeros([np.size(doppler_spectrum, 1), int(np.size(doppler_spectrum, 2) - np.size(freq_time_prof, 1))])], axis=1)

    doppler_spectrum = doppler_spectrum.squeeze()
    plt.pcolormesh(np.arange(np.size(doppler_spectrum, 1)), np.arange(np.size(doppler_spectrum, 0)),
                   circshift1D(np.abs(doppler_spectrum), int(np.size(doppler_spectrum, 0) / 2)), cmap='jet')
    plt.ylim([420, 550])
    plt.pause(0.1)

    doppler_spectrum_center = int(np.where(f == np.max(f))[0])
    return doppler_spectrum, circshift1D(f, doppler_spectrum_center)[doppler_spectrum_center - freq_bin_len: doppler_spectrum_center + freq_bin_len + 1]


def get_doppler_spectrum_file(csi, rx_cnt, rx_acnt, num_tones, freq_bin_len, method):
    # 设置参数
    sample_rate = 1000
    half_rate = sample_rate / 2
    upper_order = 6
    upper_stop = 20
    lower_order = 3
    lower_stop = 1
    lu, ld = signal.butter(upper_order, upper_stop / half_rate, 'low')
    hu, hd = signal.butter(lower_order, lower_stop / half_rate, 'high')
    window_size = 300
    ii = 0
    # Select Antenna Pair[WiDance]
    csi_mean = np.mean(np.abs(csi), axis=0)
    csi_var = np.sqrt(np.var(np.abs(csi), axis=0))
    # 修复除零报错
    eps = 1e-10
    c_var = csi_var + eps
    csi_mean_var_ratio = np.divide(csi_mean, c_var)
    csi_mean_var_mean = np.mean(np.transpose(np.reshape(csi_mean_var_ratio, [rx_acnt, num_tones])), axis=0)
    idx = np.argmax(csi_mean_var_mean)
    csi_data_ref = np.tile(csi[:, idx * num_tones: (idx + 1) * num_tones], [1, rx_acnt])
    # Amp Adjust[IndoTrack]
    csi_data_adj = np.zeros((np.shape(csi)), dtype=complex)
    csi_data_ref_adj = np.zeros(np.shape(csi_data_ref), dtype=complex)
    alpha_sum = 0
    for jj in range(num_tones * rx_acnt):
        amp = np.abs(csi[:, jj])
        alpha = np.min(amp[amp != 0])
        alpha_sum = alpha_sum + alpha
        csi_data_adj[:, jj] = np.multiply(np.abs(csi[:, jj]) - alpha, np.exp(1j * np.angle(csi[:, jj])))
    beta = 1000 * alpha_sum / (num_tones * rx_acnt)
    for jj in range(num_tones * rx_acnt):
        csi_data_ref_adj[:, jj] = np.multiply(np.abs(csi[:, jj]) + beta, np.exp(1j * np.angle(csi[:, jj])))
    # Conj Mult
    conj_mult = np.multiply(csi_data_adj, np.conj(csi_data_ref_adj))
    conj_mult = np.concatenate([conj_mult[:, : int(num_tones * idx)], conj_mult[:, int(num_tones * (idx + 1)): num_tones * 3]], axis=1)
    # Filter Out Static Component & High Frequency Component
    for jj in range(np.size(conj_mult, 1)):
        conj_mult[:, jj] = signal.lfilter(lu, ld, conj_mult[:, jj])
        conj_mult[:, jj] = signal.lfilter(hu, hd, conj_mult[:, jj])
    # PCA analysis
    # pca = PCA(n_components=60)
    # conj_mult_pca = pca.fit_transform(conj_mult)
    for cc in range(1):
        if method == 'stft':
            time_instance = np.arange(len(conj_mult))
            if np.mod(window_size, 2) == 0:
                window_size = window_size + 1
            f, t, freq_time_prof = signal.stft(conj_mult[:, 12], window=signal.get_window('hann', window_size), fs=1000, nperseg=window_size, noverlap=window_size-1, return_onesided=False)
        # Spectrum Normalization By Sum For Each Snapshot
        freq_time_prof = np.divide(np.abs(freq_time_prof), np.tile(np.sum(np.abs(freq_time_prof), 0), np.size(freq_time_prof, 0)).reshape([np.size(freq_time_prof, 0), np.size(freq_time_prof, 1)]))
        if ii == 0:
            doppler_spectrum = np.zeros([rx_cnt, np.size(freq_time_prof, 0), np.size(freq_time_prof, 1)])
        if np.size(freq_time_prof, 1) >= np.size(doppler_spectrum, 2):
            doppler_spectrum[ii, :, :] = freq_time_prof[:, : np.size(doppler_spectrum, 2)]
        else:
            doppler_spectrum[ii, :, :] = np.concatenate([freq_time_prof, np.zeros([np.size(doppler_spectrum, 1), int(np.size(doppler_spectrum, 2) - np.size(freq_time_prof, 1))])], axis=1)
        doppler_spectrum_center = np.argmax(f)
        doppler_spectrum = doppler_spectrum.squeeze()
        #plt.subplot(3, 1, 3)
        from matplotlib.colors import BoundaryNorm
        from matplotlib.ticker import MaxNLocator
        cmap = plt.get_cmap('jet')
        levels = MaxNLocator(nbins=15).tick_values(doppler_spectrum.min(), doppler_spectrum.max())
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        #cf = plt.contourf(np.arange(np.size(doppler_spectrum, 1)), np.arange(np.size(doppler_spectrum, 0)),
        #               circshift1D(doppler_spectrum, int(np.size(doppler_spectrum, 0) / 2)), levels=levels,
        #                  cmap=cmap)
        #plt.ylim([int(np.size(doppler_spectrum, 0) / 2) - 15, int(np.size(doppler_spectrum, 0) / 2) + 15])
        #plt.colorbar()
    #plt.pause(1)
    return doppler_spectrum, circshift1D(f, doppler_spectrum_center)[doppler_spectrum_center - freq_bin_len: doppler_spectrum_center + freq_bin_len + 1]

def get_figures(csi, rssi=[], method='stft'):
    sample_rate = 1000
    half_rate = sample_rate / 2
    upper_order = 6
    upper_stop = 80  # Hz
    lower_order = 3
    lower_stop = 3  # Hz
    lu, ld = signal.butter(upper_order, upper_stop / half_rate, 'low')
    hu, hd = signal.butter(lower_order, lower_stop / half_rate, 'high')

    plt.subplot(3, 3, 1)
    plt.plot(np.arange(np.size(csi, 0)), csi[:, 0])
    csi_ratio = csi[:, 0] / csi[:, 56]

    f, t, freq_time_prof = signal.stft(csi[:, 0], window=signal.get_window('hann', 256), nperseg=256,
                                       noverlap=255)  # 和MATLAB的tfrsp()对应
    plt.subplot(3, 3, 2)
    plt.pcolormesh(np.arange(np.size(freq_time_prof, 1)), np.arange(np.size(freq_time_prof, 0)),
                   circshift1D(np.abs(freq_time_prof), int(np.size(freq_time_prof, 0) / 2)), cmap='jet')

    plt.subplot(3, 3, 3)
    csi_fft = np.abs(np.fft.fft(csi[:, 0])) / np.size(csi, 0)
    csi_fft = csi_fft[range(int(len(csi_fft) / 2))]
    plt.plot(np.arange(np.size(csi_fft, 0)), csi_fft)

    # Filter Out Static Component & High Frequency Component
    for jj in range(np.size(csi, 1)):
        csi[:, jj] = signal.lfilter(lu, ld, csi[:, jj])
        csi[:, jj] = signal.lfilter(hu, hd, csi[:, jj])

    plt.subplot(3, 3, 4)
    csi_ratio = signal.lfilter(lu, ld, csi_ratio)
    csi_ratio = signal.lfilter(hu, hd, csi_ratio)
    plt.plot(np.arange(len(csi_ratio)), csi_ratio / max(csi_ratio))

    plt.subplot(3, 3, 6)
    csi_fft = np.abs(np.fft.fft(csi[:, 0])) / np.size(csi, 0)
    csi_fft = csi_fft[range(int(len(csi_fft) / 2))]
    plt.plot(np.arange(np.size(csi_fft, 0)), csi_fft)

    # % TFA With CWT or STFT
    if method == 'stft':
        # f, t, freq_time_prof = signal.spectrogram(csi[:,0],window=signal.get_window('hamming',256),fs=1000,nperseg=256, noverlap=255,return_onesided=False,axis=0)  # 和MATLAB的spectragram()对应
        f, t, freq_time_prof = signal.stft(csi[:, 0], window=signal.get_window('hann', 256), nperseg=256, noverlap=255)  # 和MATLAB的tfrsp()对应
    # plt.figure(ii + 1)
    plt.subplot(3, 3, 5)
    plt.pcolormesh(np.arange(np.size(freq_time_prof, 1)), np.arange(np.size(freq_time_prof, 0)),
                   circshift1D(np.abs(freq_time_prof), int(np.size(freq_time_prof, 0) / 2)), cmap='jet')

    # PCA analysis
    pca = PCA(n_components=10)
    csi_pca_real = pca.fit_transform(csi.real)  # 不支持复数，暂时使用这种方法，和MATLAB不同
    csi_pca_imag = pca.fit_transform(csi.imag)  # 不支持复数，暂时使用这种方法，和MATLAB不同
    csi_pca = np.zeros(np.shape(csi_pca_real), dtype=complex)
    for kk in range(np.size(csi_pca_real, 0)):
        for qq in range(np.size(csi_pca_real, 1)):
            csi_pca[kk, qq] = complex(csi_pca_real[kk, qq], csi_pca_imag[kk, qq])

    if rssi != []:
        plt.subplot(3, 3, 7)
        for i in range(3):
            plt.plot(np.arange(np.size(rssi, 0)), rssi[:, i])

    plt.subplot(3, 3, 8)
    # csi_phase = np.unwrap(np.angle(csi_pca[:, 0]))
    # csi_phase = np.arctan2(csi_pca.imag, csi_pca.real)
    for i in range(3):
        plt.plot(np.arange(np.size(csi_pca, 0)), np.unwrap(np.angle(csi_pca[:, i])))

    plt.subplot(3, 3, 9)
    for i in range(3):
        plt.plot(np.arange(np.size(csi_pca, 0))[1000:1500], np.angle(csi_pca[:, i])[1000:1500])

    plt.pause(0.1)


def display(csi, aver_rssi):
    sample_rate = 1000
    half_rate = sample_rate / 2
    upper_order = 6
    upper_stop = 100  # Hz
    lower_order = 3
    lower_stop = 3  # Hz
    lu, ld = signal.butter(upper_order, upper_stop / half_rate, 'low')
    hu, hd = signal.butter(lower_order, lower_stop / half_rate, 'high')

    step = int(MAXCSI / 100)
    count = 0
    # plt.legend()
    try:
        csi_tmp = []
        csi_tmp2 = []
        for csi_k in csi:  # 载波数  csi_k=(56,3,1)
            if count % step == 0:
                csi_tmp.append(csi_k[1, 0, 0])  # csi_k[0]=3*2
                csi_tmp2.append(csi_k[1, 2, 0])
            count = count + 1

        cc = np.abs(np.array(csi_tmp))  # cc=100*2
        cc2 = np.abs(np.array(csi_tmp2))

        csi_ratio = cc / cc2
        # csi_ratio = signal.lfilter(lu, ld, csi_ratio)
        # csi_ratio = signal.lfilter(hu, hd, csi_ratio)

        lenc = len(cc)
        x = np.arange(0, lenc, 1)
        if lenc <= MAXCSI:
           # plt.plot(x, cc.real)
           # plt.plot(x, cc.imag)
           # plt.plot(x, cc)
            plt.subplot(1, 2, 1)
            plt.plot(np.arange(len(csi_ratio)), csi_ratio)
            plt.ylim([-10, 10])
            plt.subplot(1, 2, 2)
            plt.plot(np.arange(len(aver_rssi)), aver_rssi)
            plt.ylim([25, 75])
        else:
           # plt.plot(x[lenc-1-MAXCSI:], cc.real[lenc-1-MAXCSI:])
           # plt.plot(x[lenc - 1 - MAXCSI:], cc.imag[lenc - 1 - MAXCSI:])
            plt.subplot(1, 2, 1)
            plt.plot(x[lenc - 1 - MAXCSI:], cc[lenc - 1 - MAXCSI:])
            plt.ylim([-10, 10])
            plt.subplot(1, 2, 2)
            plt.plot(x[lenc - 1 - MAXCSI:], aver_rssi[lenc - 1 - MAXCSI:])
            plt.ylim([0, 200])
        plt.show()
        plt.pause(0.01)  # 需要暂停否则无法显示
    except:
        print("display有错误")


def csi_sanitization(csi_data, M, N):
    # M = 3 = A, N = 30 = F
    freq_delta = 2 * 312.5e3

    csi_phase = np.zeros(M*N)
    for ii in range(1, M+1):
        if ii == 1:
            csi_phase[(ii-1)*N: ii*N] = np.unwrap(np.angle(csi_data[(ii-1)*N: ii*N]))
        else:
            csi_diff = np.angle(np.multiply(csi_data[(ii-1)*N: ii*N], np.conj(csi_data[(ii-2)*N: (ii-1)*N])))
            csi_phase[(ii-1)*N: ii*N] = np.unwrap(csi_phase[(ii-2)*N: (ii-1)*N]+csi_diff)

    ai = 2 * np.pi * freq_delta * np.tile(np.arange(0, N), M).reshape([1, N*M])
    bi = np.ones((1, len(csi_phase)))
    ci = csi_phase
    A = np.dot(ai, np.transpose(ai))[0]
    B = np.dot(ai, np.transpose(bi))[0]
    C = np.dot(bi, np.transpose(bi))[0]
    D = np.dot(ai, np.transpose(ci))[0]
    E = np.dot(bi, np.transpose(ci))[0]
    rho_opt = (B * E - C * D) / (A * C - np.square(B))
    beta_opt = (B * D - A * E) / (A * C - np.square(B))

    csi_phase_2 = csi_phase + 2 * np.pi * freq_delta * np.tile(np.arange(0, N), M).reshape([1, N*M]) * rho_opt + beta_opt
    result = np.multiply(np.abs(csi_data), np.exp(1j * csi_phase_2))
    return result


def hampel(X, k=3, nsigma=3):
    length = X.shape[0] - 1
    iLo = np.array([i - k for i in range(0, length + 1)])
    iHi = np.array([i + k for i in range(0, length + 1)])
    iLo[iLo < 0] = 0
    iHi[iHi > length] = length
    xmad = []
    xmedian = []
    for i in range(length + 1):
        w = X[iLo[i]:iHi[i] + 1]
        medj = np.median(w)
        mad = np.median(np.abs(w - medj))
        xmad.append(mad)
        xmedian.append(medj)
    xmad = np.array(xmad)
    xmedian = np.array(xmedian)
    scale = 1.4826  # 缩放
    xsigma = scale * xmad
    xi = ~(np.abs(X - xmedian) <= nsigma * xsigma)  # 找出离群点（即超过nsigma个标准差）

    # 将离群点替换为中为数值
    xf = X.copy()
    xf[xi] = xmedian[xi]
    return xf


"""def main():
    sck = threading.Thread(target=sock())
    dis = threading.Thread(target=display())
    dis.start()
    sck.start()
"""

# 多网卡情况下，根据前缀获取IP
def GetLocalIPByPrefix(prefix):
    localIP = ''
    print(socket.gethostbyname_ex(socket.gethostname()))
    for ip in socket.gethostbyname_ex(socket.gethostname())[1]:
        print(ip)
        if ip.startswith(prefix):
            localIP = ip
    return localIP


if __name__ == '__main__':
    import os
    import numpy as np
    import scipy.io as scio
    from Bfee import Bfee
    from get_scale_csi import get_scale_csi
    from scipy import signal

    # 路径配置，输出改为 BVP_PR
    CSI_ROOT = r"../data/CSI"
    BVP_OUT = r"../data/BVP_PR"
    NUM_TONE = 30
    os.makedirs(BVP_OUT, exist_ok=True)

    # 递归遍历全部dat文件，多层子文件夹自动扫描
    all_dat = []
    for root, _, files in os.walk(CSI_ROOT):
        for fname in files:
            if fname.lower().endswith(".dat"):
                full_path = os.path.join(root, fname)
                all_dat.append(full_path)
    print(f"共检索到 {len(all_dat)} 份CSI原始数据")

    # 批量循环处理每一份样本
    for dat_path in all_dat:
        fname = os.path.basename(dat_path)
        rel_path = os.path.relpath(dat_path, CSI_ROOT)
        rel_mat = rel_path.replace(".dat", ".mat")
        save_path = os.path.join(BVP_OUT, rel_mat)
        # 先创建对应子文件夹
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        try:
            # 1 解析Intel CSI数据包
            bfee = Bfee.from_file_intel(dat_path)
            pkt_list = bfee.dicts
            frame_cnt = len(pkt_list)
            if frame_cnt < 20:
                print(f"跳过短样本 {fname} 帧数={frame_cnt}")
                continue

            # 2 CSI标准化，压缩发射天线维度
            csi_seq = []
            for pkt in pkt_list:
                scaled = get_scale_csi(pkt)
                csi_seq.append(np.squeeze(scaled))
            csi_data = np.array(csi_seq)  # [帧数, 接收天线3, 子载波30]

            # 3 拼接3根天线所有子载波到一维 [帧数, 90]
            frame_num = csi_data.shape[0]
            csi = np.zeros([frame_num, 3 * NUM_TONE], dtype=complex)
            for ant_idx in range(3):
                csi[:, ant_idx * NUM_TONE : (ant_idx + 1) * NUM_TONE] = csi_data[:, :, ant_idx]

            # 4 计算多普勒时频频谱
            doppler_spec, freq_axis = get_doppler_spectrum_file(csi, 1, 3, NUM_TONE, NUM_TONE, 'stft')

            # 5 保存训练特征矩阵
            rel_path = os.path.relpath(dat_path, CSI_ROOT)
            rel_mat = rel_path.replace(".dat", ".mat")
            save_path = os.path.join(BVP_OUT, rel_mat)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            scio.savemat(save_path, {"velocity_spectrum_ro": doppler_spec})
            print(f"完成：{fname} | 帧数 {frame_num} | 频谱最大值 {np.max(np.abs(doppler_spec)):.4f}")

        except Exception as err:
            print(f"处理失败 {fname} 错误信息：{str(err)}")
            continue

    print("\n======== 全部数据集预处理完毕 ========")
    print(f"特征文件输出目录：{BVP_OUT}")

