
##########################################
# 1.语音分离网络（自监督，利用AV相关度）   半监督
import sys
sys.path.append('../')
from torch.utils.data import Dataset
# import torchaudio
import numpy as np
import torch
import torch.nn.functional as F
import cv2 as cv
import os
import h5py
# from torchvision import transforms
import soundfile as sf
# import speechpy
# from util import handle_scp
def handle_scp(scp_path):
    '''
    Read scp file script
    input:
          scp_path: .scp file's file path
    output:
          scp_dict: {'key':'wave file path'}
    '''
    scp_dict = dict()
    line = 0
    lines = open(scp_path, 'r').readlines()
    for l in lines:
        scp_parts = l.strip().split()
        line += 1
        if len(scp_parts) != 2:
            raise RuntimeError("For {}, format error in line[{:d}]: {}".format(
                scp_path, line, scp_parts))
        if len(scp_parts) == 2:
            key, value = scp_parts
        # if key in scp_dict:
        #     raise ValueError("Duplicated key \'{0}\' exists in {1}".format(
        #         key, scp_path))

        scp_dict[key] = value

    return scp_dict


def check_parameters(net):
    '''
        Returns module parameters. Mb
    '''
    parameters = sum(param.numel() for param in net.parameters())    # param.numel() 统计模型参数量
    return parameters / 10**6


def check_parameters_amount(net):
    # 参数量   百万级M
    total = sum([param.nelement() for param in net.parameters()])
    return total / 1e6
    # print("Number of parameter: %.2fM" % (total / 1e6))

def read_wav(fname, return_rate=False):
    '''
         Read wavfile using Pytorch audio
         input:
               fname: wav file path
               return_rate: Whether to return the sampling rate
         output:
                src: output tensor of size C x L
                     L is the number of audio frames
                     C is the number of channels.
                sr: sample rate
    '''
    src, sr = sf.read(fname,dtype="float32")
    # src = src[0, 0:24000]  # VoxCeleb2数据集 取第一通道的前24000个
    # src = src.squeeze()
    # src = (src - torch.mean(src)) / torch.max(torch.abs(src))   # 在预处理voxceleb2数据集时已经对音频进行了归一化并保存，所以此处不必再进行归一化
    if return_rate:
        return src, sr
    else:
        return src   # 去掉维度为1的维度，对数据维数进行压缩


def read_img(input_Path):
    # print('input_Path:', input_Path)
    # transform = transforms.Compose([transforms.ToTensor()])
    img_paths = []
    for (path, dirs, files) in os.walk(input_Path):
        # print('files:', files)
        files.sort(key=lambda x:int(x.split('.')[0][-2:]))
        for filename in files:
            if filename.endswith(('.jpg','.png')):
                img_paths.append(path+'/'+filename)
    img_data = []
    # print('img_paths:', img_paths)
    for im in img_paths:
        frame = cv.imread(im)  # 读取为BGR格式0-255，默认以原始图像格式读取，彩色即为彩色，灰度即为灰度
        # print('BGR:', frame)   # H, W, C   BGR
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)   # 颜色空间转换函数numpy.ndarray  gray=0.3*R+0.59*G+0.11*B，所以范围仍是0-255
        # print('gray:', frame)  # H, W
        # frame = cv.resize(frame, (100, 60), interpolation=cv.INTER_CUBIC)
        frame = (frame - np.mean(frame)) / np.std(frame)
        # frame = transform(frame)   # 将H,W,(C)的numpy.ndarray转换为C,H,W的tensor，同时每个值归一化至[0,1]
        # print('min-max:', frame, frame.size())
        # frame = torch.squeeze(frame)  # 将通道C为1的通道去掉，只保留H和W
        # print(frame, frame.size())
        frame = torch.from_numpy(frame)
        # print('frame1:', frame.size())   # torch.Size([60, 100])
        img_data.append(frame)
    # print('img_data:', img_data)
    img_data = torch.stack(img_data)
    # print('img_data1:', img_data, img_data.size())   #  torch.Size([75, 60, 100])
    return img_data


class Datasets(Dataset):
    '''
       Load audio and image data
       mix_scp: file path of mix audio (type: str)
       ref_scp: file path of ground truth audio (type: list[spk1,spk2])
    '''
    def __init__(self, mix_scp=None, real_scp=None,ref_scp=None):
        super(Datasets, self).__init__()

        self.mix_audio_pathlist = handle_scp(mix_scp)
        self.mix_audio_keyslist = list(self.mix_audio_pathlist.keys())
        self.ref_visual1_pathlist = handle_scp(ref_scp[0])
        self.ref_visual1_keyslist = list(self.ref_visual1_pathlist.keys())
        self.ref_visual2_pathlist = handle_scp(ref_scp[1])
        self.ref_visual2_keyslist = list(self.ref_visual2_pathlist.keys())
        self.real_audio1_pathlist = handle_scp(real_scp[0])
        self.real_audio1_keyslist = list(self.real_audio1_pathlist.keys())
        self.real_audio2_pathlist = handle_scp(real_scp[1])
        self.real_audio2_keyslist = list(self.real_audio2_pathlist.keys())

    def __len__(self):
        return len(self.mix_audio_pathlist)

    def __getitem__(self, index):
        mix_audio_path = self.mix_audio_pathlist[self.mix_audio_keyslist[index]]
        mix_audio = read_wav(mix_audio_path)
        ref_visual1_path = self.ref_visual1_pathlist[self.ref_visual1_keyslist[index]]
        ref_visual1 = read_img(ref_visual1_path)
        ref_visual2_path = self.ref_visual2_pathlist[self.ref_visual2_keyslist[index]]
        ref_visual2 = read_img(ref_visual2_path)
        real_audio1_path = self.real_audio1_pathlist[self.real_audio1_keyslist[index]]
        real_audio1 = read_wav(real_audio1_path)
        real_audio2_path = self.real_audio2_pathlist[self.real_audio2_keyslist[index]]
        real_audio2 = read_wav(real_audio2_path)
        # print(mix_audio.size(), ref_visual1.size(), real_audio1.size())
        sources = torch.stack([torch.from_numpy(real_audio1),torch.from_numpy(real_audio2)])
        return mix_audio, sources


if __name__ == "__main__":
    dataset = Datasets('/home/LYG/AVSS/DataBasemixSNR/create_scp2_selfsupervised_av/tr_mix.scp',
               ['/home/LYG/AVSS/DataBasemixSNR/create_scp2_selfsupervised_av/tr_v_1.scp','/home/LYG/AVSS/DataBasemixSNR/create_scp2_selfsupervised_av/tr_v_2.scp'],
               ['/home/LYG/AVSS/DataBasemixSNR/create_scp2_selfsupervised_av/tr_1.scp','/home/LYG/AVSS/DataBasemixSNR/create_scp2_selfsupervised_av/tr_2.scp'])
    print(dataset.__getitem__(0))
    # for i in range(77594):
    #     dataset.__getitem__(i)
    print('succesful')
    
    


