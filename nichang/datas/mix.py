import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch import hub
from torch.utils.data import Dataset, DataLoader
import random as random
import os
import shutil
import zipfile
import json
from typing import Dict, Iterable, List, Iterator
# from .transform import get_preprocessing_pipelines
from random import randrange
import h5py
import sys
import scipy.io.wavfile as wavfile
import cv2
MAX_INT16 = np.iinfo(np.int16).max

def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)

def get_mouthroi_audio_pair(mouthroi, audio, window, num_of_mouthroi_frames, audio_sampling_rate):
    '''
    唇部与音频对齐
    '''
    audio_start = 0
    audio_sample = audio[audio_start:(audio_start+window)]
    frame_index_start = 0
    mouthroi = mouthroi[frame_index_start:(frame_index_start + num_of_mouthroi_frames), :, :]
    return mouthroi, audio_sample

def load_mouthroi(filename):
    try:
        if filename.endswith('npz'):
            return np.load(filename)['data']
        elif filename.endswith('h5'):
            with h5py.File(filename, 'r') as hf:
                return hf["data"][:]
        else:
            return np.load(filename,allow_pickle=True)
    except IOError:
        print( "Error when reading file: {}".format(filename) )
        sys.exit()

def augment_audio(audio):
    audio = audio * (random.random() * 0.2 + 0.9) # 0.9 - 1.1
    audio[audio > 1.] = 1.
    audio[audio < -1.] = -1.
    return audio
def normalize(samples, desired_rms = 0.1, eps = 1e-4):
  rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
  samples = samples * (desired_rms / rms)
  return samples
def write_wav(fname, samps, sampling_rate=16000, normalize=True):
	"""
	Write wav files in int16, support single/multi-channel
	"""
	# for multi-channel, accept ndarray [Nsamples, Nchannels]
	if samps.ndim != 1 and samps.shape[0] < samps.shape[1]:
		samps = np.transpose(samps)
		samps = np.squeeze(samps)
	# same as MATLAB and kaldi
	if normalize:
		samps = samps * MAX_INT16
		samps = samps.astype(np.int16)
	fdir = os.path.dirname(fname)
	if fdir and not os.path.exists(fdir):
		os.makedirs(fdir)
	# NOTE: librosa 0.6.0 seems could not write non-float narray
	#       so use scipy.io.wavfile instead
	wavfile.write(fname, sampling_rate, samps)


def extract_paths_from_scp(scp_file):
    paths = []
    with open(scp_file, 'r') as file:
        for line in file:
            # 移除'/mnt'之后的内容
            path = line.split('/root')[0]
            # 分割成两个部分
            parts = path.split('#')

            id1_path, id2_path = parts
            # 将加号('+')替换为正斜杠('/')
            audio_path = "/root/data2/data/voxceleb2/audio/dev/aac/"
            video_path = "/root/data2/data/voxceleb2/video/dev/mp4/"
            lip_embedding_direc = '/root/data2/data/voxceleb2-mouth/train/VoxCeleb2/mouth_roi/train_mouth_roi/'
            formatted_id1_path = id1_path.replace('+', '/').rstrip()
            formatted_id2_path = id2_path.replace('+', '/').rstrip()
            # 添加到结果列表中
            if formatted_id1_path == 'id04247/4oEh89WmoDo/00043'or formatted_id2_path == 'id04247/4oEh89WmoDo/00043':
                print('file_A')
                # formatted_id1_path = formatted_id1_path.replace('id04247/4oEh89WmoDo/00043', '/id04247/5p5uG5OdEdE/00061')

            else:
                mouthroi_A = load_mouthroi(lip_embedding_direc + formatted_id1_path+'.h5')
                mouthroi_B = load_mouthroi(lip_embedding_direc + formatted_id2_path+'.h5')
            # paths.append((formatted_id1_path, formatted_id2_path))

    return paths

# 假设'scp_file.txt'是你的scp文件名
scp_file = '/root/data1/LZR/AV-Sepformer-master/data_list/train_new.scp'
extracted_paths = extract_paths_from_scp(scp_file)

# 打印提取出的路径


if __name__ == '__main__':
    scp_file = "/root/data1/LZR/AV-Sepformer-master/data_list/train_new.scp"
    extract_paths_from_scp(scp_file)
    for path_pair in extracted_paths:
        print(f"Path 1: {path_pair[0]}")
        print(f"Path 2: {path_pair[1]}")