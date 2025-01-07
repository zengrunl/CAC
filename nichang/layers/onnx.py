import onnxruntime as ort
import numpy as np
import librosa
import torch
import logging
import os
import onnxruntime
logger = logging.getLogger(__name__)
import subprocess
MIN_SIGNAL_LEN = 25


import librosa
import soundfile as sf
import logging
from onnxruntime import InferenceSession

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def sliding_window_cmvn(mfcc, window_size=300, norm_vars=False):
    num_frames, num_coeffs = mfcc.shape
    half_window = window_size // 2
    cmvn_mfcc = np.copy(mfcc)
    for i in range(num_frames):
        start = max(0, i - half_window)
        end = min(num_frames, i + half_window)
        window = mfcc[start:end, :]
        mean = np.mean(window, axis=0)
        cmvn_mfcc[i, :] -= mean
        if norm_vars:
            std = np.std(window, axis=0) + 1e-10
            cmvn_mfcc[i, :] /= std
    return cmvn_mfcc

def extract_mfcc_librosa(audio_path, config):
    data, samplerate = sf.read(audio_path)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    if samplerate != config['sample_frequency']:
        data = librosa.resample(data, orig_sr=samplerate, target_sr=config['sample_frequency'])
    n_fft = int(config['frame_length'] * config['sample_frequency'])
    hop_length = int(config['frame_step'] * config['sample_frequency'])
    mfcc = librosa.feature.mfcc(y=data,
                                sr=config['sample_frequency'],
                                n_mfcc=config['num_ceps'],
                                n_fft=n_fft,
                                hop_length=hop_length,
                                fmin=config['low_freq'],
                                fmax=config['high_freq'],
                                center=config['center'],
                                window='hamming')
    mfcc = mfcc.T.astype(np.float32)
    if config['apply_cmvn_sliding']:
        mfcc = sliding_window_cmvn(mfcc, config['cmn_window'], config['norm_vars'])
    return mfcc

class XVectorExtractor:
    def __init__(self, model_path):
        self.session = InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def extract(self, mfcc_features):
        input_tensor = np.expand_dims(mfcc_features, axis=0).transpose(0, 2, 1).astype(np.float32)  # [batch_size, features, time]
        outputs = self.session.run(None, {self.input_name: input_tensor})
        x_vector = outputs[0]
        return x_vector.squeeze()

def extract_embeddings(features_dict, embedding_extractor):
    embeddings = []
    for (start, end), feature in features_dict.items():
        embedding = embedding_extractor.extract(feature)
        embeddings.append(embedding)
    return np.array(embeddings)

def process_file(file_name, wav_dir, out_dir, xvector_extractor, config, wav_suffix='.wav'):
    # logger.info('Processing file {}.'.format(file_name.split()[0]))
    # num_speakers = None
    # if len(file_name.split()) > 1:
    #     file_name, num_speakers = file_name.split()[0], int(file_name.split()[1])

    # wav_dir, out_dir = os.path.abspath(wav_dir), os.path.abspath(out_dir)

    # 提取MFCC特征
    # audio_path = os.path.join(wav_dir, f'{file_name}{wav_suffix}')
    mfcc_features = extract_mfcc_librosa(wav_dir, config)
    logger.info(f'Extracted MFCC features with shape: {mfcc_features.shape}')

    # 由于音频始终有活动，直接对整个音频提取嵌入
    features_dict = {("0", "2.0"): mfcc_features}  # 假设音频长度为2秒

    # 提取嵌入
    embedding_set = extract_embeddings(features_dict, xvector_extractor)

    # 保存嵌入
    out_path = os.path.join(out_dir, f'{file_name}_embedding.npy')
    np.save(out_path, embedding_set)
    logger.info(f'Embedding saved to {out_path}')

if __name__ == "__main__":
    # 配置路径
    wav_dir = 'path/to/your/wav_dir'  # 替换为实际路径
    out_dir = 'path/to/your/out_dir'  # 替换为实际路径
    file_name = 'your_audio_file 1'    # 示例文件名，包含说话人数（可选）

    # 配置参数
    config = {
        'sample_frequency': 16000,
        'frame_length': 0.025,    # 25 ms
        'frame_step': 0.010,      # 10 ms
        'low_freq': 20,
        'high_freq': 7700,
        'num_ceps': 23,
        'snip_edges': False,
        'apply_cmvn_sliding': True,
        'norm_vars': False,
        'center': True,
        'cmn_window': 300
    }

    # 初始化x-vector提取器
    model_path = 'path/to/final.onnx'   # 替换为实际路径
    xvector_extractor = XVectorExtractor(model_path)

    # 处理文件
    process_file(file_name, wav_dir, out_dir, xvector_extractor, config)
