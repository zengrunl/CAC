from memory_profiler import profile
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
import torchvision.transforms as transforms
from typing import Dict, Iterable, List, Iterator
from .transform import get_preprocessing_pipelines
from PIL import Image, ImageEnhance, ImageOps
from ..utils.video_reader import VideoReader
from random import randrange
import h5py
import sys
import librosa
import onnxruntime as ort
from onnxruntime import InferenceSession
import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# Model is downloaded from the speechbrain HuggingFace repo
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
classifier.eval()
for param in classifier.parameters():
    param.requires_grad = False
# classifier = EncoderClassifier.from_hparams(
#     source="speechbrain/spkrec-ecapa-voxceleb",
#     savedir="/root/data1/save/",/root/data1/LZR/x-vector/
# )


class XVectorExtractor:
    def __init__(self, model_path):
        self.session = InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def extract(self, mfcc_features):
        input_tensor = np.expand_dims(mfcc_features, axis=0).transpose(0, 2, 1).astype(np.float32)  # [batch_size, features, time]
        outputs = self.session.run(None, {self.input_name: input_tensor})
        x_vector = outputs[0]
        return x_vector.squeeze()
config = {
    'sample_frequency': 16000,
    'frame_length': 0.025,  # 25 ms
    'frame_step': 0.010,  # 10 ms
    'low_freq': 20,
    'high_freq': 7700,
    'num_ceps': 23,
    'snip_edges': False,
    'apply_cmvn_sliding': True,
    'norm_vars': False,
    'center': True,
    'cmn_window': 300
}

def extract_embeddings(features_dict, embedding_extractor):
    embeddings = []
    for (start, end), feature in features_dict.items():
        embedding = embedding_extractor.extract(feature)
        embeddings.append(embedding)
    return np.array(embeddings)
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
def extract_mfcc_librosa(data, config):
    # data, samplerate = sf.read(audio_path)
    # if len(data.shape) > 1:
    #     data = np.mean(data, axis=1)
    # if samplerate != config['sample_frequency']:
    #     data = librosa.resample(data, orig_sr=samplerate, target_sr=config['sample_frequency'])
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
    desired_frames = 100
    current_frames = mfcc.shape[0]
    if current_frames > desired_frames:
        # 截断
        mfcc = mfcc[:desired_frames, :]
    return mfcc
def generate_spectrogram_complex(audio, stft_frame, stft_hop, n_fft):
    spectro = librosa.core.stft(audio, hop_length=stft_hop, n_fft=n_fft, win_length=stft_frame, center=True)
    real = np.expand_dims(np.real(spectro), axis=0)
    imag = np.expand_dims(np.imag(spectro), axis=0)
    spectro_two_channel = np.concatenate((real, imag), axis=0)
    return spectro_two_channel
def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)

def load_frame(clip_path):
    video_reader = VideoReader(clip_path, 1)
    start_pts, time_base, total_num_frames = video_reader._compute_video_stats()
    end_frame_index = total_num_frames - 1
    if end_frame_index < 0:
        clip, _ = video_reader.read(start_pts, 1)
    else:
        clip, _ = video_reader.read(random.randint(0, end_frame_index) * time_base, 1)
    frame = Image.fromarray(np.uint8(clip[0].to_rgb().to_ndarray())).convert('RGB')
    return frame
def augment_image(image):
    if(random.random() < 0.5):
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.random()*0.6 + 0.7)
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(random.random()*0.6 + 0.7)
    return image
class AVSpeechDataset(Dataset):
    def __init__(
        self,
        json_dir: str = "",
        n_src: int = 2,
        sample_rate: int = 8000,
        segment: float = 4.0,
        normalize_audio: bool = False,
        return_src_path: bool = False,
        mode: str = ""
    ):
        super().__init__()
        if json_dir == None:
            raise ValueError("JSON DIR is None!")
        if n_src not in [1, 2]:
            raise ValueError("{} is not in [1, 2]".format(n_src))
        self.json_dir = json_dir
        self.mode = mode
        self.sample_rate = sample_rate
        self.normalize_audio = normalize_audio
        self.return_src_path = return_src_path
        self.lipreading_preprocessing_func = get_preprocessing_pipelines()[
            "train" if segment != None else "val"
        ]
        if segment is None:
            self.seg_len = None
        else:
            self.seg_len = int(segment * sample_rate)
        self.n_src = n_src
        self.test = self.seg_len is None
        mix_json = os.path.join(json_dir, "mix.json")
        sources_json = [os.path.join(json_dir, source + ".json") for source in ["s1", "s2"]]

        with open(mix_json, "r") as f:
            mix_infos = json.load(f)
        sources_infos = []
        for src_json in sources_json:
            with open(src_json, "r") as f:
                sources_infos.append(json.load(f))

        self.mix = []
        self.sources = []

        if self.n_src == 1:
            orig_len = len(mix_infos) * 2
            drop_utt, drop_len = 0, 0
            if not self.test:
                for i in range(len(mix_infos) - 1, -1, -1):
                    if mix_infos[i][1] < self.seg_len:
                        drop_utt = drop_utt + 1
                        drop_len = drop_len + mix_infos[i][1]
                        del mix_infos[i]
                        for src_inf in sources_infos:
                            del src_inf[i]
                    else:
                        for src_inf in sources_infos:
                            self.mix.append(mix_infos[i])
                            self.sources.append(src_inf[i])
            else:
                for i in range(len(mix_infos)):
                    for src_inf in sources_infos:
                        self.mix.append(mix_infos[i])
                        self.sources.append(src_inf[i])

            print(
                "Drop {} utts({:.2f} h) from {} (shorter than {} samples)".format(
                    drop_utt, drop_len / sample_rate / 3600, orig_len, self.seg_len
                )
            )
            self.length = orig_len

        elif self.n_src == 2:
            orig_len = len(mix_infos)
            drop_utt, drop_len = 0, 0
            if not self.test:
                for i in range(len(mix_infos) - 1, -1, -1):
                    if mix_infos[i][1] < self.seg_len:
                        drop_utt = drop_utt + 1
                        drop_len = drop_len + mix_infos[i][1]
                        del mix_infos[i]
                        for src_inf in sources_infos:
                            del src_inf[i]
                    else:
                        self.mix.append(mix_infos[i])
                        self.sources.append([src_inf[i] for src_inf in sources_infos])
            else:
                self.mix = mix_infos
                self.sources = sources_infos
            print(
                "Drop {} utts({:.2f} h) from {} (shorter than {} samples)".format(
                    drop_utt, drop_len / sample_rate / 3600, orig_len, self.seg_len
                )
            )
            self.length = orig_len

            vision_transform_list = [transforms.Resize(224), transforms.ToTensor()]
            self.vision_transform = transforms.Compose(vision_transform_list)

    def __len__(self):
        return self.length

    # @profile(precision=5)
    def __getitem__(self, idx: int):
        self.EPS = 1e-8
        if self.n_src == 1:
            # print(self.test, self.seg_len, self.mix[idx])
            # if self.mix[idx][1] == self.seg_len or self.test:
            #     rand_start = 0
            # else:
            #     rand_start = np.random.randint(0, self.mix[idx][1] - self.seg_len)

            # if self.test:
            #     stop = None
            # else:
            #     stop = rand_start + self.seg_len
            rand_start = 0
            stop = self.seg_len

            mix_source, _ = sf.read(self.mix[idx][0], start=rand_start, stop=stop, dtype="float32")
            source = sf.read(self.sources[idx][0], start=rand_start, stop=stop, dtype="float32")[0]
            source_mouth = self.lipreading_preprocessing_func(np.load(self.sources[idx][1])["data"])

            source = torch.from_numpy(source)
            mixture = torch.from_numpy(mix_source)

            if self.normalize_audio:
                m_std = mixture.std(-1, keepdim=True)
                mixture = normalize_tensor_wav(mixture, eps=self.EPS, std=m_std)
                source = normalize_tensor_wav(source, eps=self.EPS, std=m_std)
#             return mixture, source, torch.stack([torch.from_numpy(source_mouth)]), self.mix[idx][0].split("/")[-1]
#             print(self.sample_rate*2, mixture.shape, source.shape)"
            if self.return_src_path:
                return mixture[:self.sample_rate*2], source[:self.sample_rate*2], \
                    torch.stack([torch.from_numpy(source_mouth)]), \
                    self.mix[idx][0].split("/")[-1], \
                    self.sources[idx][0]
            else:
                return mixture[:self.sample_rate*2], source[:self.sample_rate*2], \
                    torch.stack([torch.from_numpy(source_mouth)]), \
                    self.mix[idx][0].split("/")[-1]

        if self.n_src == 2:
            if self.mix[idx][1] == self.seg_len or self.test:
                rand_start = 0
            else:
                rand_start =   np.random.randint(0, self.mix[idx][2] - self.seg_len)

            if self.test:
                stop = None
            else:
                stop = rand_start + self.seg_len
            assert rand_start == 0

            mix_source, _ = sf.read(self.mix[idx][0], start=rand_start, stop=stop, dtype="float32")
            sources = []
            embeddings = []
            for src in self.sources[idx]:
                # import pdb; pdb.set_trace()
                sources.append(sf.read(src[0], start=rand_start, stop=stop, dtype="float32")[0])
                embeddings.append(torchaudio.load(src[0])[0])


            # embedding = model.encode_file(sources[0])
            # embedding = model.encode_file(sources[1])
            with torch.no_grad():
                embeddings_0 = classifier.encode_batch(embeddings[0]).detach() #1,1,192
                embeddings_1 =  classifier.encode_batch(embeddings[1]).detach() #1,1,192


            # mfcc_features_0 = extract_mfcc_librosa(sources[0], config)
            # mfcc_features_1 = extract_mfcc_librosa(sources[1], config)
            sources_mouths = [
                torch.from_numpy(self.lipreading_preprocessing_func(np.load(src[1])["data"]))
                for src in self.sources[idx]
            ]
            # import pdb; pdb.set_trace()
            sources = torch.stack([torch.from_numpy(source) for source in sources])
            mixture = torch.from_numpy(mix_source)



            # # 由于音频始终有活动，直接对整个音频提取嵌入
            # features_dict_0 = {("0", "2.0"): mfcc_features_0}  # 假设音频长度为2秒
            # features_dict_1 = {("0", "2.0"): mfcc_features_1}  # 假设音频长度为2秒
            # xvector_extractor = XVectorExtractor("/root/data1/LZR/VBDiarization-master/models/final.onnx")
            # # 提取嵌入
            # embedding_set_0 = extract_embeddings(features_dict_0, xvector_extractor)
            # embedding_set_1 = extract_embeddings(features_dict_1, xvector_extractor)
            # embedding_set_0 = torch.from_numpy(embedding_set_0)
            # embedding_set_1 = torch.from_numpy(embedding_set_1) #1，512

            embedding = torch.cat([embeddings_0, embeddings_1], dim=0)  #2，1,192
            embedding = embedding.squeeze()#2,192

            if self.normalize_audio:
                m_std = mixture.std(-1, keepdim=True)
                mixture = normalize_tensor_wav(mixture, eps=self.EPS, std=m_std)
                sources = normalize_tensor_wav(sources, eps=self.EPS, std=m_std)

            audio_spec_1 = generate_spectrogram_complex(np.array(sources[0]), 128, stft_hop=64,
                                                         n_fft=128)
            audio_spec_2 = generate_spectrogram_complex(np.array(sources[1]), 128, stft_hop=64,
                                                         n_fft=128)
            audio_spec_1 = torch.from_numpy(audio_spec_1).transpose(1,2)
            audio_spec_2 = torch.from_numpy(audio_spec_2).transpose(1,2)
            audio_spec = torch.cat([audio_spec_1, audio_spec_2],dim=0)

            
#             return mixture, sources, torch.stack(sources_mouths), self.mix[idx][0].split("/")[-1]
#             print(self.sample_rate*2, mixture.shape)
            return mixture[:self.sample_rate*2], sources[:self.sample_rate*2], \
                torch.stack(sources_mouths), self.mix[idx][0].split("/")[-1],audio_spec,embedding


