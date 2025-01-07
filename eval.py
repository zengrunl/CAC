###
# Author: Kai Li
# Date: 2021-06-21 23:29:31
# LastEditors: Kai Li
# LastEditTime: 2021-09-05 22:34:03
###

import re
from typing import OrderedDict
from nichang.utils import tensors_to_device
# from nichang.videomodels import VideoModel, update_frcnn_parameter
from nichang.videomodels.video_process import VideoModel
from nichang.models.test import TF_Patch
# from nichang.models.grid_TFnet import TFGridNetV2
from nichang.datas.avspeech_test import AVSpeechDataset

from nichang.losses import PITLossWrapper, pairwise_neg_sisdr
from nichang.metrics import ALLMetricsTracker
import os
import os.path as osp
import random
import soundfile as sf
import torch
import yaml
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint
import warnings
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

# from nichang.models.avfrcnn_videofrcnn import AVFRCNNVideoFRCNN

parser = argparse.ArgumentParser()
parser.add_argument(
    "-t", "--test_dir", default="/root/data2/LRS3/tt",type=str,
    help="Test directory including the json files"
)
parser.add_argument(
    "-c", "--conf_dir", default="/root/data1/LZR/CTCNet-main/local/lrs2_conf_64_64_3_adamw_1e-1_blocks16_pretrain.yml",
    help="Full path to save best validation model"
)
parser.add_argument(
    "-s", "--save_dir", default="/root/data1/LSR/data/tt/ceshi/",
    help="Full path to save the results wav"
)
parser.add_argument("--exp_dir", default="/root/data1/ceshi/",
                    help="Experiment root")
parser.add_argument(
    "--n_save_ex", type=int, default=2,
    help="Number of audio examples to save, -1 means all"
)


compute_metrics = ["si_sdr", "sdr"]
from collections import defaultdict

def load_ckpt(path, submodule=None):
    _state_dict = torch.load(path, map_location="cpu")['state_dict']
    if submodule is None:
        return _state_dict

    state_dict = OrderedDict()
    for k, v in _state_dict.items():
        if submodule in k:
            L = len(submodule)
            state_dict[k[L+1:]] = v
    return state_dict


def main(conf):
    conf["exp_dir"] = os.path.join(
        "/root/data1/ceshi/")
    conf["test_dir"] = os.path.join(
        "/root/data1/LSR/data/tt/")
    # conf["audionet"].update({"n_src": 1})

    model_path = os.path.join(conf["exp_dir"], "epoch=51-val_loss=-15.72.ckpt")
    print(model_path)

    videomodel = VideoModel(**conf["videonet"])
    audiomodel = TF_Patch(**conf["TFGridNet"])
    # ckpt = torch.load(model_path, map_location="cpu")['state_dict']
    ckpt1 = load_ckpt(model_path, "audio_model")
    audiomodel.load_state_dict(ckpt1)

    # Handle device placement
    audiomodel.eval()
    videomodel.eval()
    audiomodel.cuda()
    videomodel.cuda()
    model_device = next(audiomodel.parameters()).device

    test_set = AVSpeechDataset(
        conf["test_dir"],
        n_src=conf["data"]["nondefault_nsrc"],
        sample_rate=16000,
        segment=None,
        normalize_audio=conf["data"]["normalize_audio"],
        return_src_path=True
    )  # Uses all segment length
    # Used to reorder sources only
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")

    # Randomly choose the indexes of sentences to save.
    ex_save_dir = os.path.join(conf["exp_dir"], "results/")
    os.makedirs(ex_save_dir, exist_ok=True)
    # if conf["n_save_ex"] == -1:
    #     conf["n_save_ex"] = len(test_set)
    metrics = ALLMetricsTracker(
        save_file=os.path.join(ex_save_dir, "metrics.csv"))
    torch.no_grad().__enter__()

    pbar = tqdm(range(len(test_set)))
    results = defaultdict(list)
    for idx in pbar:
        # Forward the network on the mixture.
        mix, sources, target_mouths,src_path = tensors_to_device(test_set[idx], device=model_device)
        # sources = sources[:,:32000]

        # print(target_mouths[idx].shape)
        mouth_emb,_ = videomodel(target_mouths.unsqueeze(0).float())
        est_sources,input_c, input_p,a,batch2,mouth_comp,batch_comp = audiomodel(mix.unsqueeze(0), mouth_emb)
        loss, reordered_sources = loss_func(
            est_sources, sources.unsqueeze(0), return_ests=True)
        mix_np = mix
        sources_np = sources
        est_sources_np = reordered_sources.squeeze(0)
        metrics(mix=mix_np, clean=sources_np, estimate=est_sources_np, key=idx)
        if not (idx % 10):
            pbar.set_postfix(metrics.get_mean())
        current_metrics = metrics.get_mean()
        with open("/root/data1/ceshi/metrics_results_15.72.txt.txt", "a") as f:
            for k, v in current_metrics.items():
                results[k].append(v)  # 将每个指标值保存到对应的字典列表中
                # Write the current metric to the file
                f.write(f"Index {idx} - {k}: {v}\n")


    metrics.final()
    mean, std = metrics.get_mean(), metrics.get_std()
    keys = list(mean.keys() & std.keys())
    
    order = ["sdr_i", "si-snr_i", "pesq", "stoi", "sdr", "si-snr"]
    def get_order(k):
        try:
            ind = order.index(k)
            return ind
        except ValueError:
            return 100

    keys.sort(key=get_order)
    # df = pd.DataFrame(results)  # 将字典转换为 DataFrame
    # df["mean"] = [mean.get(k, None) for k in keys]  # 添加mean列
    # df["std"] = [std.get(k, None) for k in keys]  # 添加std列

    # 将DataFrame保存到Excel文件
    # output_file = "/root/data1/ceshi/metrics_results.xlsx"
    # df.to_excel(output_file, index=False)
    # print(f"Results saved to {output_file}")
    for k in keys:
        m, s = mean[k], std[k]
        print(f"{k}\tmean: {m:.4f}  std: {s:.4f}")
    # Open the file again to write the mean and std values
    with open("/root/data1/ceshi/metrics_results_15.72.txt.txt", "a") as f:
        for k in keys:
            m, s = mean[k], std[k]
            f.write(f"{k}\tmean: {m:.4f}  std: {s:.4f}\n")
            print(f"{k}\tmean: {m:.4f}  std: {s:.4f}")


if __name__ == "__main__":
    from nichang.utils.parser_utils import prepare_parser_from_dict, parse_args_as_dict

    args = parser.parse_args()

    with open(args.conf_dir) as f:
        def_conf = yaml.safe_load(f)

    arg_dic = parse_args_as_dict(parser)
    def_conf.update(arg_dic)
    main(def_conf)
