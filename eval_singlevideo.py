###
# Author: Kai Li
# Date: 2022-04-03 08:50:42
# Email: lk21@mails.tsinghua.edu.cn
# LastEditTime: 2022-04-03 18:02:56
###
###
# Author: Kai Li
# Date: 2021-06-21 23:29:31
# LastEditors: Please set LastEditors
# LastEditTime: 2021-11-07 23:17:39
###

from typing import OrderedDict
from nichang.videomodels import VideoModel
from nichang.models.ctcnet import CTCNet
from nichang.datas.transform import get_preprocessing_pipelines
import os
import soundfile as sf
import torch
import yaml
import argparse
import numpy as np
from torch.utils import data
import warnings
warnings.filterwarnings("ignore")
from nichang.models.test import TF_Patch

parser = argparse.ArgumentParser()
parser.add_argument(
    "-c", "--conf_dir", default="/root/data1/LZR/CTCNet-main/local/lrs2_conf_64_64_3_adamw_1e-1_blocks16_pretrain.yml",
    help="Full path to save best validation model"
)
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
    # conf["exp_dir"] = os.path.join(
    #     "exp", conf["log"]["exp_name"])
    # conf["audionet"].update({"n_src": 1})

    model_path = os.path.join(conf["exp_dir"], "epoch=28-val_loss=-15.39.ckpt")
    # model_path = "exp/vox2_10w_frcnn2_64_64_3_adamw_1e-1_blocks16_pretrain/best_model.pth"
    sample_rate = conf["data"]["sample_rate"]
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

    # Randomly choose the indexes of sentences to save.
    torch.no_grad().__enter__()
    for idx in range(1, 2):
        spk, sr = sf.read("/root/data1/LSR/LSR2/lrs2_rebuild/audio/wav16k/min/tt/mix/6330311066473698535_00011_0.53084_6339356267468615354_00010_-0.53084.wav", dtype="float32")
        mouth = get_preprocessing_pipelines()["val"](np.load("/root/data1/LSR/LSR2/lrs2_rebuild/mouths/6330311066473698535_00011.npz".format(idx))["data"])
        key = "spk{}".format(idx)
        
        # Forward the network on the mixture.
        target_mouths = torch.from_numpy(mouth).to(model_device)
        mix = torch.from_numpy(spk).to(model_device)
        # import pdb; pdb.set_trace()
        mouth_emb = videomodel(target_mouths.unsqueeze(0).unsqueeze(1).float())
        est_sources = audiomodel(mix[None], mouth_emb)

        gt_dir = "/root/data1/test/sep_result/"
        os.makedirs(gt_dir, exist_ok=True)
        # import pdb; pdb.set_trace()
        sf.write(os.path.join(gt_dir, key+".wav"), est_sources.squeeze(0).squeeze(0).cpu().numpy(), 16000)
        # import pdb; pdb.set_trace()


if __name__ == "__main__":
    from nichang.utils.parser_utils import prepare_parser_from_dict, parse_args_as_dict

    args = parser.parse_args()

    with open(args.conf_dir) as f:
        def_conf = yaml.safe_load(f)

    arg_dic = parse_args_as_dict(parser)
    def_conf.update(arg_dic)
    main(def_conf)
