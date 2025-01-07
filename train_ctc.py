import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import json
import random
import yaml
from pprint import pprint
from nichang.utils.parser_utils import prepare_parser_from_dict, parse_args_as_dict
import torch

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch.nn as nn
import torch.nn.functional as F
from nichang.datas import AVSpeechDataset
from nichang.system.optimizers import make_optimizer
from nichang.system.core import System
# from nichang.system.tensorboard import TensorBoardLogger
from nichang.losses import PITLossWrapper, pairwise_neg_sisdr, pairwise_neg_snr
from nichang.models.ctcnet import CTCNet
from nichang.videomodels.video_process import VideoModel
from nichang.videomodels.face import build_facial
from nichang.models.dual_path import Cross_Sepformer_warpper
from pytorch_lightning.loggers import TensorBoardLogger
# from nichang.models.sepformer import Cross_Sepformer_warpper_1
# from nichang.models.sand import Sandglasset
from nichang.models.av_conv import AV_model
# from nichang.models.dual_rnn import Dual_RNN_model
# from nichang.models.mamba import SPMamba
# from nichang.models.tf_gridnet import TFGridNet, TFGridNet_S6
# from nichang.models.S6 import TF_S6,TFGridNet_M
# from nichang.models.patch_817 import TF_Patch
from nichang.models.circle_911 import TF_Patch
from nichang.models.grid_TFnet import TFGridNetV2
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]
import tracemalloc
from scipy.optimize import linear_sum_assignment
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# By default train.py will use all available GPUs. The `id` option in run.sh
# will limit the number of available GPUs for train.py .
parser = argparse.ArgumentParser()
parser.add_argument(
    "-e", "--exp_dir", default="exp/tmp", 
    help="Full path to save best validation model")
parser.add_argument(
    "-c", "--conf_dir", default="/root/data1/LZR/CTCNet-main/local/lrs2_conf_64_64_3_adamw_1e-1_blocks16_pretrain.yml",
    help="Full path to save best validation model")
parser.add_argument(
    "-n", "--name", default=None, 
    help="Experi ment name")
parser.add_argument(
    "--gpus", type= int, default=8,
    help="#gpus of each node")
parser.add_argument(
    "--nodes", type=int, default=1,
    help="#node")
pl.seed_everything(3407)
class BaseLoss(nn.Module):
    def __init__(self):
        super(BaseLoss, self).__init__()
    def forward(self, preds, targets, weight=None):
        if isinstance(preds, list):
            N = len(preds)
            if weight is None:
                weight = preds[0].new_ones(1)

            errs = [self._forward(preds[n], targets[n], weight[n])
                    for n in range(N)]
            err = torch.mean(torch.stack(errs))

        elif isinstance(preds, torch.Tensor):
            if weight is None:
                weight = preds.new_ones(1)
            err = self._forward(preds, targets, weight)
        return err
class TripletLossCosine(BaseLoss):
    """
    Triplet loss with cosine distance
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLossCosine, self).__init__()
        self.margin = margin

    def _forward(self, anchor, positive, negative, size_average=True):
        # print("distance_positive:", positive.shape)
        # print("distance_anchor:", anchor.shape)
        # print("distance_negative:", negative.shape)
        distance_positive = 1 - F.cosine_similarity(anchor, positive)  #[0,2]
        distance_negative= 1 - F. cosine_similarity(anchor, negative)
        losses =  F.relu((distance_positive - distance_negative) + self.margin)#[0,2]
        # print("distance_positive:",distance_positive)
        # print("distance_negative:",distance_negative)
        return losses.mean() if size_average else losses.sum()

def build_dataloaders(conf):
    train_set = AVSpeechDataset(
        conf["data"]["train_dir"],
        sample_rate=conf["data"]["sample_rate"],
        segment=conf["data"]["segment"],
        normalize_audio=conf["data"]["normalize_audio"],
        mode="train",

    )
    val_set = AVSpeechDataset(
        conf["data"]["valid_dir"],
        n_src=conf["data"]["nondefault_nsrc"],
        sample_rate=conf["data"]["sample_rate"],
        segment=conf["data"]["segment"],
        normalize_audio=conf["data"]["normalize_audio"],

    )
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
        pin_memory=True,

    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
        pin_memory=True,
    )
    return train_loader, val_loader



def main(conf):

    train_loader, val_loader = build_dataloaders(conf)
    # Define model and optimizer
    sample_rate = conf["data"]["sample_rate"]
    videomodel = VideoModel(**conf["videonet"])
    audiomodel = TF_Patch(**conf["TFGridNet"])
    optimizer = make_optimizer(audiomodel.parameters(), **conf["optim"])
    # Define scheduler

    scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=1)

    # Just after instantiating, save the args. Easy loading in the future.
    exp_dir = os.path.join("exp", conf["log"]["exp_name"])
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)
    # Define Loss function.
    loss_func = {
        # "train": PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx"),
        "train": PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx"),
        "val": PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx"),
        "train1": PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx"),
        "val1": PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx"),
        "tripletlosscosine":TripletLossCosine(margin=0.3),
        "losscosine": nn.CosineSimilarity(dim=1),
        "losscosine1": nn.CosineSimilarity(dim=1),

        }
    # train_loader1, val_loader1 = build_dataloaders(conf)
    system = System(
        audio_model=audiomodel,
        video_model=videomodel,
        face_model = None,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        config=conf,
    )
    # Define callbacks
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(
        checkpoint_dir,
        filename="{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=5,
        verbose=True,
        save_last=True,
    )
    callbacks += [checkpoint]
    if conf["training"]["early_stop"]:
        callbacks+=[EarlyStopping(monitor="val_loss", mode="min", patience=4, verbose=True)]

    # distributed_backend = "ddp"
    print(os.environ)
    print(torch.cuda.device_count())

    # default logger used by trainer
    os.makedirs(conf["log"]["path"], exist_ok=True)
    comet_logger = TensorBoardLogger(save_dir="/root/data1/LZR/CTCNet-main/log/tmp", name="experiment_name")
    trainer = pl.Trainer(
        max_epochs=conf["training"]["epochs"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        devices=[0],
        # strategy='ddp_spawn',
        # devices=1,
        accelerator="cuda",
        # strategy = 'ddp',
        num_nodes=1,
       # Useful for fast experiment
        gradient_clip_val=5.0,
        logger=comet_logger,
        sync_batchnorm=True,
        check_val_every_n_epoch=1,

    )
    trainer.fit(system)

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.audio_model.serialize()
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))


if __name__ == "__main__":

    args = parser.parse_args()

    with open(args.conf_dir) as f:
        def_conf = yaml.safe_load(f)
    if args.name is not None:
        def_conf['log']['exp_name'] = args.name

    arg_dic = parse_args_as_dict(parser)
    def_conf.update(arg_dic)
    main(def_conf)

