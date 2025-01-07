###
# Author: Kai Li
# Date: 2021-06-20 00:21:33
# LastEditors: Kai Li
# LastEditTime: 2021-09-09 23:12:28
###
# import comet_ml
import os
import argparse
import json
import random

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from nichang.datas import AVSpeechDataset
from nichang.system.optimizers import make_optimizer
from nichang.system.core import System
# from nichang.system.tensorboard import TensorBoardLogger
from nichang.losses import PITLossWrapper, pairwise_neg_sisdr, pairwise_neg_snr
from nichang.models.ctcnet import CTCNet
from nichang.videomodels.video_process import VideoModel
from nichang.models.dual_path import Cross_Sepformer_warpper
from pytorch_lightning.loggers import TensorBoardLogger
from nichang.models.sepformer import Cross_Sepformer_warpper_1
from nichang.models.sand import Sandglasset
from nichang.models.av_conv import AV_model
from nichang.models.dual_rnn import Dual_RNN_model
from nichang.models.attention import Sepformer
from nichang.models.tf_gridnet import TFGridNet
from nichang.models.grid_TFnet import TFGridNetV2
from nichang.datas.Dataset import Datasets
from torch.utils.data import DataLoader as Loader
# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]

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
    help="Experiment name")
parser.add_argument(
    "--gpus", type=int, default=8,
    help="#gpus of each node")
parser.add_argument(
    "--nodes", type=int, default=1,
    help="#node")


def make_dataloader(conf):
    train_dataset = Datasets(
        conf['data']['train_dir'],
        [conf['data']['train_dir_1'],
         conf['data']['train_dir_2']],
        [conf['data']['train_dir_v1'],
         conf['data']['train_dir_v2']]

    )

    train_dataloader = Loader(train_dataset,
                              batch_size=conf['training']['batch_size'],
                              num_workers=conf['training']['num_workers'],
                              shuffle=True,
                              pin_memory=True, drop_last=True)

    val_dataset = Datasets(
        conf['data']['valid_dir'],
        [conf['data']['valid_dir_1'],
         conf['data']['valid_dir_2']],
        [conf['data']['valid_dir_v1'],
         conf['data']['valid_dir_v2']]
    )

    val_dataloader = Loader(val_dataset,
                            batch_size=conf['training']['batch_size'],
                            num_workers=conf['training']['num_workers'],
                            shuffle=True, pin_memory=True, drop_last=True)   # 验证集同样需要打乱顺序，因为数据集前半部分是匹配的，后半部分是不匹配的

    # print("val_dataset:\n", val_dataset)
    # print("val_dataloader:\n", val_dataloader)
    return train_dataloader, val_dataloader

def main(conf):
    train_loader, val_loader = make_dataloader(conf)
    # Define model and optimizer
    sample_rate = conf["data"]["sample_rate"]
    videomodel = None
    # audiomodel = CTCNet(sample_rate=sample_rate, **conf["audionet"])
    # audiomodel = Cross_Sepformer_warpper_1(**conf["dual_path"])
    # audiomodel = Sandglasset(**conf["sandglasset"])
    # audiomodel = AV_model(**conf["AV_model"])
    # audiomodel = Dual_RNN_model(**conf["Dual_Path_RNN"])
    audiomodel = TFGridNetV2(**conf["TFGridNet"])
    # audiomodel = Sepformer(**conf["sepformer"])
    optimizer = make_optimizer(audiomodel.parameters(), **conf["optim"])
    # Define scheduler
    scheduler = None
    if conf["training"]["half_lr"]:
        scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=10)

    # Just after instantiating, save the args. Easy loading in the future.
    conf["main_args"]["exp_dir"] = os.path.join("exp", conf["log"]["exp_name"])
    exp_dir = conf["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    # Define Loss function.
    loss_func = {
        # "train": PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx"),
        "train": PITLossWrapper(pairwise_neg_snr, pit_from="pw_mtx"),
        "val": PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx"),
        }
    system = System(
        audio_model=audiomodel,
        video_model=videomodel,
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
    callbacks.append(checkpoint)
    if conf["training"]["early_stop"]:
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=15, verbose=True))

    distributed_backend = "ddp"
    print(os.environ)
    print(torch.cuda.device_count())

    # default logger used by trainer
    os.makedirs(conf["log"]["path"], exist_ok=True)
    comet_logger = TensorBoardLogger(save_dir="/root/data1/LZR/CTCNet-main/log/tmp", name="experiment_name")

    trainer = pl.Trainer(
        max_epochs=conf["training"]["epochs"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        gpus=[0],
        num_nodes=conf["main_args"]["nodes"],
        # distributed_backend=distributed_backend,
        limit_train_batches=1.0,  # Useful for fast experiment
        gradient_clip_val=5.0,
        logger=comet_logger,
        sync_batchnorm=True
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
    import yaml
    from pprint import pprint
    from nichang.utils.parser_utils import prepare_parser_from_dict, parse_args_as_dict

    args = parser.parse_args()

    with open(args.conf_dir) as f:
        def_conf = yaml.safe_load(f)
    if args.name is not None:
        def_conf['log']['exp_name'] = args.name

    arg_dic = parse_args_as_dict(parser)
    def_conf.update(arg_dic)
    main(def_conf)