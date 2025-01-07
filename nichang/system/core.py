###
# Author: Kai Li
# Date: 2021-06-19 11:43:37
# LastEditors: Kai Li
# LastEditTime: 2021-08-30 18:01:27
###

import torch
from pprint import pprint
import pytorch_lightning as pl
from mpmath import eps
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections.abc import MutableMapping
import warnings
import time
from memory_profiler import profile
warnings.filterwarnings("ignore")
import objgraph
import gc
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment
# from ..layers.STFT import STFTEncoder
from speechbrain.inference.speaker import EncoderClassifier
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
classifier.eval()
for param in classifier.parameters():
    param.requires_grad = False
def pit_cosine_loss(embeddings, x_vectors):
    """
    embeddings: (batch_size, num_speakers, embedding_dim)
    x_vectors: (batch_size, num_speakers, embedding_dim)
    """
    batch_size, num_speakers, embedding_dim = embeddings.size()
    loss = 0.0
    for i in range(batch_size):
        # 获取当前批次的嵌入和 x-vector
        emb = embeddings[i].detach().cpu().numpy()  # (num_speakers, embedding_dim)
        xv = x_vectors[i].detach().cpu().numpy()  # (num_speakers, embedding_dim)

        # 计算余弦相似度矩阵
        cos_sim_matrix = np.dot(emb, xv.T) / (
                np.linalg.norm(emb, axis=1, keepdims=True) * np.linalg.norm(xv, axis=1, keepdims=True).T + 1e-8
        )  # (num_speakers, num_speakers)

        # 使用匈牙利算法找到最佳匹配
        cost_matrix = -cos_sim_matrix  # 因为 linear_sum_assignment 是最小化
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # 计算匹配后的余弦相似度损失
        matched_cos_sim = cos_sim_matrix[row_ind, col_ind]
        loss += (1 - matched_cos_sim).sum()

    return loss / batch_size
def flatten_dict(d, parent_key="", sep="_"):
    """Flattens a dictionary into a single-level dictionary while preserving
    parent keys. Taken from
    `SO <https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys>`_

    Args:
        d (MutableMapping): Dictionary to be flattened.
        parent_key (str): String to use as a prefix to all subsequent keys.
        sep (str): String to use as a separator between two key levels.

    Returns:
        dict: Single-level dictionary, flattened.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def log_graph_growth(message=""):
    print(message)
    objgraph.show_growth()

def clear_memory(self):
    gc.collect()
    torch.cuda.empty_cache()


def log_object_growth(self, message=""):
    print(message)
    objgraph.show_growth()
    # 获取当前活动对象图
    gc.collect()
    objgraph.show_most_common_types()
# def log_graph_growth(message):
#     torch_gc()  # Force garbage collection to get accurate object counts
#     print(f"{message}")
#     objgraph.show_growth(limit=5)
#
# def log_object_growth(message):
#     torch_gc()  # Force garbage collection to get accurate object counts
#     print(f"{message}")
#     objgraph.show_most_common_types(objects=[torch.Tensor])

def torch_gc():
    # Force garbage collection to remove any unreferenced tensors
    gc.collect()
    torch.cuda.empty_cache()


class MI_Estimator(nn.Module):
    def __init__(self, input_dim):
        super(MI_Estimator, self).__init__()
        # 根据特征维度构建一个简单的MLP
        self.fc1 = nn.Linear(input_dim * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x, y):
        # x: [B, ...], y: [B, ...]
        # 将特征在最后一维concat，然后通过MLP输出一个标量分数
        xy = torch.cat([x, y], dim=-1)
        h = F.relu(self.fc1(xy))
        h = F.relu(self.fc2(h))
        out = self.fc3(h)
        return out  # [B, 1]


mine = MI_Estimator(64).cuda()
# def flatten_feat(feat):
#     # 简单平均池化，将C,T,F展平成D
#     # feat: [B, C, T, F]
#     B, C, T, F = feat.shape
#     return feat.view(B, -1)  # [B, C*T*F]

def compute_mi_loss(input_c, input_p, mine):
    # 确保input_c和input_p形状一致，并在最后一维concat前flatten
    c_feat =(input_c)  # [B, D]
    p_feat =(input_p)  # [B, D]
    B = c_feat.size(0)
    # 正样本对： (c_feat[i], p_feat[i])
    # 负样本对： (c_feat[i], p_feat[j]) j为打乱索引
    idx = torch.randperm(B)
    c_shuffle = c_feat[idx]  # 打乱顺序

    # 正样本得分
    T_xy = mine(c_feat, p_feat)  # [B,1]
    # 负样本得分
    T_x_y = mine(c_feat, c_shuffle)  # [B,1]

    # 计算MINE下界的估计
    # I(X;Y) >= E[T_xy] - log(E[exp(T_x_y)])
    # 我们将通过梯度下降最小化MI，所以loss = -(E[T_xy] - log(E[exp(T_x_y)]))
    # 这里使用log-sum-exp技巧
    E_T_xy = torch.mean(T_xy)
    E_exp_T_x_y = torch.mean(torch.exp(T_x_y))
    mi_lower_bound = E_T_xy - torch.log(E_exp_T_x_y)

    mi_loss = -mi_lower_bound

    return mi_loss
class System(pl.LightningModule):
    default_monitor: str = "val_loss"

    def __init__(
        self,
        audio_model=None,
        video_model=None,
        face_model=None,
        optimizer=None,
        loss_func=None,
        train_loader=None,
        val_loader=None,
        scheduler=None,
        config=None,
        train_video_model=False,
        train_loader1=None,
        val_loader1=None
    ):
        super().__init__()
        self.audio_model = audio_model
        self.video_model = video_model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_loader1 = train_loader1
        self.val_loader1 = val_loader1
        self.scheduler = scheduler
        self.config = {} if config is None else config
        self.train_video_model = train_video_model
        # Save lightning"s AttributeDict under self.hparams
        self.save_hyperparameters(self.config_to_hparams(self.config))
        self.face_model = face_model
        self.validation_step_outputs = []
        self.training_step_outputs = []
        self.initial_weights = torch.tensor([0.333, 0.333, 0.333])
        self.total_other_weight = 1.0
        # self.enc = STFTEncoder(
        #     128, 128, 64, window="hann", use_builtin_complex=False
        # )
        # self.audio_pool = nn.AdaptiveAvgPool2d((50, 1))


    # @profile(precision=5)
    def forward(self, wav, mouth, embedding):
        """Applies forward pass of the model.
        Returns:
            :class:`torch.Tensor`
        """
        if self.video_model == None:
            return self.audio_model(wav)
        else:
            # face_1,face_flow1 = self.face_model(faces[:, :1, :, :, :, :].squeeze(1))  # [1, 512, 1, 1]
            # face_2,face_flow2 = self.face_model(faces[:, 1:, :, :, :, :].squeeze(1))
            # log_graph_growth("1")
            if not self.train_video_model:
                with torch.no_grad():
                    mouth_emb,mouth_loss= self.video_model(mouth.type_as(wav))
            else:
                mouth_emb,mouth_loss = self.video_model(mouth.type_as(wav))
            # log_graph_growth("2")
            s = self.audio_model(wav, mouth_emb,embedding)
            # log_graph_growth("3")
            return s,mouth_loss


    # @profile(precision=5, stream=open('/root/data1/ceshi/memory_profiler_core.log', 'w+'))
    def common_step(self, batch, batch_nb, is_train=True):
        if self.video_model == None:
            if self.config["training"]["online_mix"] == True:
                inputs, targets, _ = self.online_mixing_collate(batch)
            else:
                inputs, targets = batch
            est_targets = self(inputs)
            if is_train:
                loss = self.loss_func["train"](est_targets, targets)
            else:
                loss = self.loss_func["val"](est_targets, targets)
            return loss
        elif self.video_model != None:
            # start_time = time.time()
            inputs, targets, target_mouths, _,audio_spec,embedding = batch #1, 2, 50, 3, 224, 224
            # ilens = torch.full((targets.shape[0],), 32000, dtype=torch.long)
            #             # mix_std_ = torch.std(targets, dim=(1, 2), keepdim=True)  # [B, 1, 1]
            #             # input = targets / mix_std_
            #             # batch = self.enc(input, ilens)[0]

            s,mouth_loss = self(inputs, target_mouths, embedding)
            est_targets,batch_com,batch_con= s
            # embeddings_0 = classifier.encode_batch(est_targets[:,:1,:].squeeze()).detach().cuda()  # 1,1,192
            # embeddings_1 = classifier.encode_batch(est_targets[:,1:,:].squeeze()).detach().cuda()    # 1,1,192
            # audio_spec = self.audio_pool(audio_spec)
            # est_targets  = self(inputs, target_mouths)
            # embeddings_normalized = F.normalize(batch_comp, p=2, dim=2)  # (batch_size, num_speakers, embedding_dim)
            # x_vectors_normalized = F.normalize(embedding, p=2, dim=2)  # (batch_size, num_speakers, embedding_dim)
            # mouth_embedding_normalized = F.normalize(mouth_embedding, p=2, dim=2)  # (batch_size, num_speakers, embedding_dim)
            # batch_normalized = F.normalize(batch_, p=2, dim=2)
            # embedding_t_normalized = F.normalize(embedding_t, p=2, dim=2)  # (batch_size, num_speakers, embedding_dim)
            # a=embeddings_normalized[:, :1, :]
            # b=embeddings_normalized[:, 1:, :]
            if targets.ndim == 2:
                targets = targets.unsqueeze(1)
            if is_train:
                cosine_similarity = self.loss_func["losscosine"](batch_com.flatten(2),batch_con.flatten(2))
                # cosine_similarity1 = self.loss_func["losscosine"](mouth_embedding_normalized, batch_normalized )

                # cosine_similarity2 =self.loss_func['tripletlosscosine'](batch_normalized[:,:1,:], x_vectors_normalized[:,:1,:], x_vectors_normalized[:,1:,:]) +\
                #                     self.loss_func['tripletlosscosine'](batch_normalized[:,1:,:], x_vectors_normalized[:,1:,:], x_vectors_normalized[:,:1,:])#pit_cosine_loss(embeddings_normalized,x_vectors_normalized)
                # cosine_similarity2 = self.loss_func['tripletlosscosine'](embeddings_0,
                #                                                          embedding[:, :1, :],
                #                                                          embedding[:, 1:, :]) + \
                #                      self.loss_func['tripletlosscosine'](embeddings_1,
                #                                                          embedding[:, 1:, :],
                #                                                          embedding[:, :1, :])
                                                                          # pit_cosine_loss(embeddings_normalized,x_vectors_normalized)

                loss_orthogonal =torch.mean(torch.abs(cosine_similarity))
                #     # torch.mean(torch.abs(cosine_similarity)) (F.mse_loss(batch_s1, audio_spec))
                # loss_reconstruction1 =0
                #     # (F.mse_loss(batch_s1, audio_spec)+ (F.mse_loss(batch_s2, audio_spec) )
                #     #                     +F.mse_loss(batch_s3, audio_spec)) #F.mse_loss(mouth_comp,mouth_loss) +
                #
                #
                # loss_reconstruction2 =cosine_similarity2
                #
                #     # (self.loss_func['tripletlosscosine'](batch_s2, audio_spec, batch_s1)
                #     #                     + self.loss_func['tripletlosscosine'](batch_s2, audio_spec, batch_s3))
                #
                #
                loss = self.loss_func["train"](est_targets, targets)
                loss1 = 0
                loss2 = loss_orthogonal
                term3 = 0

                # # 加权求和
                # loss1 = loss_orthogonal  # term1.sum() + term2.sum()
                # # loss1 = 0
                # loss2 = loss_reconstruction1
                # term3 = loss_reconstruction2

            else:
                cosine_similarity = self.loss_func["losscosine"](batch_com,batch_con)
                #
                # cosine_similarity2 = self.loss_func['tripletlosscosine'](batch_normalized[:,:1,:], x_vectors_normalized[:,:1,:], x_vectors_normalized[:,1:,:]) +\
                #                     self.loss_func['tripletlosscosine'](batch_normalized[:,1:,:], x_vectors_normalized[:,1:,:], x_vectors_normalized[:,:1,:])#pit_cosine_loss(embeddings_normalized,x_vectors_normalized)
                #
                loss_orthogonal = torch.mean(torch.abs(cosine_similarity))
                # loss_reconstruction1 = 0
                # loss_reconstruction2 = cosine_similarity2

                loss = self.loss_func["val"](est_targets, targets)
                # cosine_similarity2 = self.loss_func['tripletlosscosine'](embeddings_0,
                #                                                          embedding[:, :1, :],
                #                                                          embedding[:, 1:, :]) + \
                #                      self.loss_func['tripletlosscosine'](embeddings_1,
                #                                                          embedding[:, 1:, :],
                #                                                          embedding[:, :1, :])

                # loss1 = loss_orthogonal  # term1.sum() + term2.sum()
                #
                loss1 = 0
                loss2 = loss_orthogonal
                term3 = 0
                # loss2 = loss_reconstruction1
                # term3 = loss_reconstruction2

            return loss, loss1, loss2, term3


    # @profile(precision=5, stream=open('/root/data1/ceshi/memory_profiler_core.log', 'w+'))
    def training_step(self, batch, batch_nb):
        loss, loss1,loss2,term3 = self.common_step(batch, batch_nb)
        total_other_loss = loss1 + loss2 + term3

        self.training_step_outputs += [loss]
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_tripletloss", loss1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_orthogonalloss", loss2, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("ceshi", term3, on_epoch=True, prog_bar=True, sync_dist=True)
        loss_total = loss +loss2

        return {"loss": loss_total}

    # def on_train_epoch_end(self):
    #     avg_loss = torch.stack(self.training_step_outputs).mean()
    #     train_loss = torch.mean(self.all_gather(avg_loss))
    #     # import pdb; pdb.set_trace()
    #     self.logger.experiment.add_scalar("train_sisnr", -train_loss, self.current_epoch)
    # @profile(precision=5, stream=open('/root/data1/ceshi/memory_profiler_core.log', 'w+'))
    def validation_step(self, batch, batch_nb):
        loss, loss1, loss2,term3= self.common_step(batch, batch_nb, is_train=False)

         # loss1, loss2, term3 的权重和

        # 计算相对权重
        total_other_loss = loss1 + loss2 + term3

        self.validation_step_outputs += [loss]
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_tripletloss", loss1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_orthogonalloss", loss2, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_ceshi", term3, on_epoch=True, prog_bar=True, sync_dist=True)
        # log_graph_growth(f"After batch val")
        loss_total = loss
        return {"val_loss": loss_total}


    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_step_outputs).mean()
        val_loss = torch.mean(self.all_gather(avg_loss))
        self.log(
            "lr", self.optimizer.param_groups[0]["lr"], on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.logger.experiment.add_scalar(
            "learning_rate", self.optimizer.param_groups[0]["lr"], self.current_epoch
        )
        self.logger.experiment.add_scalar("val_sisnr", -val_loss, self.current_epoch)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        """Initialize optimizers, batch-wise and epoch-wise schedulers."""
        if self.scheduler is None:
            return self.optimizer

        # if not isinstance(self.scheduler, (list, tuple)):
        self.scheduler = [self.scheduler]  # support multiple schedulers
        epoch_schedulers = {"scheduler": self.scheduler, "monitor": "val_loss"}
        # for sched in self.scheduler:
        #     if not isinstance(sched, dict):
        #         if isinstance(sched, ReduceLROnPlateau):
        #             sched = {"scheduler": sched, "monitor": self.default_monitor}
        #         epoch_schedulers += [sched]
        #     else:
        #         sched.setdefault("monitor", self.default_monitor)
        #         sched.setdefault("frequency", 1)
        #         # Backward compat
        #         if sched["interval"] == "batch":
        #             sched["interval"] = "step"
        #         assert sched["interval"] in [
        #             "epoch",
        #             "step",
        #         ], "Scheduler interval should be either step or epoch"
        #         epoch_schedulers += [sched]
        return {
        "optimizer": self.optimizer,
        "lr_scheduler": {
            "scheduler": ReduceLROnPlateau(self.optimizer, mode="min", patience=2, verbose=True,factor=0.5),
            "monitor": "val_loss"
        },
    }


    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return self.val_loader

    def on_save_checkpoint(self, checkpoint):
        """Overwrite if you want to save more things in the checkpoint."""
        checkpoint["training_config"] = self.config
        return checkpoint
    @staticmethod
    def online_mixing_collate(batch):
        """Mix target sources to create new mixtures.
        Output of the default collate function is expected to return two objects:
        inputs and targets.
        """
        # Inputs (batch, time) / targets (batch, n_src, time)
        inputs, targets = batch
        batch, n_src, _ = targets.shape

        energies = torch.sum(targets ** 2, dim=-1, keepdim=True)
        new_src = []
        for i in range(targets.shape[1]):
            new_s = targets[torch.randperm(batch), i, :]
            new_s = new_s * torch.sqrt(energies[:, i] / (new_s ** 2).sum(-1, keepdims=True))
            new_src.append(new_s)

        targets = torch.stack(new_src, dim=1)
        inputs = targets.sum(1)
        return inputs, targets

    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.training_step_outputs).mean()
        train_loss = torch.mean(self.all_gather(avg_loss))
        self.logger.experiment.add_scalar("train_sisnr", -train_loss, self.current_epoch)
        if self.config["sche"]["patience"] > 0 and self.config["training"]["divide_lr_by"] != None:
            if (
                self.current_epoch % self.config["sche"]["patience"] == 0
                and self.current_epoch != 0
            ):
                new_lr = self.config["optim"]["lr"] / (
                    self.config["training"]["divide_lr_by"]
                    ** (self.current_epoch // self.config["sche"]["patience"])
                )
                # print("Reducing Learning rate to: {}".format(new_lr))
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = new_lr
        self.training_step_outputs.clear()

    @staticmethod
    def config_to_hparams(dic):
        """Sanitizes the config dict to be handled correctly by torch
        SummaryWriter. It flatten the config dict, converts ``None`` to
        ``"None"`` and any list and tuple into torch.Tensors.
        Args:
            dic (dict): Dictionary to be transformed.

        Returns:
            dict: Transformed dictionary.
        """
        dic = flatten_dict(dic)
        for k, v in dic.items():
            if v is None:
                dic[k] = str(v)
            elif isinstance(v, (list, tuple)):
                dic[k] = torch.tensor(v)
        return dic
