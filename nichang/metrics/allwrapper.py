###
# Author: Kai Li
# Date: 2021-06-22 12:41:36
# LastEditors: Please set LastEditors
# LastEditTime: 2021-11-05 18:12:18
###
import csv
import torch
import numpy as np
import logging
from pesq import pesq
from pystoi import stoi

from ..losses import PITLossWrapper, pairwise_neg_sisdr, singlesrc_neg_snr, singlesrc_neg_sisdr,SI_SNR
import mir_eval.separation
logger = logging.getLogger(__name__)
from scipy import signal

def calculate_snr(clean_signal, noisy_signal):
    """
    计算信号的 SNR（信噪比）

    参数:
    clean_signal (numpy array): 原始干净的信号
    noisy_signal (numpy array): 含噪声的信号

    返回:
    snr (float): 以分贝 (dB) 为单位的信噪比
    """
    # 计算噪声信号
    noise = noisy_signal - clean_signal

    # 计算信号的功率
    signal_power = np.mean(clean_signal ** 2)

    # 计算噪声的功率
    noise_power = np.mean(noise ** 2)

    # 计算 SNR
    snr = 10 * np.log10(signal_power / noise_power)

    return snr

class ALLMetricsTracker:
    def __init__(self, save_file: str = ""):
        self.all_sdrs = []
        self.all_sdrs_i = []
        self.all_sisnrs = []
        self.all_sisnrs_i = []
        self.all_pesqs = []
        self.all_stois = []

        csv_columns = ["snt_id", "sdr", "sdr_i", "si-snr", "si-snr_i", "pesq", "stoi"]
        self.results_csv = open(save_file, "w")
        self.writer = csv.DictWriter(self.results_csv, fieldnames=csv_columns)
        self.writer.writeheader()
        self.pit_snr = singlesrc_neg_snr
        self.pit_sisnr = singlesrc_neg_sisdr
        self.sisnr = SI_SNR()



    def __call__(self, mix, clean, estimate, key):
        # sisnr
        sisnr = self.sisnr(clean.unsqueeze(0),estimate.unsqueeze(0))
        mix = torch.stack([mix] * clean.shape[0], dim=0)
        sisnr_baseline = self.sisnr(clean.unsqueeze(0),mix.unsqueeze(0))
        sisnr_i = sisnr - sisnr_baseline

        (sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(clean.cpu().numpy(), estimate.cpu().numpy())
        (sdr_base, sir, sar, perm) = mir_eval.separation.bss_eval_sources(clean.cpu().numpy(), mix.cpu().numpy())
        sdr = sdr.mean()
        sdr_i = sdr - sdr_base.mean()


        # # sdr
        # sdr = self.pit_snr(
        #     estimate.unsqueeze(0),
        #     clean.unsqueeze(0),
        # )
        # sdr_baseline = self.pit_snr(
        #     mix.unsqueeze(0),
        #     clean.unsqueeze(0),
        # )
        # sdr_i = sdr - sdr_baseline

        # stoi pesq
        # PESQ
        _pesq = pesq(deg= estimate[0].squeeze(0).cpu().numpy(),ref= clean[0].squeeze(0).cpu().numpy(), fs=16000)
        _pesq_1 = pesq(deg=estimate[1].squeeze(0).cpu().numpy(), ref=clean[1].squeeze(0).cpu().numpy(), fs=16000)
        _pesq =  (_pesq +  _pesq_1)/2

        # STOI
        _stoi = stoi(clean[0].squeeze(0).cpu().numpy(), estimate[0].squeeze(0).cpu().numpy(), 16000, extended=False)
        _stoi_1 = stoi(clean[1].squeeze(0).cpu().numpy(), estimate[1].squeeze(0).cpu().numpy(), 16000, extended=False)

        row = {
            "snt_id": key,
            "sdr": sdr.item(),
            "sdr_i": sdr_i.item(),
            "si-snr": -sisnr.item(),
            "si-snr_i": -sisnr_i.item(),
            "pesq": _pesq,
            "stoi": _stoi,
        }
        self.key = key
        self.writer.writerow(row)
        # Metric Accumulation
        self.all_sdrs.append(-sdr.item())
        self.all_sdrs_i.append(-sdr_i.item())
        self.all_sisnrs.append(-sisnr.item())
        self.all_sisnrs_i.append(-sisnr_i.item())
        self.all_pesqs.append(_pesq)
        self.all_stois.append(_stoi)

    def get_mean(self):
        return {
            "sdr": np.mean(self.all_sdrs),
            "sdr_i": np.mean(self.all_sdrs_i),
            "si-snr": np.mean(self.all_sisnrs),
            "si-snr_i": np.mean(self.all_sisnrs_i),
            "pesq": np.mean(self.all_pesqs),
            "stoi": np.mean(self.all_stois)
        }

    def get_std(self):
        return {
            "sdr": np.std(self.all_sdrs),
            "sdr_i": np.std(self.all_sdrs_i),
            "si-snr": np.std(self.all_sisnrs),
            "si-snr_i": np.std(self.all_sisnrs_i),
            "pesq": np.std(self.all_pesqs),
            "stoi": np.std(self.all_stois)
        }

    def final(
        self,
    ):
        row = {
            "snt_id": "avg",
            "sdr": np.array(self.all_sdrs).mean(),
            "sdr_i": np.array(self.all_sdrs_i).mean(),
            "si-snr": np.array(self.all_sisnrs).mean(),
            "si-snr_i": np.array(self.all_sisnrs_i).mean(),
            "pesq": np.array(self.all_pesqs).mean(),
            "stoi": np.array(self.all_stois).mean()
        }
        self.writer.writerow(row)
        row = {
            "snt_id": "std",
            "sdr": np.array(self.all_sdrs).std(),
            "sdr_i": np.array(self.all_sdrs_i).std(),
            "si-snr": np.array(self.all_sisnrs).std(),
            "si-snr_i": np.array(self.all_sisnrs_i).std(),
            "pesq": np.array(self.all_pesqs).std(),
            "stoi": np.array(self.all_stois).std()
        }
        self.writer.writerow(row)
        self.results_csv.close()
    
