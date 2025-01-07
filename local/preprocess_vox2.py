###
# Author: Kai Li
# Date: 2021-06-19 23:35:14
# LastEditors: Kai Li
# LastEditTime: 2021-08-01 11:46:24
###

import argparse
import json
import os
import soundfile as sf
from tqdm import tqdm
import re

def get_mouth_path(in_mouth_dir, wav_file, out_filename, data_type):
#     wav_file = wav_file.split("_")
    p = re.compile(r'id\d{5}_.{11}_\d{5}')
    res = p.findall(wav_file)
    assert len(res) == 2, f"matching failded for case: {wav_file}"
    if out_filename == "s1":
        file_path = os.path.join(
            in_mouth_dir, "{}.npz".format(res[0])
        )
    else:
        file_path = os.path.join(
            in_mouth_dir, "{}.npz".format(res[1])
        )
#     file_path = os.path.join(
#         in_mouth_dir, "{}.npz".format(wav_file[:25])
#     )
    return file_path


def preprocess_one_dir(in_audio_dir, in_video_dir, out_dir, out_filename, data_type):
    """Create .json file for one condition."""
    file_infos = []
    in_dir = os.path.abspath(in_audio_dir)
    wav_list = os.listdir(in_dir)
    wav_list.sort()
    for wav_file in tqdm(wav_list):
        if not wav_file.endswith(".wav"):
            continue
        wav_path = os.path.join(in_audio_dir, wav_file)
        samples = sf.SoundFile(wav_path)
        file_A = wav_file[0:7] + '/' + wav_file[8:19] + '/' + wav_file[20:25]
        file_B = wav_file[25:32] + '/' + wav_file[33:44] + '/' + wav_file[45:50]
        lip_embedding_direc = '/root/data2/voxceleb2-mouth/seen-test/mouth_roi_hdf5/'
        visualFeaturesFile_B = lip_embedding_direc + file_B + '.h5'
        visualFeaturesFile_A = lip_embedding_direc + file_A + '.h5'
        audio_direc = "/root/data2/data/voxceleb2/audio/dev/aac/"
        audios_path_A = audio_direc + file_A + '.wav'
        audios_path_B = audio_direc + file_B + '.wav'
        # file_infos.append((wav_path, audios_path_A, visualFeaturesFile_A, len(samples)))
        # file_infos.append((wav_path, audios_path_B, visualFeaturesFile_B, len(samples)))
        if out_filename == "mix":
            file_infos.append((wav_path, len(samples)))
        elif out_filename == "s1":
            file_infos.append(
                (
                    audios_path_A,
                    visualFeaturesFile_A,
                    len(samples),
                )
            )
        elif out_filename == "s2":
            file_infos.append(
                (
                    audios_path_B,
                    visualFeaturesFile_B,
                    len(samples),
                )
            )
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, out_filename + ".json"), "w") as f:
        json.dump(file_infos, f, indent=4)


def preprocess(inp_args):
    """Create .json files for all conditions."""
    speaker_list = ["mix", "s1", "s2"]
    for data_type in [ "tt"]:
        for spk in speaker_list:
            preprocess_one_dir(
                os.path.join(inp_args.in_audio_dir),
                inp_args.in_mouth_dir,
                os.path.join(inp_args.out_dir, data_type),
                spk,
                data_type,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("WHAM data preprocessing")
    parser.add_argument(
        "--in_audio_dir",
        type=str,
        default="/root/data1/LSR/Voxceleb2/train/mix/",
        help="Directory path of audio including tr, cv and tt",
    )
    parser.add_argument(
        "--in_mouth_dir",
        type=str,
        default=None,
        help="Directory path of video including tr, cv and tt",
    )
    parser.add_argument(
        "--out_dir", type=str, default='/root/data1/LSR/Voxceleb2/', help="Directory path to put output files"
    )
    args = parser.parse_args()
    print(args)
    preprocess(args)
