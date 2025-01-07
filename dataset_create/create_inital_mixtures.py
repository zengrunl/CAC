import os
import random
import numpy as np
import argparse
import soundfile as sf
import logging


def check_prefix(wav1, wav2):
    prefix1 = os.path.basename(wav1).split('.')[0][:3]
    prefix2 = os.path.basename(wav2).split('.')[0][:3]

    if prefix1 == prefix2:
        return True
    else:
        return False

def CreateFiles(input_dir, output_dir, nums_file, state):
    wavList = []
    mix_files = os.path.join(output_dir, 'mix_files')
    if not os.path.exists(mix_files):
        os.mkdir(mix_files)

    for root, root1, fi in os.walk(input_dir):
        for file_ in root1:
            input1 = os.path.join(input_dir, file_)
            for root, root2, fil in os.walk(input1):
                for file__ in root2:
                    input2 = os.path.join(input1, file__)
                    for root, _, files in os.walk(input2):
                        for file in files:
                            if state.upper() in root.upper() and file.endswith('WAV') or file.endswith('wav'):
                                wavFile = os.path.join(root, file)
                                # prefix = file.split('.')[0]# 获取文件的前缀（去掉扩展名）
                                # prefix_exists = False
                                # for existing_wav in wavList:
                                #     existing_prefix = os.path.basename(existing_wav).split('.')[0]
                                #     if existing_prefix[:3] == prefix[:3]:
                                #         prefix_exists = True
                                #         break
                                #
                                # if prefix_exists:
                                #     continue  # 前三个数相同，不进行混合
                                data, sr = sf.read(wavFile)
                                logger = logging.getLogger('name')
                                logger.info(file)
                                if len(data.shape) != 1:
                                    raise ValueError
                                if data.shape[0] < 32000:
                                    pass
                                else:
                                    wavList.append(wavFile)

    random.shuffle(wavList)

    if state.upper() == 'TRAIN':

        existed_list_tr = []
        existed_list_cv = []

        wav_list_tr = wavList[:len(wavList)-int(len(wavList)*0.1)]
        wav_list_cv = wavList[len(wavList)-int(len(wavList)*0.1):]

        tr_file = os.path.join(mix_files, 'tr.txt')
        cv_file = os.path.join(mix_files, 'cv.txt')
        res_tr_list = []
        res_cv_list = []

        with open(tr_file, 'w') as ftr:
            for i in range(nums_file):
                mix = random.sample(wav_list_tr, 2)
                back_mix = [mix[1], mix[0]]
                same_speaker = True

                while same_speaker:
                    mix = random.sample(wav_list_tr, 2)  # 随机选择两个语音片段进行混合



                    # 检测是否是相同的说话人
                    same_speaker = check_prefix(mix[0], mix[1])

                if mix not in existed_list_tr:
                    res_tr_list.append(mix)
                else:
                    while mix in existed_list_tr:
                        mix = random.sample(wav_list_tr, 2)

                res_tr_list.append(mix)
                existed_list_tr.append(mix)
                existed_list_tr.append(back_mix)
                snr = np.random.uniform(0, 2.5)
                line = "{} {} {} {}\n".format(mix[0], snr, mix[1], -snr)
                ftr.write(line)
        ftr.close()

        with open(cv_file, 'w') as fcv:
            for i in range(int(nums_file / 4)):
                mix = random.sample(wav_list_cv, 2)
                back_mix = [mix[1], mix[0]]
                same_speaker = True
                while same_speaker:
                    mix = random.sample(wav_list_cv, 2)  # 随机选择两个语音片段进行混合
                    # 检测是否是相同的说话人
                    same_speaker = check_prefix(mix[0], mix[1])
                if mix not in existed_list_cv:
                    res_cv_list.append(mix)
                else:
                    while mix in existed_list_cv:
                        mix = random.sample(wav_list_cv, 2)
                res_cv_list.append(mix)
                existed_list_cv.append(mix)
                existed_list_cv.append(back_mix)
                snr = np.random.uniform(0, 2.5)
                line = "{} {} {} {}\n".format(mix[0], snr, mix[1], -snr)
                fcv.write(line)
        fcv.close()

    elif state.upper() == 'TEST':
        existed_list_tt = []
        wav_list_tt = wavList
        tt_file = os.path.join(mix_files, 'tt.txt')
        res_tt_list = []
        with open(tt_file, "w") as ftt:
            for i in range(nums_file):
                mix = random.sample(wav_list_tt, 2)
                back_mix = [mix[1], mix[0]]
                same_speaker = True
                while same_speaker:
                    mix = random.sample(wav_list_tt, 2)  # 随机选择两个语音片段进行混合
                    # 检测是否是相同的说话人
                    same_speaker = check_prefix(mix[0], mix[1])
                if mix not in existed_list_tt:
                    res_tt_list.append(mix)
                else:
                    while mix in existed_list_tt:
                        mix = random.sample(wav_list_tt, 2)
                res_tt_list.append(mix)
                existed_list_tt.append(mix)
                existed_list_tt.append(back_mix)
                snr = np.random.uniform(-5, 5)
                line = "{} {} {} {}\n".format(mix[0], snr, mix[1], -snr)
                ftt.write(line)
        ftt.close()


def run(args):
    logging.basicConfig(level=logging.INFO)

    input_dir =args.input_dir
    output_dir = args.output_dir
    state = args.state
    nums_file = args.nums_files
    CreateFiles(input_dir, output_dir, nums_file, state)
    logging.info("Done create initial data pair")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Command to make separation dataset'
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/root/data2/voxceleb2/audio-test/aac/",
        help="Path to input data directory"
    )

    parser.add_argument(
        "--output_dir",
        default="/root/data1/voxceleb2/",
        type=str,
        help='Path ot output data directory'
    )
    parser.add_argument(
        "--nums_files",
        type=int,
        default=3000,
        help='Path ot output data directory'
    )
    parser.add_argument(
        "--state",
        type=str,
        default="TEST",
        help='Whether create train or test data directory'
    )
    args = parser.parse_args()
    run(args)
