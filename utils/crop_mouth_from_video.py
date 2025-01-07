#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under: https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/blob/master/LICENSE

# Ack: Code taken from Pingchuan Ma: https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks

""" Crop Mouth ROIs from videos for lipreading"""

import os
import cv2
import glob
import argparse
import numpy as np
from collections import deque

from utils import *
from transform import *
# lrw500_detected_face.csv
def load_args(default_config=None):
    parser = argparse.ArgumentParser(description='Lipreading Pre-processing')
    # -- utils
    parser.add_argument('--video-direc', default="/root/data1/LSR/LSR2/face_50frame/", help='raw video directory')
    parser.add_argument('--landmark-direc', default="/root/data1/LSR/LSR2/lrs2_rebuild/landmark/", help='landmark directory')
    parser.add_argument('--filename-path', default='/root/data1/LSR/LSR2/lrs2_rebuild/filename_input.csv', help='list of detected video and its subject ID')
    parser.add_argument('--save-direc', default="/root/data1/LSR/LSR2/mouth_50/", help='the directory of saving mouth ROIs')
    # -- mean face utils
    parser.add_argument('--mean-face', default='/root/data1/LZR/CTCNet-main/utils/20words_mean_face.npy', help='mean face pathname')
    # -- mouthROIs utils
    parser.add_argument('--crop-width', default=96, type=int, help='the width of mouth ROIs')
    parser.add_argument('--crop-height', default=96, type=int, help='the height of mouth ROIs')
    parser.add_argument('--start-idx', default=48, type=int, help='the start of landmark index')
    parser.add_argument('--stop-idx', default=68, type=int, help='the end of landmark index')
    parser.add_argument('--window-margin', default=12, type=int, help='window margin for smoothed_landmarks')
    # -- convert to gray scale
    parser.add_argument('--convert-gray', default=False, action='store_true', help='convert2grayscale')
    # -- test set only
    parser.add_argument('--testset-only', default=False, action='store_true', help='process testing set only')

    args = parser.parse_args()
    return args

args = load_args()

# -- mean face utils
STD_SIZE = (256, 256)
mean_face_landmarks = np.load(args.mean_face)
stablePntsIDs = [33, 36, 39, 42, 45]


def crop_patch( video_pathname, landmarks):

    """Crop mouth patch
    :param str video_pathname: pathname for the video_dieo
    :param list landmarks: interpolated landmarks
    """
    if isinstance(landmarks, list):
        stacked_landmarks = np.empty((len(landmarks), 68, 2))
        for i, landmark in enumerate(landmarks):
            # 检查当前landmark的形状
            if np.array(landmark).shape == (68, 2):
                stacked_landmarks[i] = landmark
            elif np.array(landmark).shape == (1, 68, 2):
                # 正常情况下直接赋值
                stacked_landmarks[i] = np.squeeze(landmark)
            elif np.array(landmark).shape != (1, 68, 2):
                # 错误情况，需要检测与前一帧或后一帧哪个更相似
                # prev_frame = np.squeeze(landmarks[i - 1]) if i > 0 else None
                # next_frame = np.squeeze(landmarks[i + 1]) if i < len(landmarks) - 1 else None
                # # 计算与前一帧和后一帧的差异
                # diff_with_prev = np.linalg.norm(prev_frame - landmark[0]) if prev_frame is not None else float('inf')
                # diff_with_next = np.linalg.norm(next_frame - landmark[1]) if next_frame is not None else float('inf')
                #
                # # 选择与之前/之后帧差异较小的landmark
                # if diff_with_prev < diff_with_next:
                stacked_landmarks[i] = landmark[0]
                # else:
                #     stacked_landmarks[i] = landmark[1]
        landmarks = stacked_landmarks

        # multi_sub_landmarks_array = np.array([np.array(sublist) for sublist in landmarks])



    frame_idx = 0
    frame_gen = read_video(video_pathname)
    while True:
        try:
            frame = frame_gen.__next__() ## -- BGR
        except StopIteration:
            break
        if frame_idx == 0:
            q_frame, q_landmarks = deque(), deque()
            sequence = []

        q_landmarks.append(landmarks[frame_idx])
        q_frame.append(frame)
        if len(q_frame) == args.window_margin:

            smoothed_landmarks = np.mean(q_landmarks, axis=0)
            cur_landmarks = q_landmarks.popleft()
            cur_frame = q_frame.popleft()
            # -- affine transformation
            trans_frame, trans = warp_img(smoothed_landmarks[stablePntsIDs, :], mean_face_landmarks[stablePntsIDs, :],cur_frame,STD_SIZE)
            trans_landmarks = trans(cur_landmarks)
            # -- crop mouth patch
            sequence.append( cut_patch( trans_frame,
                                        trans_landmarks[args.start_idx:args.stop_idx],
                                        args.crop_height//2,
                                        args.crop_width//2,))
        if frame_idx == len(landmarks)-1:
            #deal with corner case with video too short
            if len(landmarks) < args.window_margin:
                smoothed_landmarks = np.mean(q_landmarks, axis=0)
                cur_landmarks = q_landmarks.popleft()
                cur_frame = q_frame.popleft()

                # -- affine transformation
                trans_frame, trans = warp_img(smoothed_landmarks[stablePntsIDs, :],
                                            mean_face_landmarks[stablePntsIDs, :],
                                            cur_frame,
                                            STD_SIZE)
                trans_landmarks = trans(cur_landmarks)
                # -- crop mouth patch
                sequence.append(cut_patch( trans_frame,
                                trans_landmarks[args.start_idx:args.stop_idx],
                                args.crop_height//2,
                                args.crop_width//2,))

            while q_frame:
                cur_frame = q_frame.popleft()
                # -- transform frame
                trans_frame = apply_transform( trans, cur_frame, STD_SIZE)
                # -- transform landmarks
                trans_landmarks = trans(q_landmarks.popleft())
                # -- crop mouth patch
                sequence.append( cut_patch( trans_frame,
                                            trans_landmarks[args.start_idx:args.stop_idx],
                                            args.crop_height//2,
                                            args.crop_width//2,))
            return np.array(sequence)
        frame_idx += 1
    return None

def landmarks_interpolate(landmarks):
    
    """Interpolate landmarks
    param list landmarks: landmarks detected in raw videos
    """

    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    if not valid_frames_idx:
        return None
    for idx in range(1, len(valid_frames_idx)):
        if valid_frames_idx[idx] - valid_frames_idx[idx-1] == 1:
            continue
        else:
            landmarks = linear_interpolate(landmarks, valid_frames_idx[idx-1], valid_frames_idx[idx])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    # -- Corner case: keep frames at the beginning or at the end failed to be detected.
    if valid_frames_idx:
        landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
        landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    assert len(valid_frames_idx) == len(landmarks), "not every frame has landmark"
    return landmarks

videos_path = '/root/data1/LSR/LSR2/lrs2_rebuild/faces/'
# lines = open(args.filename_path).read().splitlines()
with open("/root/data1/LSR/LSR2/1.txt") as file:
    file_names = file.readlines()
file_paths = []
for file_name in file_names:
    file_name = file_name.strip()  # 去除换行符
    # file_name = file_name.split('*')[0]
    # file_name = file_name + '.mp4'
    file_path = os.path.join(videos_path, file_name)
    file_paths.append(file_path)


    # 使用os.listdir和os.path.splitext获取指定路径下的所有.mp4文件
video_files = [f for f in os.listdir(videos_path) if os.path.splitext(f)[1] == '.mp4']
video_files = [os.path.join(videos_path, f) for f in video_files]  # 获取完整路径
# for filename_idx, line in enumerate(lines):
#     x = line.split(',')
#
#     filename, person_id = x[3], x[2]

# lines = list(filter(lambda x: 'test' in x, lines)) if args.testset_only else lines


for video_file in video_files:
    video_name = os.path.basename(video_file)
    filename = os.path.splitext(video_name)[0]
    skip_file = False
    person_id = filename[0:19]
    # print('idx: {} \tProcessing.\t{}'.format(filename_idx, filename))

    video_pathname = os.path.join(args.video_direc, filename+'.mp4')
    landmarks_pathname = os.path.join(args.landmark_direc, filename+'.npz')
    dst_pathname = os.path.join( args.save_direc, filename+'.npz')

    # if os.path.exists(dst_pathname):
    #     continue

    multi_sub_landmarks = np.load(landmarks_pathname, allow_pickle=True)['data']
    landmarks = [None] * len(multi_sub_landmarks)
    smooth_landmarks= [None] * len(multi_sub_landmarks)
    num_consecutive_lists = 0
    for frame_idx in range(len(landmarks)):
        try:
            if isinstance(multi_sub_landmarks[frame_idx], list):
                landmarks[frame_idx] = multi_sub_landmarks[frame_idx]
                if num_consecutive_lists > 100:  # 如果10帧都包含两个列表
                    skip_file = True
                    break  # 跳出当前循环，读取下一个视频
                if len(multi_sub_landmarks[frame_idx]) != 1:
                    num_consecutive_lists += 1
            elif isinstance(multi_sub_landmarks[frame_idx], np.ndarray):
                if multi_sub_landmarks[frame_idx].shape != (1,68,2):
                    min_diff = float('inf')
                    best_landmark = None
                    # 迭代并处理每一个检测到得面部
                    for i in range(multi_sub_landmarks[frame_idx].shape[0]):
                        if frame_idx > 0:  # calculate difference with previous frame
                            diff_prev = np.abs(multi_sub_landmarks[frame_idx][i] - landmarks[-1]).sum()
                        else:
                            diff_prev = 0
                        if frame_idx < len(multi_sub_landmarks) - 1:  # calculate difference with next frame
                            diff_next = np.abs(
                                multi_sub_landmarks[frame_idx][i] - multi_sub_landmarks[frame_idx + 1]).sum()
                        else:
                            diff_next = 0
                        # 获取总差值
                        total_diff = diff_prev + diff_next
                        # 如果总差值小于当前最小差值，更新最小差式并选择当前landmark
                        if total_diff < min_diff:
                            min_diff = total_diff
                            best_landmark = multi_sub_landmarks[frame_idx][i]
                    # 在表中添加最好的landmark
                    landmarks[frame_idx] = best_landmark

                else:
                    landmarks[frame_idx] = np.squeeze(multi_sub_landmarks[frame_idx])
                # landmarks[frame_idx] = smooth_landmarks[frame_idx]  #original for LRW[int(person_id)]['facial_landmarks']

            # landmarks[frame_idx] = multi_sub_landmarks[frame_idx][int(person_id)]   #VOXCELEB2
        except (IndexError, TypeError):
            continue
    if skip_file:
        skip_file = False  # 重置标记
        # 跳过剩余操作
        continue
    # -- pre-process landmarks: interpolate frames not being detected.
    preprocessed_landmarks = landmarks_interpolate(landmarks)
    if not preprocessed_landmarks:
        continue

    # -- crop
    sequence = crop_patch(video_pathname, preprocessed_landmarks)
    assert sequence is not None, "cannot crop from {}.".format(filename)

    # -- save
    data = convert_bgr2gray(sequence) if args.convert_gray else sequence[...,::-1]
    save2npz(dst_pathname, data=data)

print('Done.')
