import librosa
import soundfile as sf
import os

import os
import random
import numpy as np

import os
import soundfile as sf
import numpy as np
from tqdm import tqdm
import random  # Only needed if assigning random SNRs


# from your_module import activlev  # Uncomment and modify as per your actual import

def activlev(signal, fs, mode):
    """
    Placeholder for the activlev function.
    Replace this with the actual implementation.
    """
    # Example implementation (replace with actual)
    return signal, np.mean(np.abs(signal))


# Configuration
s1_dir = '/root/data2/LRS3/LRS3_newcut/s1/'  # Path to s1 directory
s2_dir = '/root/data2/LRS3/LRS3_newcut/s2/'  # Path to s2 directory
# outS1 = 'path/to/output/s1'  # Output directory for s1
# outS2 = 'path/to/output/s2'  # Output directory for s2
outMix = '/root/data2/LRS3/LRS3_newcut/mix'  # Output directory for mixed audio
useActive = True  # Whether to apply activlev

# # Create output directories if they don't exist
# os.makedirs(outS1, exist_ok=True)
# os.makedirs(outS2, exist_ok=True)
os.makedirs(outMix, exist_ok=True)

# Get list of files in s1 and s2
s1_files = set(os.listdir(s1_dir))
s2_files = set(os.listdir(s2_dir))

# Find common files
common_files = s1_files.intersection(s2_files)

if not common_files:
    print("No common files found in the provided directories.")
    exit(1)

# Open output text files if needed
# Example: f1, f2, f3 = open('s1_tr.txt', 'w'), open('s2_tr.txt', 'w'), open('mix_tr.txt', 'w')
# Ensure to handle them appropriately or remove if not needed

for filename in tqdm(common_files, desc="Processing files"):
    s1_path = os.path.join(s1_dir, filename)
    s2_path = os.path.join(s2_dir, filename)

    # Extract SNRs
    # Here, we assign random SNRs for demonstration. Replace with actual logic.
    # s1Snr = round(random.uniform(0, 30), 4)  # Example SNR for s1
    # s2Snr = round(random.uniform(0, 30), 4)  # Example SNR for s2

    # Generate mix name
    base_name = os.path.splitext(filename)[0]
    s1WavName = f"s1_{base_name}"
    s2WavName = f"s2_{base_name}"
    mixName = f"{s1WavName}"
    line = base_name.split('_')

    s1Snr = round(float(line[2]), 4)
    s2Snr = round(float(line[-1]), 4)

    # Write to text files if needed
    # Example:
    # f1.write(s1_path + '\n')
    # f2.write(s2_path + '\n')
    # f3.write(mixName + '\n')

    # Read audio files
    s1_16k, fs1 = sf.read(s1_path)
    s2_16k, fs2 = sf.read(s2_path)

    # Ensure both files have the same sampling rate
    if fs1 != fs2:
        print(f"Sampling rates do not match for {filename}. Skipping.")
        continue
    fs = fs1  # Common sampling rate

    # Apply activlev if required
    if useActive:
        s1_16k, lev1 = activlev(s1_16k, fs, 'n')
        s2_16k, lev2 = activlev(s2_16k, fs, 'n')

    # Apply SNR weights
    weight_1 = pow(10, s1Snr / 20)
    weight_2 = pow(10, s2Snr / 20)

    s1_16k = weight_1 * s1_16k
    s2_16k = weight_2 * s2_16k

    # Align lengths
    mix_16k_length = min(len(s1_16k), len(s2_16k))
    s1_16k = s1_16k[:mix_16k_length]
    s2_16k = s2_16k[:mix_16k_length]

    # Mix signals
    mix_16k = s1_16k + s2_16k

    # Scaling to prevent clipping
    max_amp = max(np.max(np.abs(mix_16k)), np.max(np.abs(s1_16k)), np.max(np.abs(s2_16k)))
    if max_amp == 0:
        mix_scaling = 1  # Avoid division by zero
    else:
        mix_scaling = 0.9 / max_amp
    s1_16k *= mix_scaling
    s2_16k *= mix_scaling
    mix_16k *= mix_scaling

    # Define output paths
    # s1_out = os.path.join(outS1, f"{mixName}.wav")
    # s2_out = os.path.join(outS2, f"{mixName}.wav")
    mix_out = os.path.join(outMix, f"{mixName}.wav")

    # Write audio files
    # sf.write(s1_out, s1_16k, fs, format='WAV', subtype='PCM_16')
    # sf.write(s2_out, s2_16k, fs, format='WAV', subtype='PCM_16')
    sf.write(mix_out, mix_16k, fs, format='WAV', subtype='PCM_16')

# Close text files if they were opened
# Example:
# f1.close()
# f2.close()
# f3.close()

print("Processing and mixing completed successfully.")
# # Define paths to the two source directories and the mix output directory
# s1_dir = '/root/data2/LRS3/LRS3_newcut/s1/'  # Replace with the actual path to s1
# s2_dir = '/root/data2/LRS3/LRS3_newcut/s2/'  # Replace with the actual path to s2
# mix_files = '/root/data2/LRS3/LRS3_newcut/mix/'  # Replace with the actual path where tt.txt will be saved
#
# nums_file = 3000 # Replace with the desired number of mixed files
#
#
# def check_prefix(file1, file2):
#     """
#     Placeholder function to check if two files are from the same speaker.
#     Implement this function based on your specific prefix or naming convention.
#     """
#     # Example implementation based on filename prefixes
#     prefix1 = os.path.splitext(os.path.basename(file1))[0].split('_')[0]
#     prefix2 = os.path.splitext(os.path.basename(file2))[0].split('_')[0]
#     return prefix1 == prefix2
#
#
# # Retrieve list of .wav files from both directories
# s1_files = [f for f in os.listdir(s1_dir) if f.endswith('.wav')]
# s2_files = [f for f in os.listdir(s2_dir) if f.endswith('.wav')]
#
# # Find common filenames present in both directories
# common_files = list(set(s1_files).intersection(set(s2_files)))
#
# if not common_files:
#     raise ValueError("No common .wav files found in both s1 and s2 directories.")
#
# # Ensure that nums_file does not exceed the number of available unique pairs
# nums_file = min(nums_file, len(common_files))
#
# # Shuffle the list to ensure random selection
# random.shuffle(common_files)
#
# # Initialize lists to keep track of used files
# existed_list_tt = []
# res_tt_list = []
#
# tt_file = os.path.join(mix_files, 'tt.txt')
#
# with open(tt_file, "w") as ftt:
#     for i in range(nums_file):
#         filename = common_files[i]
#         s1_file = os.path.join(s1_dir, filename)
#         s2_file = os.path.join(s2_dir, filename)
#
#         # Check if this pair has already been used
#         if filename in existed_list_tt:
#             continue  # Skip if already used
#
#         # Check if both files are from the same speaker
#         # same_speaker = check_prefix(s1_file, s2_file)
#         # if same_speaker:
#         #     print(f"Skipping {filename} as both files are from the same speaker.")
#         #     continue  # Skip mixing if same speaker
#
#         # Append the pair to the results and mark as used
#         mix = [s1_file, s2_file]
#         res_tt_list.append(mix)
#         existed_list_tt.append(filename)
#
#         # Generate a random SNR value between 0 and 2.5
#         snr = np.random.uniform(-5, 5)
#
#         # Write the mixing information to tt.txt
#         line = f"{s1_file} {snr:.2f} {s2_file} {-snr:.2f}\n"
#         ftt.write(line)
#
# print(f"Mixing completed. {len(res_tt_list)} pairs have been written to {tt_file}.")
# # 定义参数
# folder_path = "/root/data2/LRS3/lrs3_rebuild/audio/wav16k/min/tt/s2/"  # 替换为你的音频文件夹路径
# output_folder = "/root/data2/LRS3/LRS3_newcut/s2/"    # 替换为输出文件夹路径
# target_samples = 32000  # 目标样本数
#
# # 确保输出文件夹存在
# os.makedirs(output_folder, exist_ok=True)
#
# # 支持的音频格式
# supported_formats = ('.wav', '.mp3', '.flac', '.ogg', '.aac', '.m4a')
#
# # 遍历文件夹中的所有文件
# for filename in os.listdir(folder_path):
#     if filename.lower().endswith(supported_formats):
#         file_path = os.path.join(folder_path, filename)
#         try:
#             # 加载音频
#             y, sr = librosa.load(file_path, sr=None)
#             # 裁剪音频
#             if len(y) > target_samples:
#                 y_trimmed = y[:target_samples]
#             else:
#                 y_trimmed = y
#             # 保存裁剪后的音频
#             output_path = os.path.join(output_folder, filename)
#             sf.write(output_path, y_trimmed, sr)
#             print(f"已处理并保存: {output_path}")
#         except Exception as e:
#             print(f"处理文件 {filename} 时出错: {e}")