import os
import json

# Define input and output paths
cv_mix_scp = '/root/data2/vocveleb_li/tt/mix.json'
cv_s1_scp = '/root/data2/vocveleb_li/tt/s1.json'
cv_s2_scp = '/root/data2/vocveleb_li/tt/s2.json'

cv_mix = '/root/data2/vocveleb_li/vox2/audio/wav16k/min/tt/mix/'
cv_s1 = '/root/data2/vocveleb_li/vox2/audio/wav16k/min/tt/s1'
cv_s2 = '/root/data2/vocveleb_li/vox2/audio/wav16k/min/tt/s2'

# Initialize dictionaries to store paths for JSON output
cv_mix_data = []
cv_s1_data = []
cv_s2_data = []

# Process mixed audio files
for root, dirs, files in os.walk(cv_mix):
    files.sort()
    for file in files:
        path = os.path.join(root, file)
        cv_mix_data.append([path, 32000])

# Process s1 audio files
for root, dirs, files in os.walk(cv_s1):
    files.sort()
    for file in files:
        path = os.path.join(root, file)
        file =  file.split('.wav')[0]
        parts = file.split('_')
        # 遍历每个子字符串
        for i, part in enumerate(parts):
            try:
                # 尝试将当前子字符串转换为浮点数
                num1 = float(part)
            except ValueError:
                # 如果转换失败，跳过当前子字符串
                continue

            # 遍历当前数值之后的子字符串，寻找相反数
            for j in range(i + 1, len(parts)):
                try:
                    num2 = float(parts[j])
                    if num2 == -num1:
                        # 找到相反数，提取两者之间的子字符串
                        between = parts[0 :i]
                        s1 = '_'.join(between)
                except ValueError:
                    continue
        mouth_path = os.path.join('/root/data2/vocveleb_li/vox2/mouths', s1 + '.npz')
        cv_s1_data.append([path, mouth_path, 32000])

# Process s2 audio files
for root, dirs, files in os.walk(cv_s2):
    files.sort()
    for file in files:
        path = os.path.join(root, file)
        file = file.split('.wav')[0]
        parts = file.split('_')
        for i, part in enumerate(parts):
            try:
                # 尝试将当前子字符串转换为浮点数
                num1 = float(part)
            except ValueError:
                # 如果转换失败，跳过当前子字符串
                continue

            # 遍历当前数值之后的子字符串，寻找相反数
            for j in range(i + 1, len(parts)):
                try:
                    num2 = float(parts[j])
                    if num2 == -num1:
                        # 找到相反数，提取两者之间的子字符串
                        between = parts[i + 1:j]
                        s2 = '_'.join(between)
                except ValueError:
                    continue

        mouth_path = os.path.join('/root/data2/vocveleb_li/vox2/mouths', s2 + '.npz')
        cv_s2_data.append([path, mouth_path, 32000])

# Write the data to JSON files
with open(cv_mix_scp, 'w') as f:
    json.dump(cv_mix_data, f, indent=4)

with open(cv_s1_scp, 'w') as f:
    json.dump(cv_s1_data, f, indent=4)

with open(cv_s2_scp, 'w') as f:
    json.dump(cv_s2_data, f, indent=4)

print("JSON files created successfully.")