import librosa
import numpy as np
import matplotlib.pyplot as plt


def global_threshold_partitioning(spec, threshold):
    """
    基于全局阈值的自适应分块
    :param spec: 语谱图 (2D numpy array)
    :param threshold: 能量阈值
    :return: 分块列表 [(start_row, end_row, start_col, end_col), ...]
    """
    blocks = []
    rows, cols = spec.shape
    mask = spec > threshold

    def find_next_block(start_row, start_col):
        if mask[start_row, start_col] == 0:
            return None
        end_row, end_col = start_row, start_col
        while end_row < rows and mask[end_row, start_col]:
            end_row += 1
        while end_col < cols and mask[start_row, end_col]:
            end_col += 1
        return (start_row, end_row, start_col, end_col)

    for i in range(rows):
        for j in range(cols):
            block = find_next_block(i, j)
            if block:
                blocks.append(block)
                for r in range(block[0], block[1]):
                    for c in range(block[2], block[3]):
                        mask[r, c] = 0

    return blocks


# 加载音频文件并生成语谱图
y, sr = librosa.load('/root/data1/LSR/LSR2/lrs2_rebuild/audio/wav16k/min/tr/mix/5535415699068794046_00004_2.4022_6160231649911868423_00015_-2.4022.wav', sr=None)
D = np.abs(librosa.stft(y))
S = librosa.amplitude_to_db(D, ref=np.max)

# 设定全局能量阈值
global_threshold = -40  # 以dB为单位

# 执行基于全局阈值的自适应分块
blocks = global_threshold_partitioning(S, global_threshold)

# 可视化语谱图及其分块
plt.figure(figsize=(10, 6))
librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='log', cmap='magma')
for block in blocks:
    start_row, end_row, start_col, end_col = block
    plt.gca().add_patch(plt.Rectangle((start_col, start_row), end_col - start_col, end_row - start_row,
                                      edgecolor='cyan', facecolor='none', lw=2))
plt.colorbar(format='%+2.0f dB')
plt.title('Global Threshold Partitioning of Spectrogram')
plt.show()