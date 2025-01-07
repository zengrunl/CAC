import numpy as np
import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('/root/data1/LSR/mvlrs_v1/main/6001854301505575951/00005.mp4')


# ShiTomasi 角点检测参数
# feature_params = dict( maxCorners = 100,
#                        qualityLevel = 0.3,
#                        minDistance = 7,
#                        blockSize = 7 )
#
# # lucas kanade光流法参数
# lk_params = dict( winSize  = (15,15),
#                   maxLevel = 2,
#                   criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
#
# # 创建随机颜色
# color = np.random.randint(0,255,(100,3))
#
# # 获取第一帧，找到角点
# ret, old_frame = cap.read()
# #找到原始灰度图
# old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
#
# #获取图像中的角点，返回到p0中
# p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# 创建一个蒙版用来画轨迹
# mask = np.zeros_like(old_frame)
landmarks_data = np.load("/root/data1/LSR/LSR2/lrs2_rebuild/mouths/5892689547109465164_00013.npz")['data']
B, C, T = landmarks_data.shape  # 2,2,50,88,88
flow = np.zeros((B- 1, C, T , 2))  # 2,2,49,88,88,2
# 计算光流
a=flow[0]
for i in range(B-1):
            # a = mouth[i, j, t]
            # print("mouth[i, j, t] shape: ", mouth[i, j, t].shape)
            # print("mouth[i, j, t] dtype: ", mouth[i, j, t].dtype)
            # 计算帧 t 和帧 t+1 之间的光流
    # prev_frame = cv2.cvtColor(flow[i], cv2.COLOR_BGR2GRAY)
    # next_frame = cv2.cvtColor(flow[i+1], cv2.COLOR_BGR2GRAY)
    flow[i] = cv2.calcOpticalFlowFarneback(landmarks_data[i], landmarks_data[i+1], None, 0.5, 3, 15, 3, 5, 1.2, 0)



flow_to_display = flow[0, ...]

# 光流的两个通道分别代表水平和竖直方向的光流
flow_horizontal = flow_to_display[..., 0]
flow_vertical = flow_to_display[..., 1]
# 前面准备hsv图像的代码应该是正确的，无需修改
hsv = np.zeros((flow_horizontal.shape[0], flow_horizontal.shape[1], 3), dtype=np.uint8)
magnitude, angle = cv2.cartToPolar(flow_horizontal, flow_vertical)

# 将角度转换为用于HSV颜色空间的0-180度
hsv[..., 0] = angle * (180 / np.pi / 2)

# 将速度标准化到0-255
hsv[..., 1] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

# # 设置亮度值为255
hsv[..., 2] = 255

# 将HSV转换为RGB
rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

plt.imshow(rgb)
plt.title('Optical Flow Visualization')
plt.show()
# 创建光流可视化图，这里简单地使用水平和竖直方向光流的大小
# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# axs[0].imshow(flow_horizontal, cmap='hot')
# axs[0].set_title('Horizontal Flow')
#
# axs[1].imshow(flow_vertical, cmap='hot')
# axs[1].set_title('Vertical Flow')
#
# plt.suptitle('Optical Flow Visualization')
# plt.savefig("optical_flow.png") # 保存光流图像
# plt.show() # 显示光流图像
# while(1):
    # ret,frame = cap.read()
    # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # # 选取好的跟踪点
    # good_new = p1[st==1]
    # good_old = p0[st==1]
    # print(p1.shape)
    # # 画出轨迹
    # for i,(new,old) in enumerate(zip(good_new,good_old)):
    #     a,b = new.ravel()
    #     c,d = old.ravel()
    #     mask = cv2.line(mask, (int(a),int(b)),(int(c),int(d)), color[i].tolist(), 2)
    #     frame = cv2.circle(frame,(int(a),int(b)),5,color[i].tolist(),-1)
    # img = cv2.add(frame,mask)
    #
    # cv2.imwrite('/root/data1/save/optical_flow/frame.png',img)
    # k = cv2.waitKey(30) & 0xff
    # if k == 27:
    #     break
    #
    # # 更新上一帧的图像和追踪点
    # old_gray = frame_gray.copy()
    # p0 = good_new.reshape(-1,1,2)

# cv2.destroyAllWindows()
cap.release()



# """

# Python implementation of technology discussed in 'Dynamic Image Networks for Action Recognition' by Bilen et al.
# Their paper and GitHub can be found here: https://github.com/hbilen/dynamic-image-nets
# """获取动态图像，
#
#
# def get_dynamic_image(frames, normalized=True):
#     """ Takes a list of frames and returns either a raw or normalized dynamic image."""
#     num_channels = frames[0].shape[2]
#     channel_frames = _get_channel_frames(frames, num_channels)
#     channel_dynamic_images = [_compute_dynamic_image(channel) for channel in channel_frames]
#
#     dynamic_image = cv2.merge(tuple(channel_dynamic_images))
#     if normalized:
#         dynamic_image = cv2.normalize(dynamic_image, None, 0, 255, norm_type=cv2.NORM_MINMAX)
#         dynamic_image = dynamic_image.astype('uint8')
#
#     return dynamic_image
#
#
# def _get_channel_frames(iter_frames, num_channels):
#     """ Takes a list of frames and returns a list of frame lists split by channel. """
#     frames = [[] for channel in range(num_channels)]
#
#     for frame in iter_frames:
#         for channel_frames, channel in zip(frames, cv2.split(frame)):
#             channel_frames.append(channel.reshape((*channel.shape[0:2], 1)))
#     for i in range(len(frames)):
#         frames[i] = np.array(frames[i])
#     return frames
#
# def _compute_dynamic_image(frames):
#     """ Adapted from https://github.com/hbilen/dynamic-image-nets """
#     num_frames, h, w, depth = frames.shape
#
#     # Compute the coefficients for the frames.
#     coefficients = np.zeros(num_frames)
#     for n in range(num_frames):
#         cumulative_indices = np.array(range(n, num_frames)) + 1
#         coefficients[n] = np.sum(((2*cumulative_indices) - num_frames) / cumulative_indices)
#
#     # Multiply by the frames by the coefficients and sum the result.
#     x1 = np.expand_dims(frames, axis=0)
#     x2 = np.reshape(coefficients, (num_frames, 1, 1, 1))
#     result = x1 * x2
#     return np.sum(result[0], axis=0).squeeze()