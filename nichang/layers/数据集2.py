import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

data_path = "/root/data1/shuxue/day_temp/1979_avg/19790101_avg.tif/"
ds = xr.open_dataset(data_path,engine='netcdf4')
plt.rc('font',family='WenQuanYi Micro Hei')
my_longitude = ds["longitude"].to_numpy()
my_latitude = ds["latitude"].to_numpy()
my_time = ds["time"].to_numpy()
my_pre = ds["pre"].to_numpy()

# my_pre = np.clip(my_pre, 0, None)
start = my_pre.shape[0] - 365*20
# font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
sum1 = my_pre[start:start+365*10].mean(axis=0)

for i in range(sum1.shape[0]):
    for j in range(sum1.shape[1]):
        if sum1[i,j] < 7.5 and sum1[i,j] >= 0:
            sum1[i,j] = 0
        if sum1[i,j] > 7.5:
            sum1[i, j] = 1
        if sum1[i,j] < 0:
            sum1[i, j]=-1
max_values = np.max(sum1, axis=1)
min_values = np.min(sum1, axis=1)
print(max_values)
print(min_values)

# sum2 = my_pre[start+365*10:start+365*20].sum(axis=0)

plt.imshow(sum1, cmap='hot', interpolation='nearest', origin='lower')
plt.colorbar()  
plt.ylabel('纬度')


save_path = r'/root/data1/shuxue/kk3333.png'
plt.savefig(save_path)
plt.show()
