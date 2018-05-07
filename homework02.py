import math
import numpy as np
from matplotlib import pyplot as plt


def normalize_array(array, percent):
    arr = array
    arr.sort()
    count = math.trunc(len(array) * percent / 100)
    arr = arr[count:len(array) - count]
    result = np.array(object=[], dtype=float)
    x_mean = sum(arr) / len(arr)
    s_dev = (sum([(x - x_mean) ** 2 for x in arr]) / len(arr)) ** 0.5
    for i in range(len(array)):
        result = np.append(arr=result, values=[(array[i] - x_mean) / s_dev])
    return result

def normalize_array_min_max(array, percent):
    arr = array
    arr.sort()
    count = math.trunc(len(array) * percent / 100)
    arr = arr[count:len(array) - count]
    result = np.array(object=[], dtype=float)
    x_min = min(arr)
    x_max = max(arr)
    for i in range(len(array)):
        result = np.append(arr=result, values=[(array[i] - x_min) / (x_max - x_min)])
    return result


data = np.random.normal(size=100) * 10
print('non-normalized array = ' + str(data))
print()

print('Standardization')
print()
data_np = (data - data.mean()) / data.std()
print('numpy = ' + str(data_np))
print()
data_usr = normalize_array(data, 2.5)
print('Python user function = ' + str(data_usr))
print()

print('Min-Max scaling')
print()
data_np_mm = (data - data.min()) / (data.max() - data.min())
print('numpy = ' + str(data_np_mm))
print()
data_usr_mm = normalize_array_min_max(data, 2.5)
print('Python user function = ' + str(data_usr_mm))
print()

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))

y = np.zeros(shape=len(data), dtype=int)

ax1.scatter(data_usr, y, color='g')
ax1.set_title('Python standardization', color='g')

ax2.scatter(data_usr_mm, y, color='g')
ax2.set_title('Python Min-Max scaling', color='g')

ax3.scatter(data_np, y, color='b')
ax3.set_title('NumPy standardization', color='b')

ax4.scatter(data_np_mm, y, color='b')
ax4.set_title('NumPy Min-Max scaling', color='b')

plt.tight_layout()
for ax in (ax1, ax2, ax3, ax4):
    ax.get_yaxis().set_visible(False)
    ax.grid()
plt.show()
