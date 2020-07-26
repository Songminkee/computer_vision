import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
import cv2
import matplotlib.pyplot as plt

## 가중치 고려하지 않은 다운샘플링
def down_simple(img):
    out_h,out_w,c = img.shape
    out_h//=2
    out_w//=2
    down_img = np.zeros([out_h,out_w,c],dtype=np.uint8)
    for y in range(out_h):
        for x in range(out_w):
            down_img[y,x] = img[y*2,x*2]
    return down_img


img = cv2.cvtColor(cv2.imread('../data/lena.jpg'), cv2.COLOR_BGR2RGB)

# 다운 샘플링 횟수 지정 (16x16 까지만 다운샘플링)
min_axis = np.minimum(img.shape[0], img.shape[1])
if np.log2(min_axis) > 4:
    iteration = int(np.log2(min_axis) - 4)
else:
    iteration = 1

pyramid = []
pyramid.append(img)

# subplot 범위 지정
subplot_y = (iteration + 1) // 3  # subplot 지정, 원본도 plot 하기위해 +1, 한 행당 3개의 그림 plot.
if (iteration + 1) % 3 > 0:  # 3으로 나눠떨어지지 않으면 행 추가
    subplot_y += 1

sub_num = subplot_y * 100 + 31
fig = plt.figure(figsize=(13, 13))
plt.subplot(sub_num)
plt.imshow(pyramid[0])

for i in range(iteration):
    pyramid.append(down_simple(pyramid[i]))
    plt.subplot(sub_num + i + 1)
    plt.imshow(pyramid[i + 1])

fig.tight_layout()
plt.show()


## 버트&아델슨 다운샘플링
smooed_kernel = np.array([[0.0025,0.0125,0.02,0.0125,0.0025],
                          [0.0125,0.0625,0.1,0.0625,0.0125],
                          [0.02,0.1,0.16,0.1,0.02],
                          [0.0125,0.0625,0.1,0.0625,0.0125],
                          [0.0025,0.0125,0.02,0.0125,0.0025]])

# 가우시안 필터와 값 차이 확인하기
gaussian_filter_1d = cv2.getGaussianKernel(5,0)
gaussian_filter_2d = np.outer(gaussian_filter_1d, gaussian_filter_1d.transpose())
print(gaussian_filter_2d)


def smoothing_down(img,kernel):
    smooed = cv2.filter2D(img,-1,kernel)
    out_h, out_w, c = img.shape
    out_h //= 2
    out_w //= 2
    down_img = np.zeros([out_h, out_w, c], dtype=np.uint8)
    for y in range(out_h):
        for x in range(out_w):
            down_img[y, x] = smooed[y * 2, x * 2]
    return down_img

smooed_pyramid = []
smooed_pyramid.append(img)


fig = plt.figure(figsize=(13,13))
plt.subplot(sub_num)
plt.imshow(smooed_pyramid[0])

for i in range(iteration):
    smooed_pyramid.append(smoothing_down(smooed_pyramid[i],smooed_kernel))
    plt.subplot(sub_num+i+1)
    plt.imshow(smooed_pyramid[i+1])

fig.tight_layout()
plt.show()

# 필터 생성 없지 opencv의 가우시안 블러 사용
def gaussian_smoothing_down(img):
    smooed = cv2.GaussianBlur(img,(5,5),0)
    out_h, out_w, c = img.shape
    out_h //= 2
    out_w //= 2
    down_img = np.zeros([out_h, out_w, c], dtype=np.uint8)
    for y in range(out_h):
        for x in range(out_w):
            down_img[y, x] = smooed[y * 2, x * 2]
    return down_img

gaussian_smooed_pyramid = []
gaussian_smooed_pyramid.append(img)

fig = plt.figure(figsize=(13,13))
plt.subplot(sub_num)
plt.imshow(gaussian_smooed_pyramid[0])

for i in range(iteration):
    gaussian_smooed_pyramid.append(gaussian_smoothing_down(gaussian_smooed_pyramid[i]))
    plt.subplot(sub_num+i+1)
    plt.imshow(gaussian_smooed_pyramid[i+1])

fig.tight_layout()
plt.show()

## opencv resize

# inter_nearest
inter_nearest_pyramid = []
inter_nearest_pyramid.append(img)
fig = plt.figure(figsize=(13, 13))
fig.suptitle("Nearest", fontsize=20)
plt.subplot(sub_num)
plt.imshow(inter_nearest_pyramid[0])
for i in range(iteration):
    inter_nearest_pyramid.append(
        cv2.resize(inter_nearest_pyramid[i], dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST))
    plt.subplot(sub_num + i + 1)
    plt.imshow(inter_nearest_pyramid[i + 1])

fig.tight_layout()
plt.show()

# inter_linear
inter_linear_pyramid = []
inter_linear_pyramid.append(img)
fig = plt.figure(figsize=(13, 13))
fig.suptitle("Linear", fontsize=20)
plt.subplot(sub_num)
plt.imshow(inter_linear_pyramid[0])
for i in range(iteration):
    inter_linear_pyramid.append(
        cv2.resize(inter_linear_pyramid[i], dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR))
    plt.subplot(sub_num + i + 1)
    plt.imshow(inter_linear_pyramid[i + 1])
fig.tight_layout()
plt.show()

# inter_area
inter_area_pyramid = []
inter_area_pyramid.append(img)
fig = plt.figure(figsize=(13, 13))
fig.suptitle("Area", fontsize=20)
plt.subplot(sub_num)
plt.imshow(inter_area_pyramid[0])
for i in range(iteration):
    inter_area_pyramid.append(
        cv2.resize(inter_area_pyramid[i], dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
    plt.subplot(sub_num + i + 1)
    plt.imshow(inter_area_pyramid[i + 1])
fig.tight_layout()
plt.show()

# inter_cubic
inter_cubic_pyramid = []
inter_cubic_pyramid.append(img)
fig = plt.figure(figsize=(13, 13))
fig.suptitle("Cubic", fontsize=20)
plt.subplot(sub_num)
plt.imshow(inter_cubic_pyramid[0])
for i in range(iteration):
    inter_cubic_pyramid.append(
        cv2.resize(inter_cubic_pyramid[i], dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC))
    plt.subplot(sub_num + i + 1)
    plt.imshow(inter_cubic_pyramid[i + 1])
fig.tight_layout()
plt.show()
