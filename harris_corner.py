import matplotlib.pyplot as plt
from util import *
import numpy as np
import cv2


# harris with triangle image
img = np.zeros([12,12])
img[3:8,3]=1
img[4:8,4]=1
img[5:8,5]=1
img[6:8,6]=1
img[7,7]=1
plt.imshow(img,cmap='gray')
plt.show()

def get_gaussian_kernel(sigma,size):
    if size % 2 == 0:
        size += 1
    aran = np.arange(-1 * (size // 2), size // 2 + 1)
    XX, YY = np.meshgrid(aran, aran)
    ker = gaussian2d(XX, YY, sigma)
    ker = ker/np.sum(ker) # normalization
    return ker

mask = get_gaussian_kernel(1, 3)
dy = np.pad(img[1:],((0,1),(0,0))) - np.pad(img[:-1],((1,0),(0,0))) # [-1,0,1]^T 마스크를 적용한 것과 동일
dx = np.pad(img[...,1:],((0,0),(0,1))) - np.pad(img[...,:-1],((0,0),(1,0))) # [-1,0,1] 마스크를 적용한 것과 동일

A_01 = np.expand_dims(conv(dx * dy, mask), -1)
A_00 = np.expand_dims(conv(dy * dy, mask), -1)
A_11 = np.expand_dims(conv(dx * dx, mask), -1)
A = np.concatenate([np.concatenate([A_00, A_01], -1), np.concatenate([A_01, A_11], -1)], 2)
print("point a\n",A[7,7])
print("point b\n",A[5,3])
print("point c\n",A[2,8])

k=0.04
deter = np.linalg.det(A)
trace = np.trace(np.transpose(A, (2, 3, 0, 1)))
C = deter - (k* (trace ** 2))
print("point a\n",C[7,7])
print("point b\n",C[5,3])
print("point c\n",C[2,8])


# harris with red_deer image
img = cv2.imread('./data/red_deer.jpg',cv2.IMREAD_GRAYSCALE)
plt.figure(figsize=(14,14))
plt.imshow(img,cmap='gray')
plt.show()

import time

def Harris_corner(img,threshold,use_det=True,k=0.04,max_value=255):
    mask = get_gaussian_kernel(1, 3)
    img = img/max_value
    dy = np.pad(img[1:],((0,1),(0,0))) - np.pad(img[:-1],((1,0),(0,0)))
    dx = np.pad(img[...,1:],((0,0),(0,1))) - np.pad(img[...,:-1],((0,0),(1,0)))

    A_01 = np.expand_dims(conv(dx * dy, mask), -1)
    A_00 = np.expand_dims(conv(dy * dy, mask), -1)
    A_11 = np.expand_dims(conv(dx * dx, mask), -1)
    A = np.concatenate([np.concatenate([A_00, A_01], -1), np.concatenate([A_01, A_11], -1)], 2)

    if use_det:
        deter = np.linalg.det(A)
        trace = np.trace(np.transpose(A, (2, 3, 0, 1)))
        C = deter - (k* (trace ** 2))
    else:
        eigens = np.linalg.eigvals(A)
        C = eigens[..., 0] * eigens[..., 1] - (k * ((eigens[..., 0] + eigens[..., 1]) ** 2))

    return np.where(C>threshold,C,0)

def draw_harris_circle(img,harris,print_harris_value=False):
    ys, xs = np.where(harris > 0)

    gray = np.expand_dims(img, -1)
    harris_img = np.concatenate([gray, gray, gray], -1)
    for i in range(len(ys)):
        harris_img = cv2.circle(harris_img, (xs[i], ys[i]), 10, (0, 0, 255),-1) # (x,y)로 들어감
        if print_harris_value:
            print("y = {}, x = {}, C = {}".format(ys[i],xs[i],harris[ys[i],xs[i]]))
    return harris_img

start=time.time()
harris = Harris_corner(img,0.02,False)
plt.figure(figsize=(14,14))
plt.imshow(draw_harris_circle(img,harris))
plt.show()
print("use eigen = ", time.time()-start)

start=time.time()
harris = Harris_corner(img,0.02,True)
plt.figure(figsize=(14,14))
plt.imshow(draw_harris_circle(img,harris))
plt.show()
print("use det = ", time.time()-start)