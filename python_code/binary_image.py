import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import cv2
import matplotlib.pyplot as plt
import numpy as np


def Threshold(img, T):
    return np.uint8(img>T)*255

bi_img = Threshold(img,120)

plt.imshow(bi_img,cmap='gray')
plt.show()

img = cv2.imread('../data/lena.jpg',cv2.IMREAD_GRAYSCALE)

hist = np.zeros(256)
img_flat = np.reshape(img,-1)
for i in range(len(img_flat)):
    hist[img_flat[i]]+=1
hist = hist/len(img_flat)

fig = plt.figure()
plt.subplot(121)
plt.imshow(img,cmap='gray')
plt.subplot(122)
plt.bar(np.arange(256),hist)
plt.xlim([0,256])
fig.tight_layout()
plt.show()

# outsu
mu = 0

for i in range(256):
    mu += i * hist[i]

w = np.zeros(256)
mu0 = np.zeros(256)
mu1 = np.zeros(256)
T_bet = 0
w[0] = hist[0]
threshold=0
for i in range(1,256):
    w[i] = w[i-1]+hist[i]
    mu0[i] = (w[i-1]*mu0[i-1]+i*hist[i])/(w[i]+1e-10)
    mu1[i] = (mu - w[i]*mu0[i])/(1-w[i]+1e-10)
    n_t = w[i]*(1-w[i])*((mu0[i]-mu1[i])**2)

    if n_t>T_bet:
        T_bet = n_t
        threshold = i
otsu_img = Threshold(img,threshold)

plt.imshow(otsu_img,cmap='gray')
plt.show()

# opencv
ret, thr = cv2.threshold(img,120,255,cv2.THRESH_BINARY)
ret2, thr2 = cv2.threshold(img,120,255,cv2.THRESH_BINARY_INV)
ret3, thr3 = cv2.threshold(img,120,255,cv2.THRESH_TRUNC) # 픽셀값이 임계값 보다 크면 임계값, 작으면 픽셀값 그대로
ret4, thr4 = cv2.threshold(img,120,255,cv2.THRESH_TOZERO) # 픽셀값이 임계값 보다 크면 픽셀 값 그대로, 작으면 0
ret5, thr5 = cv2.threshold(img,120,255,cv2.THRESH_TOZERO_INV) # 픽셀값이 임계값 보다 크면 0, 작으면 픽셀값 그대로
ret6, thr6 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

fig = plt.figure()
plt.subplot(231)
plt.imshow(thr,cmap='gray')
plt.subplot(232)
plt.imshow(thr2,cmap='gray')
plt.subplot(233)
plt.imshow(thr3,cmap='gray')
plt.subplot(234)
plt.imshow(thr4,cmap='gray')
plt.subplot(235)
plt.imshow(thr5,cmap='gray')
plt.subplot(236)
plt.imshow(thr6,cmap='gray')
fig.tight_layout()
plt.show()