import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
import cv2
import matplotlib.pyplot as plt

img =cv2.imread('../data/home.jpg',cv2.IMREAD_GRAYSCALE)

plt.imshow(img,cmap='gray')
plt.show()

def moravec(img):
    h,w = img.shape
    SSD01 = (img-np.pad(img[...,1:],((0,0),(0,1))))**2
    SSD01[...,-1]=0
    SSD01 = np.pad(SSD01,((1,1),(1,1)))
    SSD0m1 = (img - np.pad(img[..., :-1], ((0, 0), (1, 0)))) ** 2
    SSD0m1[..., 0] = 0
    SSD0m1 = np.pad(SSD0m1, ((1, 1), (1, 1)))
    SSD10 = (img-np.pad(img[1:],((0,1),(0,0))))**2
    SSD10[-1,...]=0
    SSD10 = np.pad(SSD10, ((1, 1), (1, 1)))
    SSDm10 = (img-np.pad(img[:-1],((1,0),(0,0))))**2
    SSDm10[0,...]=0
    SSDm10 = np.pad(SSDm10, ((1, 1), (1, 1)))
    S01=np.zeros([h,w,3,3])
    S0m1 = np.zeros([h, w, 3, 3])
    S10 = np.zeros([h, w, 3, 3])
    Sm10 = np.zeros([h, w, 3, 3])
    for i in range(h):
        for j in range(w):
            S01[i,j]=SSD01[i:i+3,j:j+3]
            S0m1[i,j] = SSD0m1[i:i+3,j:j+3]
            S10[i,j] = SSD10[i:i+3,j:j+3]
            Sm10[i,j] = SSDm10[i:i+3,j:j+3]

    confidence = np.min([np.sum(np.sum(S01,axis=2),axis=2),np.sum(np.sum(S0m1,axis=2),axis=2),np.sum(np.sum(S10,axis=2),axis=2),np.sum(np.sum(Sm10,axis=2),axis=2)],axis=0)

    return confidence,(S01,S0m1,S10,Sm10)


_,Ss = moravec(img)

img_ex = np.expand_dims(img,-1)

plt.figure(figsize=(15,15))
tmp_img= np.concatenate([img_ex,img_ex,img_ex],-1)
tmp_img[128,283,0]=255
tmp_img[128,283,1]=0
tmp_img[128,283,2]=0
tmp_img[100,280,0]=0
tmp_img[100,280,1]=0
tmp_img[100,280,2]=255
roi = tmp_img[80:140,250:320]
plt.imshow(roi)
plt.show()


fig=plt.figure(figsize=(13,13))
plt.subplot(421)
plt.imshow(Ss[0][129,284],cmap='gray')
plt.xlabel('S01_red')
plt.subplot(423)
plt.imshow(Ss[1][129, 284], cmap='gray')
plt.xlabel('S0m1_red')
plt.subplot(425)
plt.imshow(Ss[2][129, 284], cmap='gray')
plt.xlabel('S10_red')
plt.subplot(427)
plt.imshow(Ss[3][129, 284], cmap='gray')
plt.xlabel('Sm10_red')
plt.subplot(422)
plt.imshow(Ss[0][101, 281], cmap='gray')
plt.xlabel('S01_blue')
plt.subplot(424)
plt.imshow(Ss[1][101, 281], cmap='gray')
plt.xlabel('S0m1_blue')
plt.subplot(426)
plt.imshow(Ss[2][101, 281], cmap='gray')
plt.xlabel('S10_blue')
plt.subplot(428)
plt.imshow(Ss[3][101, 281], cmap='gray')
plt.xlabel('Sm10_blue')
fig.tight_layout()
plt.show()