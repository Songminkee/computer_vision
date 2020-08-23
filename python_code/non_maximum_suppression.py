import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from util import *
import matplotlib.pyplot as plt
import numpy as np
import cv2


img = cv2.imread('../data/red_deer.jpg',cv2.IMREAD_GRAYSCALE)
plt.imshow(img,cmap='gray')
plt.show()


def NMS(feature,n=8,threshold=0.02):
    feature = np.expand_dims(feature,-1)
    n_r = np.pad(feature[1:], ((0, 1), (0, 0),(0,0)))

    n_l = np.pad(feature[:-1], ((1, 0), (0, 0),(0,0)))
    n_d = np.pad(feature[:,1:], ((0, 0), (0, 1),(0,0)))
    n_u = np.pad(feature[:,:-1], ((0, 0), (1, 0),(0,0)))

    if n==8:
        n_ul = np.pad(n_u[:-1], ((1, 0), (0, 0),(0,0)))
        n_ur = np.pad(n_u[1:], ((0, 1), (0, 0),(0,0)))
        n_dl = np.pad(n_d[:-1], ((1, 0), (0, 0),(0,0)))
        n_dr = np.pad(n_d[1:], ((0, 1), (0, 0),(0,0)))
        ret = np.concatenate([feature,n_r,n_l,n_d,n_u,n_ul,n_ur,n_dl,n_dr],axis=-1)
    else:
        ret = np.concatenate([feature,n_r,n_l,n_d,n_u],axis=-1)
    ret = np.expand_dims(np.argmax(ret,-1),-1)

    return np.squeeze(np.where(np.logical_and(ret==0,feature>threshold),feature,0))

harris = Harris_corner(img,threshold=0.02)
fig = plt.figure(figsize=(13,13))
plt.subplot(121)
plt.imshow(draw_featrue_point(img,harris,dot_size=5))
plt.xlabel('before NMS')

NMS_harris = NMS(harris)
plt.subplot(122)
plt.imshow(draw_featrue_point(img,NMS_harris,dot_size=5))
plt.xlabel('after NMS')
fig.tight_layout()
plt.show()

print(np.sum(harris>0),np.sum(NMS_harris>0))

#NMS(harris)
