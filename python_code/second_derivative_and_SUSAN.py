import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
import matplotlib.pyplot as plt
import cv2
from util import *

img =cv2.imread('../data/red_deer.jpg',cv2.IMREAD_GRAYSCALE)

plt.imshow(img,cmap='gray')
plt.show()

# Hessian feature
def Hessian_feature(img,threshold,sigma=1,kernel_size=3,max_value=255):
    # Gaussian Smoothing
    mask = get_gaussian_kernel(sigma,kernel_size)
    img = (conv(img,mask)/max_value).squeeze()

    # Second Derivative Y
    dy = np.pad(img[1:], ((0, 1), (0, 0))) - np.pad(img[:-1], ((1, 0), (0, 0)))
    dy2 = np.pad(dy[1:], ((0, 1), (0, 0))) - np.pad(dy[:-1], ((1, 0), (0, 0)))

    # Second Derivative X
    dx = np.pad(img[..., 1:], ((0, 0), (0, 1))) - np.pad(img[..., :-1], ((0, 0), (1, 0)))
    dx2 = np.pad(dx[..., 1:], ((0, 0), (0, 1))) - np.pad(dx[..., :-1], ((0, 0), (1, 0)))

    # Derivate Y and X
    dydx = np.pad(dy[..., 1:], ((0, 0), (0, 1))) - np.pad(dy[..., :-1], ((0, 0), (1, 0)))

    # Determinant of Hessian Matrix
    Determinant = dy2*dx2 - (dydx**2)
    # LOG (=Trace of Hessian Matrix)
    LOG = dy2+dx2
    return np.where(Determinant>threshold,Determinant,0), LOG

def Hessian_LOG_test(log_img,threshold=None):
    if threshold == None:
        threshold = np.max(log_img)*0.05
    e_img = np.pad(log_img[:,1:],((0,0),(0,1)))
    w_img = np.pad(log_img[:,:-1],((0,0),(1,0)))
    n_img = np.pad(log_img[:-1,:],((1,0),(0,0)))
    s_img = np.pad(log_img[1:, :], ((0,1), (0, 0)))
    se_img = np.pad(s_img[:,1:],((0,0),(0,1)))
    ne_img = np.pad(n_img[:, 1:], ((0, 0), (0, 1)))
    sw_img = np.pad(s_img[:,:-1],((0,0),(1,0)))
    nw_img = np.pad(n_img[:,:-1], ((0, 0), (1, 0)))

    w_vs_e = np.int8(np.logical_and(np.absolute(w_img-e_img)>=threshold,sign(w_img,e_img)))
    s_vs_n = np.int8(np.logical_and(np.absolute(n_img - s_img) >= threshold,sign(n_img,s_img)))
    sw_vs_ne = np.int8(np.logical_and(np.absolute(sw_img - ne_img) >= threshold,sign(sw_img,ne_img)))
    nw_vs_se = np.int8(np.logical_and(np.absolute(nw_img - se_img) >= threshold,sign(nw_img,se_img)))

    return np.uint8(w_vs_e+s_vs_n+sw_vs_ne+nw_vs_se>=1)*255


Det_C,LOG_C = Hessian_feature(img,0.1,1,7)

plt.imshow(draw_featrue_point(img,Det_C))
plt.show()

plt.imshow(Hessian_LOG_test(LOG_C),cmap='gray')
plt.show()

plt.imshow(get_LOG_img(img,1),cmap='gray')
plt.show()

# SUSAN
def get_circle_filter(filter_size):
    radius = filter_size // 2
    if filter_size%2:
        filter = np.zeros((2 * radius + 1, 2 * radius + 1))
        y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    else:
        print("Filter size must be odd.")
        return
    mask = x ** 2 + y ** 2 <= radius ** 2
    filter[mask] = 1
    return filter

def get_susan_filter():
    filter = np.array([[0,0,1,1,1,0,0],
                       [0,1,1,1,1,1,0],
                       [1,1,1,1,1,1,1],
                       [1,1,1,1,1,1,1],
                       [1,1,1,1,1,1,1],
                       [0,1,1,1,1,1,0],
                       [0,0,1,1,1,0,0]])
    return filter

def usan_area(img, filter,threshold,max_value=255):
    filter_h,filter_w = filter.shape
    if len(img.shape) == 3:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    H, W = img.shape
    img = np.expand_dims(img/max_value, -1)

    u_pad_h = (filter_h-1)//2
    d_pad_h = (filter_h-1)//2
    l_pad_w = (filter_w-1)//2
    r_pad_w = (filter_w-1)//2
    if (filter_h-1) %2 ==1:
        u_pad_h +=1
    if (filter_w-1)%2 ==1:
        l_pad_w +=1

    input_data = cv2.copyMakeBorder(img, u_pad_h, d_pad_h, l_pad_w, r_pad_w, cv2.BORDER_CONSTANT)
    input_data= np.expand_dims(input_data,-1)
    input_data = np.transpose(input_data,(2,0,1))
    col = np.zeros(( 1, filter_h, filter_w, H, W))

    for y in range(filter_h):
        y_max = y + H
        for x in range(filter_w):
            x_max = x + W
            col[:, y, x, :, :] = input_data[:, y:y_max:1, x:x_max:1]

    col = np.transpose(col,(0,3,4,1,2)).reshape(H*W, -1)
    usan = np.abs(np.transpose(img,(2,0,1)).reshape([-1,1]) - col*filter.reshape((1,-1)))<=threshold
    return np.sum(usan,axis=1).reshape(H,W)

def SUSAN(img,threshold,threshold2=None,q=None,filter=None):
    if filter==None:
        filter = get_susan_filter()
    if threshold2 ==None:
        threshold2 = np.sum(filter)*0.5
    if q==None:
        q = np.sum(filter)*0.75
    USAN = usan_area(img,filter,threshold)
    return np.where(USAN<=threshold2,q-USAN,0)

SUSAN_feat = SUSAN(img,0.4)

plt.imshow(draw_featrue_point(img,SUSAN_feat),cmap='gray')
plt.show()