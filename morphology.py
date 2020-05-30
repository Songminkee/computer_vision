import numpy as np
import cv2
import matplotlib.pyplot as plt

img =cv2.cvtColor(cv2.imread('./data/baboon.jpg'),cv2.COLOR_BGR2RGB)
_,binary_im = cv2.threshold(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),175,255,cv2.THRESH_BINARY)

fig =plt.figure()
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(binary_im,cmap='gray')
fig.tight_layout()
plt.show()




## binary morphology
def bi_dilation(img,kernel,ker_center=None):
    ker_H, ker_W = kernel.shape
    if ker_center == None:
        ker_center = np.array([ker_H//2,ker_W//2])

    out = np.zeros_like(img)
    img//=np.max(img)
    for y in range(ker_H):
        for x in range(ker_W):
            if kernel[y,x]:
                y_diff = y-ker_center[0]
                x_diff = x-ker_center[1]
                h_pad = abs(y_diff)
                w_pad = abs(x_diff)
                if y_diff >0 and x_diff >0:
                    out+=np.pad(img[:-1*y_diff,:-1*x_diff],((h_pad,0),(w_pad,0)),'constant',constant_values=0)
                elif y_diff >0:
                    out += np.pad(img[:-1*y_diff,-1*x_diff:], ((h_pad,0), (0, w_pad)), 'constant', constant_values=0)
                elif x_diff >0:
                    out += np.pad(img[-1*y_diff:, :-1*x_diff], ((0,h_pad), (w_pad,0)), 'constant', constant_values=0)
                else:
                    out += np.pad(img[-1*y_diff:, -1*x_diff:], ((0, h_pad), (0,w_pad)), 'constant', constant_values=0)

    return np.uint8(out>0)*255

def bi_erosion(img,kernel,ker_center=None):
    ker_H, ker_W = kernel.shape
    if ker_center == None:
        ker_center = np.array([ker_H//2,ker_W//2])

    out = np.zeros_like(img)
    ker_sum = np.sum(kernel)
    img//=np.max(img)
    for y in range(ker_H):
        for x in range(ker_W):
            if kernel[y,x]:
                y_diff = ker_center[0]-y
                x_diff = ker_center[1]-x
                h_pad = abs(y_diff)
                w_pad = abs(x_diff)
                if y_diff >0 and x_diff >0:
                    out+=np.pad(img[:-1*y_diff,:-1*x_diff],((h_pad,0),(w_pad,0)),'constant',constant_values=0)
                elif y_diff >0:
                    out += np.pad(img[:-1*y_diff,-1*x_diff:], ((h_pad,0), (0, w_pad)), 'constant', constant_values=0)
                elif x_diff >0:
                    out += np.pad(img[-1*y_diff:, :-1*x_diff], ((0,h_pad), (w_pad,0)), 'constant', constant_values=0)
                else:
                    out += np.pad(img[-1*y_diff:, -1*x_diff:], ((0, h_pad), (0,w_pad)), 'constant', constant_values=0)

    return np.uint8(out==ker_sum)*255


struct_small=np.array([[1,1,1]])

struct_big = np.array([[0,1,0],
                   [1,1,1],
                   [0,1,0]])

fig =plt.figure(figsize=(13,13))
plt.subplot(321)
plt.imshow(binary_im,cmap='gray')
plt.xlabel('Original')


plt.subplot(323)
plt.imshow(bi_dilation(binary_im,struct_small),cmap='gray')
plt.xlabel('Dilation small')

plt.subplot(324)
plt.imshow(bi_erosion(binary_im,struct_small),cmap='gray')
plt.xlabel('Erosion small')

plt.subplot(325)
plt.imshow(bi_dilation(binary_im,struct_big),cmap='gray')
plt.xlabel('Dilation big')

plt.subplot(326)
plt.imshow(bi_erosion(binary_im,struct_big),cmap='gray')
plt.xlabel('Erosion big')

fig.tight_layout()
plt.show()


## color morphology
def dilation(img,kernel,ker_center=None):
    ker_H, ker_W = kernel.shape
    H,W,C = img.shape
    if ker_center == None:
        ker_center = np.array([ker_H//2,ker_W//2])
    out = np.zeros([H,W,C,ker_W*ker_H])
    count = 0
    for y in range(ker_H):
        for x in range(ker_W):
            y_diff = y-ker_center[0]
            x_diff = x-ker_center[1]
            h_pad = abs(y_diff)
            w_pad = abs(x_diff)
            if y_diff >0 and x_diff >0:
                out[...,count] = np.pad(img[:-1*y_diff,:-1*x_diff],((h_pad,0),(w_pad,0),(0,0)),'constant',constant_values=0)
            elif y_diff >0:
                out[...,count] = np.pad(img[:-1*y_diff,-1*x_diff:], ((h_pad,0), (0, w_pad),(0,0)), 'constant', constant_values=0)
            elif x_diff >0:
                out[...,count] = np.pad(img[-1*y_diff:, :-1*x_diff], ((0,h_pad), (w_pad,0),(0,0)), 'constant', constant_values=0)
            else:
                out[...,count] = np.pad(img[-1*y_diff:, -1*x_diff:], ((0, h_pad), (0,w_pad),(0,0)), 'constant', constant_values=0)
            count+=1

    return np.uint8(np.max(out,-1))

def erosion(img,kernel,ker_center=None):
    ker_H, ker_W = kernel.shape
    H,W,C = img.shape
    if ker_center == None:
        ker_center = np.array([ker_H//2,ker_W//2])
    out = np.zeros([H,W,C,ker_W*ker_H])
    count = 0

    for y in range(ker_H):
        for x in range(ker_W):
            y_diff = ker_center[0]-y
            x_diff = ker_center[1]-x
            h_pad = abs(y_diff)
            w_pad = abs(x_diff)
            if y_diff >0 and x_diff >0:
                out[...,count] = np.pad(img[:-1*y_diff,:-1*x_diff],((h_pad,0),(w_pad,0),(0,0)),'constant',constant_values=255)
            elif y_diff >0:
                out[...,count] = np.pad(img[:-1*y_diff,-1*x_diff:], ((h_pad,0), (0, w_pad),(0,0)), 'constant', constant_values=255)
            elif x_diff >0:
                out[...,count] = np.pad(img[-1*y_diff:, :-1*x_diff], ((0,h_pad), (w_pad,0),(0,0)), 'constant', constant_values=255)
            else:
                out[...,count] = np.pad(img[-1*y_diff:, -1*x_diff:], ((0, h_pad), (0,w_pad),(0,0)), 'constant', constant_values=255)
            count+=1

    return np.uint8(np.min(out,-1))

ker_small = np.zeros([3,3])
ker_big = np.zeros([5,5])

fig =plt.figure(figsize=(13,13))
plt.subplot(321)
plt.imshow(img)
plt.xlabel('Original')


plt.subplot(323)
plt.imshow(dilation(img,ker_small))
plt.xlabel('Dilation small')

plt.subplot(324)
plt.imshow(erosion(img,ker_small))
plt.xlabel('Erosion small')

plt.subplot(325)
plt.imshow(dilation(img,ker_big))
plt.xlabel('Dilation big')
plt.subplot(326)
plt.imshow(erosion(img,ker_big))
plt.xlabel('Erosion big')

fig.tight_layout()
plt.show()

##opencv
ker_small = np.ones([3,3])
ker_big = np.ones([5,5])

fig =plt.figure(figsize=(13,13))
plt.subplot(321)
plt.imshow(img)
plt.xlabel('Original')

plt.subplot(323)
plt.imshow(cv2.dilate(img,ker_small))
plt.xlabel('Dilation small')

plt.subplot(324)
plt.imshow(cv2.erode(img,ker_small))
plt.xlabel('Erosion small')

plt.subplot(325)
plt.imshow(cv2.dilate(img,ker_big))
plt.xlabel('Dilation big')
plt.subplot(326)
plt.imshow(cv2.erode(img,ker_big))
plt.xlabel('Erosion big')

fig.tight_layout()
plt.show()
