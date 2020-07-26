import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
import cv2
import matplotlib.pyplot as plt

img =cv2.cvtColor(cv2.imread('../data/food.jpg'),cv2.COLOR_BGR2GRAY)


def im2col(input_data, filter_h, filter_w):
    H, W, C = input_data.shape

    u_pad_h = (filter_h-1)//2
    d_pad_h = (filter_h-1)//2
    l_pad_w = (filter_w-1)//2
    r_pad_w = (filter_w-1)//2
    if (filter_h-1) %2 ==1:
        u_pad_h +=1
    if (filter_w-1)%2 ==1:
        l_pad_w +=1
    input_data = cv2.copyMakeBorder(input_data, u_pad_h, d_pad_h, l_pad_w, r_pad_w, cv2.BORDER_REPLICATE)
    if C==1:
        input_data= np.expand_dims(input_data,-1)
    img = np.transpose(input_data,(2,0,1))
    col = np.zeros(( C, filter_h, filter_w, H, W))

    for y in range(filter_h):
        y_max = y + H
        for x in range(filter_w):
            x_max = x + W
            col[:, y, x, :, :] = img[:, y:y_max:1, x:x_max:1]

    col = np.transpose(col,(0,3,4,1,2)).reshape(C*H*W, -1)

    return col


def conv(img,filter):
    filter_h ,filter_w = filter.shape
    if len(img.shape) ==3:
        img_h,img_w,c = img.shape
    elif len(img.shape)==2:
        img_h,img_w = img.shape
        c=1
        img = np.expand_dims(img,-1)
    else:
        print("Length of image shape must over 2")
        return
    col = im2col(img,filter_h,filter_w)
    col_filetr = filter.reshape((1,-1)).T
    out = np.dot(col, col_filetr)
    return np.transpose(out.reshape((c, img_h, img_w)),(1, 2,0))

sobel_mask_y = np.array([[-1,-2,-1],
                    [0,0,0],
                    [1,2,1]])

sobel_mask_x = np.array([[-1,0,1],
                    [-2,0,2],
                    [-1,0,1]])

dy_image =conv(img,sobel_mask_y).squeeze()
dx_image =conv(img,sobel_mask_x).squeeze()

dy_image[0,:] = dy_image[-1,:] = 0 # 맨 끝부분은 0으로 처리한다.
dx_image[:,0] = dx_image[:,-1] = 0

S = np.uint8(np.sqrt((dy_image**2)+(dx_image**2)))



# image plot
fig = plt.figure(figsize=(13,13))

plt.subplot(221)
plt.imshow(img,cmap='gray')

plt.subplot(222)
plt.imshow(S,cmap='gray')
plt.xlabel('Edge strength')

plt.subplot(223)
plt.imshow(np.uint8(abs(dx_image)),cmap='gray')
plt.xlabel('dx')

plt.subplot(224)
plt.imshow(np.uint8(abs(dy_image)),cmap='gray')
plt.xlabel('dy')

fig.tight_layout()
plt.show()

## OpenCV

cv_sobel_y = cv2.Sobel(img,cv2.CV_64F,0,1)
cv_sobel_x = cv2.Sobel(img,cv2.CV_64F,1,0)
cv_s = np.uint8(np.sqrt((cv_sobel_x**2)+(cv_sobel_y**2)))


# image plot
fig = plt.figure(figsize=(13,13))

plt.subplot(221)
plt.imshow(img,cmap='gray')

plt.subplot(222)
plt.imshow(cv_s,cmap='gray')
plt.xlabel('Edge strength')

plt.subplot(223)
plt.imshow(np.uint8(abs(cv_sobel_x)),cmap='gray')
plt.xlabel('dx')

plt.subplot(224)
plt.imshow(np.uint8(abs(cv_sobel_y)),cmap='gray')
plt.xlabel('dy')

fig.tight_layout()
plt.show()



