import numpy as np
import cv2

## convolution
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

## gaussian
def gaussian2d(x,y,sigma):
    x2 = x**2
    y2 = y**2
    sigma2 = sigma**2
    return (1/(2*np.pi*sigma2))*np.exp(-1*((x2+y2)/(2*sigma2)))

def make_gaussian(sigma):
    size = int(sigma * 6)
    if size % 2 == 0:
        size += 1
    aran = np.arange(-1 * (size // 2), size // 2 + 1)
    XX, YY = np.meshgrid(aran, aran)
    ker = gaussian2d(XX, YY, sigma)
    ker = ker / np.sum(ker)  # normalization
    return ker

def get_gaussian_kernel(sigma,size):
    if size % 2 == 0:
        size += 1
    aran = np.arange(-1 * (size // 2), size // 2 + 1)
    XX, YY = np.meshgrid(aran, aran)
    ker = gaussian2d(XX, YY, sigma)
    ker = ker/np.sum(ker) # normalization
    return ker

def gaussian_blur(img,sigma):
    gau_filter = make_gaussian(sigma)
    gau = np.uint8(conv(img, gau_filter).squeeze())
    return gau

## sobel
def direction_quantization(atan):
    ret = np.where(np.logical_and(atan >= 22.5,atan < 67.5),7,4)
    ret = np.where(np.logical_and(atan >= 67.5, atan < 111.5), 6, ret)
    ret = np.where(np.logical_and(atan >= 111.5, atan < 157.5), 5, ret)
    return ret

def sobel(img):
    sobel_mask_y = np.array([[-1, -2, -1],
                             [0, 0, 0],
                             [1, 2, 1]])

    sobel_mask_x = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]])

    dy_image = conv(img, sobel_mask_y).squeeze()
    dx_image = conv(img, sobel_mask_x).squeeze()
    dy_image[0, :] = dy_image[-1, :] = 0
    dx_image[:, 0] = dx_image[:, -1] = 0

    Sobel = np.uint8(np.sqrt((dy_image ** 2) + (dx_image ** 2)))
    atan = np.arctan2(dy_image, dx_image) / np.pi * 180
    atan += 90
    atan[atan > 180] = 360 - atan[atan > 180]
    atan[atan < 0] += 180
    direc = direction_quantization(atan)
    return Sobel, direc

## canny
def non_maximum_suppression(img,direction):
    e_img = np.pad(img[:, 1:], ((0, 0), (0, 1)),constant_values=255)
    w_img = np.pad(img[:, :-1], ((0, 0), (1, 0)),constant_values=255)
    n_img = np.pad(img[:-1, :], ((1, 0), (0, 0)),constant_values=255)
    s_img = np.pad(img[1:, :], ((0, 1), (0, 0)),constant_values=255)
    sw_img = np.pad(s_img[:, 1:], ((0, 0), (0, 1)),constant_values=255)
    nw_img = np.pad(n_img[:, 1:], ((0, 0), (0, 1)),constant_values=255)
    se_img = np.pad(s_img[:, :-1], ((0, 0), (1, 0)),constant_values=255)
    ne_img = np.pad(n_img[:, :-1], ((0, 0), (1, 0)),constant_values=255)

    ret = np.where(direction == 4, np.logical_and(img >= n_img, img >= s_img), False)
    ret = np.where(direction == 5, np.logical_and(img >= ne_img, img >= sw_img), ret)
    ret = np.where(direction == 6, np.logical_and(img >= e_img, img >= w_img), ret)
    ret = np.where(direction == 7, np.logical_and(img >= nw_img, img >= se_img), ret)
    return ret

def neighbor(img):
    img = np.uint8(img)
    w_img = np.pad(img[:, 1:], ((0, 0), (0, 1)))
    e_img = np.pad(img[:, :-1], ((0, 0), (1, 0)))
    n_img = np.pad(img[:-1, :], ((1, 0), (0, 0)))
    s_img = np.pad(img[1:, :], ((0, 1), (0, 0)))
    sw_img = np.pad(s_img[:, 1:], ((0, 0), (0, 1)))
    nw_img = np.pad(n_img[:, 1:], ((0, 0), (0, 1)))
    se_img = np.pad(s_img[:, :-1], ((0, 0), (1, 0)))
    ne_img = np.pad(n_img[:, :-1], ((0, 0), (1, 0)))
    return (w_img+e_img+n_img+s_img+sw_img+nw_img+se_img+ne_img+img)>0

def hysteresis_thresholding(img,th_low,th_high,direction):
    non_maxim = np.where(non_maximum_suppression(img,direction),img,0)
    high = non_maxim>th_high
    nei = neighbor(high)
    low = np.where(nei,non_maxim>th_low,0)
    return np.uint8(low)*255

def canny(img,th_low,th_high):
    sobel_im,direction = sobel(img)
    canny_img = hysteresis_thresholding(sobel_im, th_low, th_high, direction)
    return canny_img