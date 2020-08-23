import numpy as np
import cv2
import matplotlib.pyplot as plt

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

# LOG
def sign(a,b):
    sign1 = np.logical_and(a<0,b>0)
    sign2 = np.logical_and(b<0,a>0)
    return np.logical_or(sign1,sign2)

def LOG_conv(img,filter,threshold=None):
    log_img=conv(img,filter).squeeze()
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

def get_LOG_filter(sigma,plot=False):
    size = int(sigma * 6)
    if  size %2 == 0:
        size+=1
    aran = np.arange(-1*(size//2),size//2+1)
    XX, YY = np.meshgrid(aran,aran)
    xx2 = XX**2
    yy2 = YY**2
    sigma2 = sigma**2
    log = ((xx2+yy2-(2*sigma2))/(sigma2**2)) * gaussian2d(XX,YY,sigma)
    if plot:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_wireframe(XX, YY, log)
        plt.show()
    return log

def get_LOG_img(img,sigma):
    log_filter = get_LOG_filter(sigma)
    log_img = LOG_conv(img, log_filter)
    return log_img

# harris feature
def Harris_corner(img,threshold=0.02,use_det=True,k=0.04,max_value=255):
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

# draw feature
def draw_featrue_point(img,C,print_harris_value=False,dot_size=10):
    ys, xs = np.where(C > 0)

    gray = np.expand_dims(img, -1)
    feature_point_img = np.concatenate([gray, gray, gray], -1)
    for i in range(len(ys)):
        feature_point_img = cv2.circle(feature_point_img, (xs[i], ys[i]),dot_size, (0, 0, 255),-1) # (x,y)로 들어감
        if print_harris_value:
            print("y = {}, x = {}, C = {}".format(ys[i],xs[i],C[ys[i],xs[i]]))
    return feature_point_img