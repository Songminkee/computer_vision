import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('./data/lena.jpg',cv2.IMREAD_GRAYSCALE)
a = 32

## basic operation
fig = plt.figure()
plt.subplot(221)
plt.imshow(img,cmap='gray')
plt.subplot(222)
plt.imshow(np.clip(img+a,0,255),cmap='gray')
plt.subplot(223)
plt.imshow(np.clip(img-a,0,255),cmap='gray')
plt.subplot(224)
plt.imshow(255-img,cmap='gray')
fig.tight_layout()
plt.show()

## gamma operation
def gamma_operation(img,gamma,L):
    hat_img = img.copy() / float(L)
    return np.clip(L*((hat_img)**gamma),0,L)

fig = plt.figure()
plt.subplot(231)
plt.imshow(img,cmap='gray')
plt.subplot(232)
plt.imshow(gamma_operation(img,0.4,255),cmap='gray')
plt.xlabel('r=0.4')
plt.subplot(233)
plt.imshow(gamma_operation(img,0.67,255),cmap='gray')
plt.xlabel('r=0.67')
plt.subplot(234)
plt.imshow(gamma_operation(img,1.0,255),cmap='gray')
plt.xlabel('r=1.0')
plt.subplot(235)
plt.imshow(gamma_operation(img,1.5,255),cmap='gray')
plt.xlabel('r=1.5')
plt.subplot(236)
plt.imshow(gamma_operation(img,2.5,255),cmap='gray')
plt.xlabel('r=2.5')
fig.tight_layout()
plt.show()

## desolve
lena = cv2.cvtColor(cv2.imread('./data/lena.jpg'),cv2.COLOR_BGR2RGB)
girl = cv2.cvtColor(cv2.imread('./data/girl.jpg'),cv2.COLOR_BGR2RGB)
girl=cv2.resize(girl,lena.shape[:2])

alpha = 1
leng = 5
step = alpha/leng
fig = plt.figure()
for i in range(6):
    n_img = np.uint8(lena*alpha + girl*(1-alpha))

    plt.subplot(231+i)
    plt.imshow(n_img)
    alpha-=step
fig.tight_layout()
plt.show()


## convolution
# 박스
box_filter = np.ones((3,3))/9
# 가우시안
gaussian_filter = np.array([[0.,0.,0.0002,0.,0.],
                            [0.,0.0113,0.0837,0.0113,0.],
                            [0.0002,0.0837,0.6187,0.0837,0.0002],
                            [0.,0.0113,0.0837,0.0113,0.],
                            [0.,0.,0.0002,0.,0.]])
# 샤프닝
sharpening_filter = np.array([[0,-1,0],
                             [-1,5,-1],
                             [0,-1,0]])
# 수평 에지
horizontal_filter = np.array([[1,1,1],
                              [0,0,0],
                              [-1,-1,-1]])
# 수직 에지
vertical_filter = np.array([[1,0,-1],
                              [1,0,-1],
                              [1,0,-1]])
# 모션
motion_filter = np.array([[0.0304,0.0501,0.,0.,0.],
                            [0.0501,0.1771,0.0519,0.,0.],
                            [0.,0.0519,0.1771,0.0519,0.],
                            [0.,0.,0.0519,0.1771,0.0501],
                            [0.,0.,0.,0.0501,0.0304]])

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
    img_h,img_w,c = img.shape
    col = im2col(img,filter_h,filter_w)
    col_filetr = filter.reshape((1,-1)).T
    out = np.dot(col, col_filetr)
    return np.clip(np.transpose(out.reshape((c, img_h, img_w)),(1, 2,0)),0,255)

fig = plt.figure(figsize=(13,13))
plt.subplot(331)
plt.imshow(lena)

plt.subplot(334)
box = np.uint8(conv(lena,box_filter))
plt.xlabel("Box")
plt.imshow(box)

plt.subplot(335)
gau = np.uint8(conv(lena,gaussian_filter))
plt.xlabel("Gaussian")
plt.imshow(gau)

plt.subplot(336)
sharp = np.uint8(conv(lena,sharpening_filter))
plt.xlabel("Sharpening")
plt.imshow(sharp)

plt.subplot(337)
hori = np.uint8(conv(lena,horizontal_filter))
plt.xlabel("Horizontal")
plt.imshow(hori)

plt.subplot(338)
veti = np.uint8(conv(lena,vertical_filter))
plt.xlabel("Vertical")
plt.imshow(veti)

plt.subplot(339)
motion = np.uint8(conv(lena,motion_filter))
plt.xlabel("Motion_filter")
plt.imshow(motion)

fig.tight_layout()
plt.show()

## convolution-opencv
fig = plt.figure(figsize=(13,13))
plt.subplot(331)
plt.imshow(lena)

plt.subplot(334)
cv_box = cv2.filter2D(lena,-1,box_filter)
plt.xlabel("Box")
plt.imshow(cv_box)

plt.subplot(335)
cv_gau = cv2.filter2D(lena,-1,gaussian_filter)
plt.xlabel("Gaussian")
plt.imshow(cv_gau)

plt.subplot(336)
cv_sharp = cv2.filter2D(lena,-1,sharpening_filter)
plt.xlabel("Sharpening")
plt.imshow(cv_sharp)

plt.subplot(337)
cv_hori = cv2.filter2D(lena,-1,horizontal_filter)
plt.xlabel("Horizontal")
plt.imshow(cv_hori)

plt.subplot(338)
cv_veti = cv2.filter2D(lena,-1,vertical_filter)
plt.xlabel("Vertical")
plt.imshow(cv_veti)

plt.subplot(339)
cv_motion = cv2.filter2D(lena,-1,motion_filter)
plt.xlabel("Motion_filter")
plt.imshow(cv_motion)
plt.show()

# motion kernel
def apply_motion_blur(size, angle):
    k = np.zeros((size, size), dtype=np.float32)
    k[ (size-1)// 2 , :] = np.ones(size, dtype=np.float32)
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D( (size / 2 -0.5 , size / 2 -0.5 ) , angle, 1.0), (size, size) )
    k = k * ( 1.0 / np.sum(k) )
    return k

size = 51
large_motion_filter=apply_motion_blur(size,45)

cv_large_motion = cv2.filter2D(lena,-1,large_motion_filter)
plt.xlabel("Large Motion")
plt.imshow(cv_large_motion)
plt.show()

## median filter
salt_idx = np.random.random(lena.shape[:2])
salt_img = lena.copy()
salt_img[salt_idx>0.9] =255

def median(img,filter_size):
    img_h,img_w,c = img.shape
    pad= (filter_size-1)//2
    out_img = np.zeros((img_h,img_w,c))
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad,  cv2.BORDER_REPLICATE)
    img = np.transpose(img,(2,0,1))

    for y in range(img_h):
        for x in range(img_w):
            partial = img[:,y:y+filter_size,x:x+filter_size].reshape(c,-1)
            partial.sort()
            out_img[y,x]= partial[:,(filter_size**2)//2]

    return np.uint8(out_img)

fig = plt.figure(figsize=(13,13))
plt.subplot(221)
plt.imshow(lena)

plt.subplot(222)
plt.imshow(salt_img)
plt.xlabel("salt_and_pepper")

plt.subplot(223)
salt_gaussian = np.uint8(conv(salt_img,gaussian_filter))
plt.imshow(salt_gaussian)
plt.xlabel("gaussian")

plt.subplot(224)
salt_median = median(salt_img,5)
plt.imshow(salt_median)
plt.xlabel("median")

fig.tight_layout()
plt.show()

# opencv - median, bilateral
fig=plt.figure(figsize=(13,13))
plt.subplot(221)
plt.imshow(salt_img)
plt.xlabel("salt")

cv_median_img = cv2.medianBlur(salt_img,5)

plt.subplot(222)
plt.imshow(cv_median_img)
plt.xlabel("median")

plt.subplot(223)
plt.imshow(lena[200:400,200:400])
plt.xlabel("lena")

cv_bilateral_img = cv2.bilateralFilter(lena[200:400,200:400], 10,12.0,16.0)

plt.subplot(224)
plt.imshow(cv_bilateral_img)
plt.xlabel("bilateral")
fig.tight_layout()
plt.show()


## interpolation

def rotation(img,angle,method):
    angle = angle /180*np.pi
    cos =np.cos(angle)
    sin = np.sin(angle)
    out = np.zeros_like(img)
    if(method=='bilinear'):
        for y in range(len(img)):
            for x in range(len(img[0])):
                x_1 = x-2*cos - y*sin
                y_1 = x*sin+y*cos
                if x_1<0 or y_1<0 :
                    continue
                if abs(int(x_1+1)-np.clip(int(x_1+1),0,img.shape[0]-1)) >0 or abs(int(y_1+1)-np.clip(int(y_1+1),0,img.shape[1]-1))>1:
                    continue


                alpha = x_1 - int(x_1)
                beta = y_1 - int(y_1)
                xx = int(x_1)
                yy = int(y_1)

                if xx == img.shape[1]-1 and yy ==img.shape[0]-1:
                    out[y,x]=img[yy,xx]
                elif xx == img.shape[1]-1:
                    out[y,x]=img[yy,xx]*(1-beta)+img[yy+1,xx]*beta
                elif yy == img.shape[0]-1:
                    out[y,x]=img[yy,xx]*(1-alpha) + img[yy,xx+1]*alpha
                else:
                    pixel1 = img[yy,xx]*(1-alpha) + img[yy,xx+1]*alpha
                    pixel2 = img[yy+1,xx]*(1-alpha) + img[yy+1,xx+1]*alpha
                    out[y,x] = pixel1*(1-beta)+pixel2*beta
    elif method == 'nearest':
        for y in range(len(img)):
            for x in range(len(img)):
                x_1 = x * cos - y * sin
                y_1 = x * sin + y * cos
                if x_1 < 0 or y_1 < 0:
                    continue
                if abs(int(x_1 + 1) - np.clip(int(x_1 + 1), 0, img.shape[0] - 1)) > 0 or abs(
                        int(y_1 + 1) - np.clip(int(y_1 + 1), 0, img.shape[1] - 1)) > 1:
                    continue
                out[y,x] = img[np.clip(int(y_1+0.5),0,img.shape[1]-1),np.clip(int(x_1 + 0.5), 0, img.shape[0] - 1)]
    return out

fig = plt.figure(figsize=(13,13))
plt.subplot(221)
plt.imshow(lena)

roi_resize=lena[200:300,100:200]
plt.subplot(222)
plt.imshow(roi_resize)


plt.subplot(223)
plt.imshow(rotation(roi_resize,10,'nearest'))
plt.xlabel("nearest negihbor")

plt.subplot(224)
plt.imshow(rotation(roi_resize,10,'bilinear'))
plt.xlabel('bilinear interpolation')
fig.tight_layout()
plt.show()
