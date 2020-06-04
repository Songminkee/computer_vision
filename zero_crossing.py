import numpy as np
import matplotlib.pyplot as plt
import cv2
x = np.linspace(-20,20,100)
def gaussian(x,sigma):
    return (1/(np.sqrt(2*np.pi)*sigma))*np.exp(-1*((x**2)/(2*(sigma**2))))


plt.plot(x,gaussian(x,0.5))
plt.plot(x,gaussian(x,1.0))
plt.plot(x,gaussian(x,2.0))
plt.plot(x,gaussian(x,5.0))
plt.ylim(0,1)
plt.legend(['$sigma$=0.5','$sigma$=1.0','$sigma$=2.0','$sigma$=5.0'])
plt.show()

# 2d gaussian plot
def gaussian2d(x,y,sigma):
    x2 = x**2
    y2 = y**2
    sigma2 = sigma**2
    return (1/(2*np.pi*sigma2))*np.exp(-1*((x2+y2)/(2*sigma2)))


def LOG(sigma,plot=False):
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

LOG(5.0,True)


## LOG
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

img = cv2.imread('./data/food.jpg',cv2.IMREAD_GRAYSCALE)

sigma_value = [1.0,2.0,4.0,8.0]
fig = plt.figure(figsize=(13,13))
for i in range(1,5):
    plt.subplot(220+i)
    log_filter = LOG(sigma_value[i-1])
    log_img = LOG_conv(img,log_filter)
    plt.imshow(log_img,cmap='gray')
    plt.xlabel('$sigma$={}'.format(sigma_value[i-1]))
fig.tight_layout()
plt.show()