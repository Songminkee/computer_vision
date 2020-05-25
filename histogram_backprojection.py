import numpy as np
import cv2
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def rgb_to_hsi(img):
    heigh,width,channel = img.shape
    img_flat = np.reshape(img.copy(),[-1,channel])/255
    i = np.mean(img_flat, -1, keepdims=True)
    if channel==1:
        h = np.zeros_like(img_flat[...,2])
        s = np.zeros_like(img_flat[...,2])
    else:
        r = img_flat[...,2]
        g = img_flat[...,1]
        b = img_flat[...,0]
        mins = np.min(img_flat,axis=1,keepdims=True)
        s = 1-mins/i
        theta = np.arccos(((r-g)+(r-b))/(np.sqrt((r-g)**2+(r-b)*(g-b))*2))* 180 / np.pi
        h = np.expand_dims(np.where(b>g,360-theta,theta),-1)
        h/=360
    hsi = np.concatenate([h,s,i],-1)
    return np.uint8(np.clip(np.reshape(hsi,[heigh,width,channel])*255,0,255))

def hsi_to_rgb(hsi):
    heigh, width, channel = hsi.shape
    hsi_flat = np.reshape(hsi.copy(), [-1, channel]) / 255
    h = hsi_flat[...,0] * 360
    s = hsi_flat[..., 1]
    i = hsi_flat[..., 2]
    r= np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)

    mask1 = h<=120
    mask2 = np.logical_and(h>120,h<=240)
    mask3 = h>240

    h[mask2] -=120
    h[mask3] -=240

    b = np.where(mask1,i*(1-s),b)
    r=np.where(mask1,np.clip(i*(1+s*np.cos(h*np.pi/180)/np.cos((60-h)*np.pi/180)),0.0,1.0),r)
    g=np.where(mask1, 3*i-(r+b),g)

    r = np.where(mask2,i*(1-s),r)
    g = np.where(mask2,np.clip(i*(1+s*np.cos(h*np.pi/180)/np.cos((60-h)*np.pi/180)),0.0,1.0),g)
    b= np.where(mask2,3*i-(r+g),b)

    g= np.where(mask3,i*(1-s),g)
    b= np.where(mask3,np.clip(i*(1+s*np.cos(h*np.pi/180)/np.cos((60-h)*np.pi/180)),0.0,1.0),b)
    r= np.where(mask3,3*i-(g+b),r)

    g = np.where(s==0,i,g)
    b = np.where(s == 0, i, b)
    r = np.where(s == 0, i, r)

    g = np.where(i==0,0,g)
    r = np.where(i == 0, 0, r)
    b = np.where(i == 0, 0, b)

    g = np.expand_dims(g,-1)
    b = np.expand_dims(b, -1)
    r = np.expand_dims(r, -1)
    bgr = np.reshape(np.concatenate([b,g,r],-1),[heigh,width,channel])

    return np.clip(bgr*255,0,255).astype(np.uint8)




img = cv2.imread('./data/kim.jpg')
img2 = cv2.imread('./data/girlsgeneration.JPG')
src_height, src_width ,_ = img.shape

hsi = rgb_to_hsi(img)

# hsi_to_rgb
# n_rgb = hsi_to_rgb(hsi)

L=256
q=64

hs_hist = np.zeros((q,q),np.float)
h_hist = np.zeros((q,1),np.int)
s_hist = np.zeros((q,1),np.int)

hsi_flat = np.reshape(hsi.copy(),[-1,3])
for i in range(len(hsi_flat)):
    hs_hist[int(hsi_flat[i,0]*q/L)][int(hsi_flat[i,1]*q/L)]+=1
    h_hist[int(hsi_flat[i,0]*q/L)]+=1
    s_hist[int(hsi_flat[i,1]*q/L)]+=1

hs_hist=hs_hist/(src_width*src_height)

x = np.arange(q)
y = np.arange(q)
xx,yy = np.meshgrid(x,y)

plt.subplot(121)
plt.bar(x,h_hist.squeeze())
plt.xlim([0,q])
plt.subplot(122)
plt.bar(x,s_hist.squeeze())
plt.xlim([0,q])

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(xx,yy,hs_hist.squeeze())
ax.legend()

plt.show()


dst_height,dst_width,_ = img2.shape
dst_hs_hist = np.zeros((q,q),np.float)
dst_hsi = rgb_to_hsi(img2)
hsi_flat = np.reshape(dst_hsi.copy(),[-1,3])
for i in range(len(hsi_flat)):
    dst_hs_hist[int(hsi_flat[i,0]*q/L)][int(hsi_flat[i,1]*q/L)]+=1

dst_hs_hist=dst_hs_hist/(dst_width*dst_height)

dst = np.zeros([dst_height,dst_width])
for y in range(dst_height):
    for x in range(dst_width):
        h=int(dst_hsi[y, x,0] * q / L)
        s=int(dst_hsi[y,x,1]*q/L)
        dst[y][x]=np.clip(hs_hist[h,s]/dst_hs_hist[h,s],0,1)*255


cv_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
cv_hsv2 = cv2.cvtColor(img2,cv2.COLOR_BGR2HSV)


cv_hs_hist = cv2.calcHist([cv_hsv], [0,1], None, [180,256], [0,180,0, 256])
cv2.normalize(cv_hs_hist,cv_hs_hist,0,255,cv2.NORM_MINMAX)

cv_dst=cv2.calcBackProject([cv_hsv2],[0,1], cv_hs_hist,[0,180,0,256],1)
disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)) # smooding mask
cv2.filter2D(cv_dst,-1,disc,cv_dst) # smoothing

_,thr = cv2.threshold(cv_dst,5,255,0)
thr = cv2.merge((thr,thr,thr))
cv_dst = cv2.bitwise_and(img2,thr)

cv2.imshow("ori",img2)
cv2.imshow("dst",dst)
cv2.imshow("cv_dst",cv_dst)

#
cv2.waitKey()
