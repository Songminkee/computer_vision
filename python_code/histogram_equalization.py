import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../data/hill.jpg')

b_hist = np.zeros((256,1),np.int)
g_hist = np.zeros((256,1),np.int)
r_hist = np.zeros((256,1),np.int)
img_flat = np.reshape(img.copy(),(-1,3))

leng=len(img_flat)
for i in range(leng):
    b_hist[img_flat[i,0]]+=1
    g_hist[img_flat[i,1]]+=1
    r_hist[img_flat[i,2]]+=1

plt.subplot(231)
plt.bar(np.arange(256),b_hist.squeeze()/leng)
plt.xlim([0,256])
plt.subplot(232)
plt.bar(np.arange(256),g_hist.squeeze()/leng)
plt.xlim([0,256])
plt.subplot(233)
plt.bar(np.arange(256),r_hist.squeeze()/leng)
plt.xlim([0,256])

for i in range(1,256):
    b_hist[i]+=b_hist[i-1]
    g_hist[i]+=g_hist[i-1]
    r_hist[i]+=r_hist[i-1]
print(b_hist)
b_hist= b_hist*255/leng+0.5
g_hist= g_hist*255/leng+0.5
r_hist= r_hist*255/leng+0.5

new = np.zeros_like(img)

for y in range(len(new)):
    for x in range(len(new[0])):
        new[y,x,0] = b_hist[img[y,x,0]]
        new[y,x,1] = g_hist[img[y,x,1]]
        new[y,x,2] = r_hist[img[y,x,2]]

b_hist_n = np.zeros((256,1),np.int)
g_hist_n = np.zeros((256,1),np.int)
r_hist_n = np.zeros((256,1),np.int)

img_flat_n = np.reshape(new.copy(),(-1,3))
for i in range(leng):
    b_hist_n[img_flat_n[i,0]]+=1
    g_hist_n[img_flat_n[i,1]]+=1
    r_hist_n[img_flat_n[i,2]]+=1

plt.subplot(234)
plt.bar(np.arange(256),b_hist_n.squeeze()/leng)
plt.xlim([0,256])
plt.subplot(235)
plt.bar(np.arange(256),g_hist_n.squeeze()/leng)
plt.xlim([0,256])
plt.subplot(236)
plt.bar(np.arange(256),r_hist_n.squeeze()/leng)
plt.xlim([0,256])
plt.show()
cv_new_b = np.expand_dims(cv2.equalizeHist(img[...,0]),-1)
cv_new_g = np.expand_dims(cv2.equalizeHist(img[...,1]),-1)
cv_new_r = np.expand_dims(cv2.equalizeHist(img[...,2]),-1)
cv_new = np.concatenate([cv_new_b,cv_new_g,cv_new_r],-1)
cv2.imshow("ori",img)
cv2.imshow("equalization",new)
cv2.imshow("cv_equal",cv_new)
cv2.imshow("diff_two",abs(new-cv_new))
cv2.waitKey()