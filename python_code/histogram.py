import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../data/lena.jpg')
cv2.imshow("Lena",img)
# print(img)
r_hist = np.zeros((256))
b_hist = np.zeros((256))
g_hist = np.zeros((256))
cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
im_flat = np.reshape(img.copy(),(-1,3))
leng = len(im_flat)
for i in range(leng):
    b_hist[im_flat[i][0]]+=1
    g_hist[im_flat[i][1]]+=1
    r_hist[im_flat[i][2]]+=1
plt.subplot(231)
plt.plot(b_hist,color='b')
plt.xlim([0,256])

plt.subplot(232)
plt.plot(g_hist,color='g')
plt.xlim([0,256])

plt.subplot(233)
plt.plot(r_hist,color='r')
plt.xlim([0,256])



## using cv
color = ['b','g','r']
for i in range(3):
    hist = cv2.calcHist([img], [i], None, [256], [0, 256],)
    plt.subplot(234+i)
    plt.plot(hist,color=color[i])
    plt.xlim([0,256])

plt.show()

plt.bar(np.arange(256),b_hist/leng)
plt.show()