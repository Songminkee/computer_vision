import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
import cv2
import matplotlib.pyplot as plt
import queue

img = cv2.imread('../data/lena.jpg',cv2.IMREAD_GRAYSCALE)
ret,thr = cv2.threshold(img,200,255,cv2.THRESH_BINARY_INV)

flood = thr.copy()/255.
flood -=1
plt.imshow(flood.copy()*-1,cmap='gray')
plt.show()

def flood_fill(img,j,i,label):
    q = queue.Queue()
    q.put((j,i))
    while q.qsize():
        (y,x)= q.get()
        if img[y,x] == -1:
            left=right=x
            while left-1>0 and img[y,left-1] == -1:
                left-=1
            while right+1<img.shape[1] and img[y,right+1] ==-1:
                right+=1
            for z in range(left,right+1):
                img[y,z]=label
                if y-1>0 and img[y-1,z]==-1 and (z==left or (z-1>0 and img[y-1][z-1] !=-1)):
                    q.put((y-1,z))
                if y+1<img.shape[1] and img[y+1,z]==-1 and (z==left or (z-1>0 and img[y+1][z-1] !=-1)):
                    q.put((y+1,z))


label = 1
for j in range(img.shape[0]):
    for i in range(img.shape[1]):
        if(flood[j,i]==-1):
            flood_fill(flood,j,i,label)
            label+=1

floos = np.zeros((img.shape[0],img.shape[1],label))
for i in range(1,label+1):
    floos[...,i-1] = flood == i
floos = floos.astype(np.float32)
fig = plt.figure()
for i in range(label):
    plt.imshow(floos[...,i],cmap='gray')
    plt.show()
