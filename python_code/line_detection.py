import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
import cv2
import matplotlib.pyplot as plt
from util import *
import queue

img = cv2.imread('../data/food.jpg',cv2.IMREAD_GRAYSCALE)

plt.imshow(img,cmap='gray')
plt.show()

gau_img = gaussian_blur(img,1)
canny_img = canny(gau_img,30,75)


fig = plt.figure()
plt.subplot(121)
plt.imshow(img,cmap='gray')
plt.xlabel('original')

plt.subplot(122)
plt.imshow(canny_img,cmap='gray')
plt.xlabel('original')
fig.tight_layout()
plt.show()

def SPTA(img):
    '''
    n5 n6 n7
    n4 p  n0
    n3 n2 n1
    :param img:
    :return:
    '''
    img = img>0
    n4 = np.pad(img[:, :-1], ((0, 0), (1, 0)))
    n0 = np.pad(img[:, 1:], ((0, 0), (0, 1)))
    n6 = np.pad(img[:-1, :], ((1, 0), (0, 0)))
    n2 = np.pad(img[1:, :], ((0, 1), (0, 0)))
    n1 = np.pad(n2[:, 1:], ((0, 0), (0, 1)))
    n7 = np.pad(n6[:, 1:], ((0, 0), (0, 1)))
    n3 = np.pad(n2[:, :-1], ((0, 0), (1, 0)))
    n5 = np.pad(n6[:, :-1], ((0, 0), (1, 0)))
    n0_logic = np.logical_not(n0) & (n4 & (n5 + n6 + n2 + n3) & (n6 + np.logical_not(n7)) & (n2 + np.logical_not(n1)))
    n4_logic = np.logical_not(n4) & (n0 & (n1 + n2 + n6 + n7) & (n2 + np.logical_not(n3)) & (n6 + np.logical_not(n5)))
    n2_logic = np.logical_not(n2) & (n6 & (n7 + n0 + n4 + n5) & (n0 + np.logical_not(n1)) & (n4 + np.logical_not(n3)))
    n6_logic = np.logical_not(n6) & (n2 & (n3 + n4 + n0 + n1) & (n4 + np.logical_not(n5)) & (n0 + np.logical_not(n7)))
    logical = n0_logic+n4_logic+n2_logic+n6_logic
    img[0,:] = img[:,0]=img[-1,:]=img[:,-1]=0
    return np.uint8(np.where(logical,False,img))*255


def front_pixel(y,x,dir):
    if dir%2==0:
        front=np.zeros((3,3),np.int32)
    else:
        front = np.zeros((5,3),np.int32)

    if dir==0:
        front[0]=[y-1,x+1,7]
        front[1]=[y,x+1,0]
        front[2] =[y+1,x+1,1]
    elif dir==1:
        front[0]=[y-1,x+1,7]
        front[1] =[y,x+1,0]
        front[2] = [y + 1, x + 1, 1]
        front[3] = [y + 1, x, 2]
        front[4] = [y + 1, x-1, 3]
    elif dir==2:
        front[0] = [y + 1, x + 1, 1]
        front[1] = [y + 1, x, 2]
        front[2] = [y + 1, x-1, 3]
    elif dir==3:
        front[0] = [y + 1, x + 1, 1]
        front[1] = [y + 1, x, 2]
        front[2] = [y + 1, x - 1, 3]
        front[3] = [y,x-1,4]
        front[4] = [y-1,x-1,5]
    elif dir==4:
        front[0] = [y + 1, x - 1, 3]
        front[1] = [y, x - 1, 4]
        front[2] = [y - 1, x - 1, 5]
    elif dir==5:
        front[0] = [y + 1, x - 1, 3]
        front[1] = [y, x - 1, 4]
        front[2] = [y - 1, x - 1, 5]
        front[3] = [y-1,x,6]
        front[4] = [y-1,x+1,7]
    elif dir == 6:
        front[0] = [y - 1, x - 1, 5]
        front[1] = [y - 1, x, 6]
        front[2] = [y - 1, x + 1, 7]
    elif dir == 7:
        front[0] = [y - 1, x - 1, 5]
        front[1] = [y - 1, x, 6]
        front[2] = [y - 1, x + 1, 7]
        front[3]=[y,x+1,0]
        front[4] =[y+1,x+1,1]
    return front


def edge_segment_detection(spta_img):
    dx = np.array([0, 1, 1, 1, 0, -1, -1, -1])
    dy = np.array([1, 1, 0, -1, -1, -1, 0, 1])
    spta_img[0, :] = spta_img[:, 0] = spta_img[-1, :] = spta_img[:, -1] = 0
    spta_img //=255
    c = np.zeros_like(spta_img)
    for i in range(1, len(spta_img) - 1):
        for j in range(1, len(spta_img[0]) - 1):
            if spta_img[i,j]:
                flag = False
                count = 0
                for z in range(9):
                    idx = z
                    if z == 8:
                        idx = 0
                    y = i + dy[idx]
                    x = j + dx[idx]
                    if flag and spta_img[y, x] == 0:
                        count += 1
                        flag=False
                    elif spta_img[y, x]:
                        flag = True
                    else:
                        flag=False

                c[i, j] = count

    zeros = np.zeros([spta_img.shape[0], spta_img.shape[1], 3], np.uint8)

    # start and branch
    zeros[..., 0] = spta_img * 255
    zeros[..., 1] = spta_img * 255
    zeros[..., 2] = spta_img * 255

    zeros[..., 1] = np.where(c == 1, 0, zeros[..., 1])
    zeros[..., 2] = np.where(c == 1, 0, zeros[..., 2])
    zeros[..., 0] = np.where(c >= 3, 0, zeros[..., 0])
    zeros[..., 1] = np.where(c >= 3, 0, zeros[..., 1])
    plt.imshow(zeros)
    plt.xlabel("start")
    plt.show()

    Q = queue.Queue()
    for i in range(1, len(spta_img) - 1):
        for j in range(1, len(spta_img[0]) - 1):
            if c[i, j] == 1 or c[i, j] >= 3:
                for z in range(8):
                    y = dy[z] + i
                    x = dx[z] + j
                    if spta_img[y, x]:
                        Q.put((i, j, z))

    n = 0
    visited = np.zeros_like(spta_img)
    segment = []
    while not Q.empty():
        (y, x, dir) = Q.get()
        cy = dy[dir] + y
        cx = dx[dir] + x

        if visited[cy, cx]:
            continue
        n += 1
        n_seg = []
        n_seg.append([y, x])
        n_seg.append([cy, cx])
        visited[y, x] = visited[cy, cx] = 1

        if c[cy, cx] == 1 or c[cy, cx] >= 3:
            continue
        while True:
            flag = False
            fronts = front_pixel(cy, cx, dir)
            for f in fronts:
                n_y, n_x, n_dir = f
                if c[n_y, n_x] == 1 or c[n_y, n_x] >= 3:
                    flag = True
                    visited[n_y, n_x] = 1
                    n_seg.append([n_y, n_x])
                    break
            if flag:
                break
            else:
                for f in fronts:
                    n_y, n_x, n_dir = f
                    if spta_img[n_y, n_x]:
                        n_seg.append([n_y, n_x])
                        visited[n_y, n_x] = 1
                        dir = n_dir
                        cy = n_y
                        cx = n_x
                        flag = True
                        break
                if flag == False:
                    break

        segment.append(n_seg)
    return segment






spta_img = SPTA(canny_img)

small = spta_img
segmented = edge_segment_detection(small)

# for s in segmented:
#     if len(s)>10:
#         edge = np.zeros([small.shape[0],small.shape[1]],np.uint8)
#         fig = plt.figure(figsize=(13,13))
#         for i in s:
#             y,x = i
#             edge[y,x]=255
#         plt.imshow(edge,cmap='gray')
#         plt.show()

#
# fig = plt.figure(figsize=(13,13))
# plt.subplot(121)
# plt.imshow(canny_img,cmap='gray')
# plt.xlabel('canny')
#
# plt.subplot(122)
# plt.imshow(spta_img,cmap='gray')
# plt.xlabel('spta_img')
# fig.tight_layout()
# plt.show()
#
fig = plt.figure(figsize=(13,13))
plt.subplot(121)
plt.imshow(canny_img[250:350,200:300],cmap='gray')
plt.xlabel('canny')

plt.subplot(122)
plt.imshow(spta_img[250:350,200:300],cmap='gray')
plt.xlabel('spta_img')
fig.tight_layout()
plt.show()