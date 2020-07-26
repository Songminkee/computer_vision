import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import cv2
import numpy as np
import matplotlib.pyplot as plt
from util import *
img = cv2.imread('../data/pic6.png',cv2.IMREAD_GRAYSCALE)

canny_img = canny(img,30,70)

plt.imshow(canny_img,cmap='gray')
plt.show()


def cal_distance(a,b,c,xx,yy):
    return abs(a*xx+b*yy+c)/np.sqrt(a**2+b**2)

def make_line(y_idxs,x_idxs):
    x1,x2,y1,y2=(0,0,0,0)
    while x1 ==x2 or y1 ==y2:
        ch_idxs = np.random.choice(len(y_idxs), 2, replace=False)
        y1 = y_idxs[ch_idxs[0]]
        x1 = x_idxs[ch_idxs[0]]
        y2 = y_idxs[ch_idxs[1]]
        x2 = x_idxs[ch_idxs[1]]
    slope = (y2-y1) /(x2-x1)
    c = -(y1-(x1*slope))
    return (-1*slope,1,c) # ax+by+c=0

# ax+y+c<1를 만족하는 직선을 이미지내에 그린다. 완벽하게 0을 만족하지 않는 경우도 있기 때문에 1로 설정했다.
def draw_line(m_slope,m_bias,xx,yy):
    return np.where(abs(xx*m_slope+yy+m_bias)<1,255,0)

def Least_square(idxs):
    '''
    :param idxs: (y_idxs,x_idxs)
    :return: [a,b]^T = (A^T A)^-1 A^T B    where, ax+b=y and A=((x1,1),(x2,1),(x3,1),...), B=((y1),(y2),...)

    '''
    y_idxs,x_idxs = idxs
    x_idxs = np.expand_dims(x_idxs,-1)  # (len,1)
    y_idxs = np.expand_dims(y_idxs,-1)  # (len,1)
    A = np.concatenate([x_idxs,np.ones_like(x_idxs)],-1) # (len,2)
    slope,bias = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.transpose(),A)),A.transpose()),y_idxs)
    return slope*-1,bias*-1


def ransac(img,n=50,d=2,return_image=True):
    '''
    :param img: binary(recommand)
    :param n: number of iteration
    :param d: tolerance
    :param return_image:
        True: you can get a straight line image
        False: you can get Linear equation
    :return:
    '''
    if len(img.shape)==2:
        h,w=img.shape
    else:
        h,w,c=img.shape

    y_idxs,x_idxs= np.where(img)

    x_range = np.arange(w)
    y_range = np.arange(h)
    xx, yy = np.meshgrid(x_range, y_range)

    max_line=[0,0,0,0]

    # search model
    for i in range(n):
        m_slope, one, m_bias = make_line(y_idxs, x_idxs)
        temp_idxs = np.where(np.logical_and(cal_distance(m_slope, one, m_bias, xx, yy) < d, img))
        if len(temp_idxs[0]) < 2:  # 2개 이상이어야 직선의 방정식을 세울 수 있다.
            continue
        if max_line[0] < len(temp_idxs[0]):  # 현재 찾은 직선보다 더 많은 점들을 포함하는 경우 교체한다.
            max_line = [len(temp_idxs[0]), m_slope, m_bias, temp_idxs]

    # optimization line
    optimal_param = Least_square(max_line[-1])

    if return_image:
        original_line = draw_line(max_line[1], max_line[2], xx, yy)
        optimal_line = draw_line(optimal_param[0], optimal_param[1], xx, yy)
        return original_line, optimal_line
    else:
        optimal_line = [max_line[0], optimal_param[0], optimal_param[1], max_line[3]]
        return max_line, optimal_line




def draw_line_in_image(line,img):
    if len(img.shape) is not 2:
        print("Shape of input image must have 2, but this input image have shape of {}".format(len(img.shape)))
        return None
    r = np.expand_dims(np.clip(line+img,0,255),-1)
    g = np.expand_dims(np.where(line,0,img),-1)
    b = g
    ret = np.concatenate([r,g,b],-1)
    plt.imshow(ret)




ori_line,optimal_line = ransac(canny_img)
draw_line_in_image(ori_line,canny_img)
draw_line_in_image(optimal_line,canny_img)

plt.show()