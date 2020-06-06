import numpy as np
import matplotlib.pyplot as plt

test_im = np.zeros((8,8))
test_im[1,6]=255
test_im[2,4]=255
test_im[4,1]=255
plt.imshow(test_im,cmap='gray')
plt.show()


def cal_D_max(img):
    return np.sqrt((len(img)-1)**2 + (len(img[0])-1)**2)

theta_min_max=[-90,90]
D = cal_D_max(test_im)
quenti=[int(D),int(D)]
rho_min_max=[D*-1,D]

def hough(canny_img, quenti,theta_range,rho_range,T):
    def cal_rho(theta,y,x):
        theta = theta/180*np.pi
        return y*np.cos(theta)+x*np.sin(theta)
    def cal_y(x,theta,rho):
        theta = theta/180*np.pi
        return (rho-x*np.sin(theta))/np.cos(theta)

    rho_offset = rho_range[0]*-1
    theta_offset = theta_range[0]*-1
    rho_step = (rho_range[1] - rho_range[0]) / quenti[0]
    theta_step = (theta_range[1] - theta_range[0]) / quenti[1]
    start_theta = (theta_range[0]*2+theta_step)/2
    theta_list = np.arange(start_theta,theta_range[1],theta_step)
    start_rho = (rho_range[0]*2+rho_step)/2
    rho_list = np.arange(start_rho, rho_range[1], rho_step)

    A = np.zeros(quenti,np.uint8)
    hough_img=np.zeros([canny_img.shape[0],canny_img.shape[1],3])
    for y in range(len(canny_img)):
        for x in range(len(canny_img[0])):
            if canny_img[y,x]:
                for i in theta_list:
                    rh = cal_rho(i,y,x)
                    if rh>=rho_range[0] and rh<rho_range[1]:
                        n_x =(i+theta_offset)//theta_step
                        n_y = (rh+rho_offset)//rho_step
                        A[int(n_y),int(n_x)]+=1
    A = A>=T

    for rho in range(len(A)):
        for theta in range(len(A[0])):
            if A[rho,theta]:
                for x in range(len(canny_img[0])):
                    y = cal_y(x,theta_list[theta],rho_list[rho])
                    if y>=0 and y<len(canny_img)-0.5:
                        hough_img[int(y+0.5),x,0]=1

    hough_img[..., 1] = np.where(hough_img[...,0]>0,0,canny_img)
    hough_img[..., 2] = np.where(hough_img[...,0] > 0, 0, canny_img)
    hough_img[..., 0] = np.where(hough_img[..., 0] > 0, 255, canny_img)
    return np.uint8(hough_img)

hough_img = hough(test_im,quenti,theta_min_max,rho_min_max,3)
plt.imshow(hough_img,cmap='gray')
plt.show()