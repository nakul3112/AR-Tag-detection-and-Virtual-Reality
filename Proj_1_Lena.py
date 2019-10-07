# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 22:40:09 2019

@author: nakul
"""


import numpy as np
from numpy import linalg as LA
import os, sys
from numpy import linalg as la
import math
# from PIL import Image
import random

try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2
#==============================================================================
# try:
#     #sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
# except:
#     pass

#-------------------------------------------------------------------------------





def find_homography(source,dest):
    camera1 = dest[0]
    camera2 = dest[1]
    camera3 = dest[2]
    camera4 = dest[3]

    world1 = source[0]
    world2 = source[1]
    world3 = source[2]
    world4 = source[3]

    A=np.array([[world1[0],world1[1],1,0,0,0,-camera1[0]*world1[0],-camera1[0]*world1[1],-camera1[0]],
                [0,0,0,world1[0], world1[1],1,-camera1[1]*world1[0],-camera1[1]*world1[1],-camera1[1]],
                [world2[0],world2[1],1,0,0,0,-camera2[0]*world2[0],-camera2[0]*world2[1],-camera2[0]],
                [0,0,0,world2[0], world2[1],1,-camera2[1]*world2[0],-camera2[1]*world2[1],-camera2[1]],
                [world3[0],world3[1],1,0,0,0,-camera3[0]*world3[0],-camera3[0]*world3[1],-camera3[0]],
                [0,0,0,world3[0], world3[1],1,-camera3[1]*world3[0],-camera3[1]*world3[1],-camera3[1]],
                [world4[0],world4[1],1,0,0,0,-camera4[0]*world4[0],-camera4[0]*world4[1],-camera4[0]],
                [0,0,0,world4[0], world4[1],1,-camera4[1]*world4[0],-camera4[1]*world4[1],-camera4[1]]])

    #Performing SVD
    u, s, vtrans = la.svd(A)

            # normalizing by last element of v
            #v =np.transpose(v_col)
    v = vtrans[8:,]/vtrans[8][8]

    hom_matrix = np.reshape(v,(3,3))

    return hom_matrix

def impose_lena(ctr,image,source):
    
    pts_dst = np.concatenate(ctr)
    print(pts_dst.shape)
    pts_source = np.array([[0,0],[511, 0],[511, 511],[0,511]],dtype=float)
    
    h  = find_homography(pts_source, pts_dst)
    print(h)

    print(image.shape[1],image.shape[0])

    temp = cv2.warpPerspective(source, h,(image.shape[1],image.shape[0]));
    cv2.fillConvexPoly(image, pts_dst.astype(int), 0, 16);

    image = image + temp;

    return image,h

def conv_to_bin(A):
    for i in range(0,len(A)):
        for j in range(0,len(A[0])):
            if (A[i,j]>150):
                A[i,j]=1
            else:
                A[i,j]=0
    return A
def find_tag_id(ctr,tag_image):
    gray = cv2.cvtColor(tag_image,cv2.COLOR_BGR2GRAY)
    pixel_value=conv_to_bin(gray)
    status=0
    A_ctr=ctr[0][0]
    print(A_ctr,'ctr A')
    B_ctr=ctr[0][1]
    print(B_ctr,'ctr B')
    C_ctr=ctr[0][2]
    print(C_ctr,'ctr B')
    D_ctr=ctr[0][3]
    print(D_ctr,'ctr C')
    if (pixel_value[2,2] == 1):
        L1=A_ctr
        L2=B_ctr
        L3=C_ctr
        L4=D_ctr
        status=0
        one = pixel_value[4,4]
        two = pixel_value[4,3]
        three = pixel_value[3,3]
        four = pixel_value[3,4]

    elif pixel_value[5,2]==1:
        L1=D_ctr
        L2=A_ctr
        L3=B_ctr
        L4=C_ctr
        status=1
        one = pixel_value[3,4]
        two = pixel_value[4,4]
        three = pixel_value[4,3]
        four = pixel_value[3,3]

    elif pixel_value[5,5] == 1:
        L1=C_ctr
        L2=D_ctr
        L3=A_ctr
        L4=B_ctr
        status=2
        one = pixel_value[3,3]
        two = pixel_value[3,4]
        three = pixel_value[4,4]
        four = pixel_value[4,3]

    elif pixel_value[2,5] == 1:
        L1=B_ctr
        L2=C_ctr
        L3=D_ctr
        L4=A_ctr
        status=3
        one = pixel_value[4,3]
        two = pixel_value[3,3]
        three = pixel_value[3,4]
        four = pixel_value[4,4]

    else:
        L1=A_ctr
        L2=B_ctr
        L3=C_ctr
        L4=D_ctr
        one = pixel_value[4,4]
        two = pixel_value[4,3]
        three = pixel_value[3,3]
        four = pixel_value[3,4]


    new_ctr=np.array([[L1,L2,L3,L4]])

    print(new_ctr,'new_ctr')

    tag_id = four*8 + three*4 + two*2 + one*1

    return new_ctr,tag_id

def draw(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,175,0),-3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
        # draw top layer in red color
        img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,100),3)
    return img

def find_projection_matrix(homography):
   # homography = homography*(-1)
   # Calling the projective matrix function
   K =np.array([[1406.08415449821,0,0],
       [ 2.20679787308599, 1417.99930662800,0],
       [ 1014.13643417416, 566.347754321696,1]])

   K=K.T
   rot_trans = np.dot(la.inv(K), homography)
   col_1 = rot_trans[:, 0]
   col_2 = rot_trans[:, 1]
   col_3 = rot_trans[:, 2]
   l = math.sqrt(la.norm(col_1, 2) * la.norm(col_2, 2))
   rot_1 = col_1 / l
   rot_2 = col_2 / l
   translation = col_3 / l
   c = rot_1 + rot_2
   p = np.cross(rot_1, rot_2)
   d = np.cross(c, p)
   rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
   rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
   rot_3 = np.cross(rot_1, rot_2)

   projection = np.stack((rot_1, rot_2, rot_3, translation)).T
   return np.dot(K, projection)



# Starts Here
#def Imageprocessor(path,source):
source=cv2.imread('lena.jpeg')
frame = cv2.VideoCapture('Tag0.mp4')
count = 0
success = 1
images=[]

while (success):
    if (count==0):
        success, image = frame.read()

    height,width,layers=image.shape
    size = (width,height)
    print(np.shape(image))
    if (count==0):
        old_points=0



    # Code for edgedetection
    #corners=Edgedetection(image,old_points)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur_image = cv2.medianBlur(gray,3)
    (T, thresh) = cv2.threshold(blur_image, 180, 255, cv2.THRESH_BINARY)
    _,contours,hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ctr=[]
    for j, cnt in zip(hierarchy[0], contours):
        cnt_len = cv2.arcLength(cnt,True)
        cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len,True)
        if cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt) and len(cnt) == 4  :
            cnt=cnt.reshape(-1,2)
            if j[0] == -1 and j[1] == -1 and j[3] != -1:
                ctr.append(cnt)
        print(np.shape(ctr))
        old_ctr=ctr
    corners=ctr

    if(len(corners)==0):
        corners=old_points


    # Code for Showing Upright Tag
    # tag_image=perspective_for_tag(corners,image)
    destination = np.array([
        [0, 0],
        [100, 0],
        [100, 100],
        [0, 100]], dtype = "float32")
    
    H1 = find_homography(corners[0], destination)
    warp1 = cv2.warpPerspective(image.copy(), H1, (100,100))
    warp2=cv2.medianBlur(warp1,3)
    tag_image=cv2.resize(warp2, dsize=None, fx=0.08, fy=0.08)

    # Code for Tag-ID-Detection
    new_points,tag_id=find_tag_id(corners,tag_image)

    # Code for stitching the images
    image,h=impose_lena(new_points,image,source)

    # Code for Projection matrix calc
    req_Proj_matrix=find_projection_matrix(h)
    axis = np.float32([[0,0,0,1],[0,512,0,1],[512,512,0,1],[512,0,0,1],[0,0,-512,1],[0,512,-512,1],[512,512,-512,1],[512,0,-512,1]])
    x_c1= np.matmul(axis,req_Proj_matrix.T)
    print("sdcscd:", axis.shape)
    print("dcdscssdc:", req_Proj_matrix.shape)
    print("cube: \n",x_c1)
    print(type(x_c1))

     # Reshaping the cube matrix:

    d1 = x_c1[0][2]
    d2 = x_c1[1][2]
    d3 = x_c1[2][2]
    d4 = x_c1[3][2]
    d5 = x_c1[4][2]
    d6 = x_c1[5][2]
    d7 = x_c1[6][2]
    d8 = x_c1[7][2]


    o1 = np.divide(x_c1[0],d1)
    o2 = np.divide(x_c1[1],d2)
    o3 = np.divide(x_c1[2],d3)
    o4 = np.divide(x_c1[3],d4)
    o5 = np.divide(x_c1[4],d5)
    o6 = np.divide(x_c1[5],d6)
    o7 = np.divide(x_c1[6],d7)
    o8 = np.divide(x_c1[7],d8)

    x_c1 = np.vstack((o1,o2,o3,o4,o5,o6,o7,o8))

    print("Renewed cube coord:", x_c1)
    cube_points = np.array([[x_c1[0][0], x_c1[0][1]],
                [x_c1[1][0], x_c1[1][1]],
                [x_c1[2][0], x_c1[2][1]],
                [x_c1[3][0], x_c1[3][1]],
                [x_c1[4][0], x_c1[4][1]],
                [x_c1[5][0], x_c1[5][1]],
                [x_c1[6][0], x_c1[6][1]],
                [x_c1[7][0], x_c1[7][1]]])
    print(cube_points)
    # draw(image, cube_points)
#==============================================================================


###############################################3

    old_points=corners
    count += 1
    print(count)
    #cv2.imwrite('%d.jpg' %count,edges)
    images.append(image)
    success, image = frame.read()

#--------------------------------------------------------------
#video file
def video(images,size):
    video=cv2.VideoWriter('video2.avi',cv2.VideoWriter_fourcc(*'DIVX'), 16.0,size)
    #print(np.shape(images))
    for i in range(len(images)):
        video.write(images[i])
    video.release()
#---------------------------------------------------------------
# main
if __name__ == '__main__':

    # Calling the function
    source=cv2.imread('lena.jpeg')
    print(np.size(source))

    Image = images
    video(Image,size)
