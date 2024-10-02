# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 10:18:18 2018

@author: user
"""

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler


def show(image):
    cv2.imshow("image", image)
    cv2.waitKey(0)
#%% Parameters

k1 = 5 # kernel size for first bilateral filter (pupil identification)
k2 = 9 # kernel size for first bilateral filter (iris identification)

window = 400 # size of window for subsetting image (120 * 120)

thresh1 = 80 # Binary image thresholding value for pupil
thresh2 = 150 # Binary image thresholding value for iris (155, 170)
scope1 = 1.6 # Lower boundary for iris radius size in respect to pupil radius
scope2 = 3.6 # Upper boundary for iris radius size in respect to iris radius
hough_list = [5,5] #hough variables for hough circles

#%% Projection: horizontal and vertical projection of image.
# Ignore the pixels in boundary of image (width: half of window (30)) for better approximation.

def projection(img):
    (h, w) = img.shape
    h = h-window
    w = w-window
    sumCols = []
    sumRows = []
    lim = int(window/2)
    for i in range(h):
        row = img[i+lim:i+lim+1, 0:w] 
        sumRows.append(np.sum(row))
    for j in range(w):
        col = img[0:h, j+lim:j+lim+1]
        sumCols.append(np.sum(col))
    return sumRows, sumCols
    
#%% Subsetting: getting a subset of image based on center point (posX, posY) and window size.
    
def subsetting(img, posX, posY, window):
    if ((posY<window) and (posX<window)):
        img = img[0:posY+window, 0:posX+window]
    elif ((posY<window) and (posX>=window)):
        img = img[0:posY+window, posX-window:posX+window]
    elif ((posY>=window) and (posX<window)):
        img = img[posY-window:posY+window, 0:posX+window]
    else:
        img = img[posY-window:posY+window, posX-window:posX+window]
    return img

#%% Thresholding: binary image thresholding & getting center of pupil from moments.
    
def thresholding(orig, posX, posY, window, otsu=True):
    img = orig.copy()
    img = subsetting(img, posX, posY, window)
    
    if otsu:
        ret,th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    else:
        ret,th = cv2.threshold(img, thresh2, 255, cv2.THRESH_BINARY_INV)
    
    M = cv2.moments(th)
    
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    
    posY = int(posY+cY-window)
    posX = int(posX+cX-window)
    
    return posX, posY

#%% boundary: see if iris center is inside pupil.
# circle_detect: use houghcircles to find the iris
# circle_detectX: use houghcircles & boundary to find the pupil.
    
def boundary(x1, x2, y1, y2, r):
    if np.sqrt(np.power((y1-x1), 2) + np.power((y2-x2), 2)) < r:
        return True
    else:
        return False
    
def circle_detect(edges, dp = 20, minR = 20, maxR = 0):
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, dp, 
                       param1=hough_list[0], param2=hough_list[1], 
                       minRadius = minR, maxRadius=maxR)
    return circles[0][0]   

def circle_detectX(edges, dp, posX, posY, radius, minR = 20, maxR = 0):
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, dp, 
                       param1=hough_list[0], param2=hough_list[1], 
                       minRadius = minR, maxRadius=maxR)
    circles = [x for x in circles[0] if boundary(x[0], x[1], posX, posY, (radius/2))]
    if len(circles) >= 1:
        return circles[0]  

#%%
def IrisLoc2(orig, name): 
    (h, w) = orig.shape 
    img = orig.copy()
    
    kernel = np.ones((k1,k1),np.float32)/np.power(k1, 2)
    img = cv2.filter2D(img,-1,kernel)
    
    ret,th = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    
    edges = cv2.Canny(th, 0, ret)
    
    circle = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 50, 
                               param1=5, param2=5, 
                               minRadius = 30, maxRadius=90)[0][0]
    img_copy = orig.copy()
    
    try:
        p_posY = int(circle[1])
        p_posX = int(circle[0])
        p_radius = int(circle[2])
    except TypeError:
        print("No pupil")
        p_posY = 0
        p_posX = 0
        p_radius = 0
    
    cv2.circle(img_copy,(p_posX,p_posY),p_radius,(255,255,255),2)
    cv2.circle(img_copy,(p_posX,p_posY),2,(255,255,255),3)   
    
    outer = np.mean([img[x] for x in list(zip(*np.where(th == 255)))])
    pupil = list(zip(*np.where(th == 0)))
    pupil = [(x[0], x[1]) for x in pupil]
        
    img = orig.copy()
    for x in pupil:
        img[x] = outer
    kernel = np.ones((k2,k2),np.float32)/np.power(k2, 2)
    img = cv2.filter2D(img,-1,kernel)
    
    ret2,th2 = cv2.threshold(img, thresh2, 255, cv2.THRESH_BINARY)
    
    edges2 = cv2.Canny(th2, 0, ret2)
                  
    circle2 = circle_detectX(edges2, 10, p_posX, p_posY, p_radius*0.75, 90, 120)
        
    try:
        s_posY = circle2[1]
        s_posX = circle2[0]
        s_radius = int(circle2[2])
    except TypeError:
        print("No schelra")
        s_posY = p_posY
        s_posX = p_posX
        s_radius = p_radius*3
     
    cv2.circle(img_copy,(p_posX,p_posY),p_radius,(255,255,255),2)
    cv2.circle(img_copy,(p_posX,p_posY),2,(255,255,255),3)    
    cv2.circle(img_copy,(s_posX,s_posY),s_radius,(255,255,255),2)
    cv2.circle(img_copy,(s_posX,s_posY),2,(255,255,255),3)    
    #show(img_copy)
    #cv2.imwrite('local/s_' + str(name.split('/')[3]), img_copy)
    #cv2.imwrite('local/s_' + name, img_copy)
    d_names = ["p_posX", "p_posY", "p_radius", "i_posX", "i_posY", "i_radius", "img"]   
    return dict(zip(d_names, [p_posX, p_posY, p_radius, s_posX, s_posY, s_radius, orig]))
    return 

#%% 
def IrisLoc(orig, name):  
    (h, w) = orig.shape        
     
    sumRows, sumCols = projection(orig)
    
    posX = np.argmin(sumCols) + int(window/2)
    posY = np.argmin(sumRows) + int(window/2)
    
    posX, posY = thresholding(orig, posX, posY, window)
    
    posX, posY = thresholding(orig, posX, posY, window)
    
    img = orig.copy()
    img = subsetting(img, posX, posY, window)
    img = orig
    kernel = np.ones((k1,k1),np.float32)/np.power(k1, 2)
    img = cv2.filter2D(img,-1,kernel)
    ret,th = cv2.threshold(img, thresh1, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(th, 0, ret)
    circle = circle_detect(edges, minR=20, maxR=250)
    img_copy = orig.copy()
    
    try:
        p_posY = int(posY+circle[1]-window)
        p_posX = int(posX+circle[0]-window)
        p_radius = int(circle[2])
    except TypeError:
        print("No pupil")
        p_posY = 0
        p_posX = 0
        p_radius = 0
    
    if np.sqrt(np.power(p_posX-(h/2), 2) + np.power(p_posY-(w/2), 2)) >= 80:
        return IrisLoc2(orig, name)
        
    outer = np.mean([img[x] for x in list(zip(*np.where(th == 255)))])
    pupil = list(zip(*np.where(th == 0)))
    pupil = [(posY + x[0] - window, posX + x[1] - window) for x in pupil]
    
    img = orig.copy()
    for x in pupil:
        img[x] = outer

    kernel = np.ones((k2,k2),np.float32)/np.power(k2, 2)
    img = cv2.filter2D(img,-1,kernel)
    
    ret2,th2 = cv2.threshold(img, thresh2, 255, cv2.THRESH_BINARY)
    
    edges2 = cv2.Canny(th2, 0, ret2)
    
    if int(name.split('/')[2]) == 1:
        circle2 = circle_detectX(edges2, 10, p_posX, p_posY, p_radius, 90, 140)
    else:
        circle2 = circle_detectX(edges2, 10, p_posX, p_posY, p_radius, 90, 120)
    
    try:
        s_posY = int(circle2[1])
        s_posX = int(circle2[0])
        s_radius = int(circle2[2])
    except TypeError:
        print("No schelra")
        return IrisLoc2(orig, name)

    cv2.circle(img_copy,(p_posX,p_posY),p_radius,(255,255,255),2)
    cv2.circle(img_copy,(p_posX,p_posY),2,(255,255,255),3)    
    cv2.circle(img_copy,(s_posX,s_posY),s_radius,(255,255,255),2)
    cv2.circle(img_copy,(s_posX,s_posY),2,(255,255,255),3)    

    cv2.imwrite('process/l_' + str(name.split('/')[3]), img_copy)        
    return [p_posX, p_posY, p_radius, s_posX, s_posY, s_radius, orig]





#Normalization

def normalize(row, name):
    image = row['img']
    M = 64
    N = 512
    
    theta = np.linspace(0, 2*np.pi, N)

    diffX = row['p_posX'] - row['i_posX']
    diffY = row['p_posY'] - row['i_posY']

    a = np.ones(N) * (diffX**2 + diffY**2)
    
    if diffX < 0:
        phi = np.arctan(diffY/diffX)
        sgn = -1
    elif diffX > 0:
        phi = np.arctan(diffY/diffX)
        sgn = 1
    else:
        phi = np.pi/2
        if diffY > 0:
            sgn = 1
        else:
            sgn = -1

    b = sgn * np.cos(np.pi - phi - theta)

    r = np.sqrt(a)*b + np.sqrt(a*b**2 - (a - row['i_radius']**2))
    r = np.array([r - row['p_radius']])
    r = np.dot(np.ones([M+2,1]), r) * np.dot(np.ones([N,1]),
                        np.array([np.linspace(0,1,M+2)])).transpose()
    r = r + row['p_radius']
    r = r[1:M+1, :]

    xcosmat = np.dot(np.ones([M,1]), np.array([np.cos(theta)]))
    xsinmat = np.dot(np.ones([M,1]), np.array([np.sin(theta)]))

    x = r*xcosmat
    y = r*xsinmat

    x += row['p_posX']
    x = np.round(x).astype(int)
    x[np.where(x >= image.shape[1])] = image.shape[1] - 1
    x[np.where(x < 0)] = 0
    
    y += row['p_posY']
    y = np.round(y).astype(int)
    y[np.where(y >= image.shape[0])] = image.shape[0] - 1
    y[np.where(y < 0)] = 0
    
    newImg = image[y, x]
    #cv2.imwrite('process/n_' + str(name.split('/')[3]), newImg)    
    
    return {"Image": newImg}

#Enhance

def enhancement(row):
    img = row['Image']
    dim = img.shape
    
    stride = 1 
    initialize_img = np.zeros((int(dim[0]/stride), int(dim[1]/stride)))
    
    for i in range(0,dim[0]-15,stride):
        for j in range(0,dim[1]-15,stride):
            block = img[i:i+stride, j:j+stride]
            m = np.mean(block,dtype=np.float32)
            initialize_img[i//16, j//16] = m
            
    image_set = cv2.resize(initialize_img, (dim[1],dim[0]), interpolation=cv2.INTER_CUBIC)

    enhance = img - image_set
    enhance = enhance - np.amin(enhance.ravel())
    img = enhance.astype(np.uint8)      
         
    img2 = np.zeros(dim)
    for i in range(0,img.shape[0],stride*2):
        for j in range(0,img.shape[1],stride*2):
            img2[i:i+stride*2, j:j+stride*2] = cv2.equalizeHist(img[i:i+stride*2, j:j+stride*2].astype(np.uint8))
    show(img2)
    #cv2.imwrite('process/e_' + str(row.name.split('/')[3]), img2)

    return img2




# match
def match(x_train, y_train, x_test, y_test, reduction, n_comp=120):
    if reduction:
        x_train = StandardScaler().fit_transform(x_train)
        x_test = StandardScaler().fit_transform(x_test)
        pca = PCA(n_components=n_comp).fit(x_train)
        x_train_red = pca.transform(x_train)
        x_test_red = pca.transform(x_test)
        clf = LDA().fit(x_train_red, y_train)
    else:
        x_train_red = x_train
        x_test_red = x_test

    [n1,m1] = x_train_red.shape
    [n2,m2] = x_test_red.shape
    [n, m] = x_train.shape
    
    l = len(np.unique(y_train))
    fi=np.zeros((l,m1))
    
    for i in range(l):
        group = x_train_red[list(np.where(y_train==i+1)),:][0]
        fi[i,:]=(np.mean(group, axis=0))
    
    if reduction:
        x_test_red = clf.transform(x_test_red)
        fi = clf.transform(fi)
        
    d1 = np.zeros((n2,l))
    d2 = np.zeros((n2,l))
    d3 = np.zeros((n2,l))
    
    values_y = np.zeros((n2, 3))
    pred_y = np.zeros((n2, 3))
    for i in range(n2):
        for j in range(l):
            d1[i,j] = sum(abs((x_test_red[i,:]-fi[j,:])))
            d2[i,j] = sum((x_test_red[i,:]-fi[j,:])**2);              
            d3[i,j] = 1-(np.dot(x_test_red[i,:].T, fi[j,:]))/(np.linalg.norm(x_test_red[i,:])*np.linalg.norm(fi[j,:]))
         
        values_y[i, 0] = np.min(d1[i,:])
        values_y[i, 1] = np.min(d2[i,:])
        values_y[i, 2] = np.min(d3[i,:])
        pred_y[i, 0] = np.argmin(d1[i,:])+1
        pred_y[i, 1] = np.argmin(d2[i,:])+1
        pred_y[i, 2] = np.argmin(d3[i,:])+1
        
    return values_y, pred_y

        
