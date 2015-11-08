﻿import cv2
import numpy as np
from matplotlib import pyplot as plt


def nothing(x):
    pass

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def find_squares(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in cv2.split(img):
        for thrs in xrange(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            bin, contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
    return squares

## Create a black image, a window
def Read_Image():
    img = cv2.imread('Dice1.jpg',cv2.IMREAD_GRAYSCALE)
    return img

# Equalizer
def Equalizer(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    equ = cv2.equalizeHist(img)
    #plt.subplot(221), plt.imshow(img, cmap = 'gray')
    #plt.subplot(222), plt.imshow(equ, cmap = 'gray')
    #plt.subplot(223), plt.imshow(cl1, cmap = 'gray')
    #plt.show()
    return cl1, equ


# Clean Up and Threshold
def Blur(img, kernel = (5,5)):
    blur = cv2.blur(img,kernel)
    #plt.subplot(221), plt.imshow(cl1, cmap = 'gray')
    #plt.subplot(222), plt.imshow(blur, cmap = 'gray')
    #plt.subplot(223), plt.imshow(img, cmap = 'gray')
    #plt.show()
    return blur



def Dilate_Erode(img, dilated_ker = (1,1), erode_ker = (6,6)):
    dilated = cv2.dilate(img, np.ones(dilated_ker))
    erode = cv2.erode(dilated, np.ones(erode_ker))
    #plt.subplot(211), plt.imshow(dilated, cmap = 'gray')
    #plt.subplot(212), plt.imshow(erode, cmap = 'gray')
    #plt.show()
    return erode



def Canny_Method(img):
    # create trackbars for canny change
    cv2.namedWindow('canny', cv2.WINDOW_NORMAL)
    cv2.createTrackbar("p1",'canny',0, 1000, nothing)
    cv2.createTrackbar("p2",'canny',0, 1000, nothing);

    while(1):
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        # get current positions of four trackbars
        p1 = cv2.getTrackbarPos("p1",'canny')
        p2 = cv2.getTrackbarPos("p2",'canny')

        edges = cv2.Canny(img,p1,p2,  apertureSize=5)
        cv2.imshow('canny',edges)

    cv2.destroyAllWindows()
    return edges

if __name__ == '__main__':
    # Read image
    img = Read_Image()
    # Adjust contrast of image
    cl1, equ = Equalizer(img)
    # Blur filter to the image
    blur = Blur(cl1)
    # Dilation and erosion morphological transformation
    erode = Dilate_Erode(blur)      # This can change everything
    # Find squares in the image
    squares = find_squares(erode)
    # Draw Contrours on the original image
    cv2.drawContours(img, squares, -1, (0, 255, 0), 3 )
    plt.imshow(img, 'gray')
    plt.show()