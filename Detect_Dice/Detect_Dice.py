import cv2
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
                if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.contourArea(cnt) < 100000 and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
    return squares

## Create a black image, a window
def Read_Image(img_name):
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    ori = cv2.imread(img_name, cv2.COLOR_BGR2HSV_FULL)
    return img, ori
#
#img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

def Squares_Filter(squares):
    sq = np.ones((len(squares)), dtype=bool)
    filters = []
    for i in range(len(squares)):
        if sq[i] == True:
            filters.append(squares[i])
            sq[i] = False
        for j in range(len(squares)):
            if sq[j] == True:
                diff = Center(squares[i]) - Center(squares[j])
                if abs(diff[0]) < 20 and abs(diff[1]) < 20:
                    sq[j] = False       
                    break       
    return filters

def Center(polygon):
    xc = sum(x for (x, y) in polygon) / len(polygon)
    yc = sum(y for (x, y) in polygon) / len(polygon)
    
    return np.array([xc, yc])

def Clean_Background(img, contours):
    mask = np.zeros(img.shape, np.uint8)
    cv2.drawContours(mask, contours, -1,255,-1 )
    res = cv2.bitwise_and(img,img, mask= mask)
    return res

def Draw_Contours(img, contours):
    cv2.drawContours( img, contours, -1, (0, 255, 0), 3 )

def Detect_Symbols(img):
    ret,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    bin, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    new = []
    new.append(contours[len(contours) - 2])
    Draw_Contours(img, contours[len(contours) - 2])
    return new

def Detect_Circles(res, img, nwsq):
    circles = cv2.HoughCircles(res, cv2.HOUGH_GRADIENT,1,30, param1=50,param2=30,minRadius=0,maxRadius=30)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        if len(circles[0]) == 7:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img,'7',(nwsq[0][0][0], nwsq[0][0][1]), font, 3,(255,255,255),2)
            return img, 1
        if len(circles[0]) == 8:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img,'8',(nwsq[0][0][0], nwsq[0][0][1]), font, 3,(255,255,255),2)
            return img, 1

    circles = cv2.HoughCircles(res, cv2.HOUGH_GRADIENT,1,30, param1=50,param2=30,minRadius=0,maxRadius=50)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        if len(circles[0]) == 1:
            diff = Center(nwsq[0]) - [circles[0][0][0],circles[0][0][1]]
            print diff
            if abs(diff[0]) < 10:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img,'Ace',(nwsq[0][0][0], nwsq[0][0][1]), font, 3,(255,255,255),2)
                return img, 1
    return img, 0

if __name__ == '__main__':
    # Read image
    img, ori = Read_Image('Dice9.jpg')
    # Adjust contrast of image
    cl1, equ = Equalizer(img)
    # Blur filter to the image
    blur = Blur(cl1)
    # Dilation and erosion morphological transformation
    erode = Dilate_Erode(blur)      # This can change everything
    # Find squares in the image
    squares = find_squares(erode)
    squares = Squares_Filter(squares)
    # Squares Loop
    for i in range(len(squares)):
        # Clean background from original image
        nwsq = []
        nwsq.append(squares[i])
        res = Clean_Background(img, nwsq)
        # Detect Features
        Circles, detect = Detect_Circles(res, img, nwsq)   
        if detect == 0:
            new = Detect_Symbols(res)
            res = Clean_Background(img, new)   
            #plt.imshow(res, 'gray')
            #plt.show()
            for letter in ['K', 'J', 'Q']:
                if letter == 'J':
                    imLetter = cv2.imread('J.png', cv2.IMREAD_GRAYSCALE)
                if letter == 'Q':
                    imLetter = cv2.imread('Q.png', cv2.IMREAD_GRAYSCALE)
                if letter == 'K':
                    imLetter = cv2.imread('K.png', cv2.IMREAD_GRAYSCALE)
                matching = cv2.matchShapes(res, imLetter, 1, 0.0)
                print 'matching', matching
                if matching < 0.06:
                    cv2.putText(img,letter,(nwsq[0][0][0], nwsq[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 3,(255,255,255),2)
                    break
    plt.imshow(img, 'gray')
    plt.show()