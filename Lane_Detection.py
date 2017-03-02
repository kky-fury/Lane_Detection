import numpy as np
import cv2 as cv
from collections import defaultdict
from skimage.transform import hough_line, hough_line_peaks
import scipy
import math as mt

np.set_printoptions(threshold=np .nan)

# debug = True
debug = False

def FilterLines(input_image, width_kernel_x,width_kernel_y,sigmax,sigmay):

    #Changing Data Type
    temp_image = np.float32(input_image)
    #Scaling
    temp_image = temp_image*1./255

    # if(debug):
    #     cv.imshow("Scaled_Down_Image", temp_image)
    #     cv.waitKey(0)

    #Computer x and y kernels for Filtering Image
    #y - Gaussian Kernel
    #x  - Derivative of Gaussian

    g_kernel_y = []
    variance_y = sigmay*sigmay

    for i in range(-width_kernel_y,width_kernel_y+ 1):
        k1  = np.exp((-0.5/variance_y)*i*i)
        g_kernel_y.append(k1)

    g_kernel_y = np.array(g_kernel_y, dtype = np.float32,ndmin=2)
    g_kernel_y = np.reshape(g_kernel_y,(2*width_kernel_y+1,1))

    variance = sigmax*sigmax
    g_kernel_x = []

    for i in range(-width_kernel_x,width_kernel_x+1 ):
        k1 = np.exp(-i*i*0.5/variance)
        function = (1/variance)*k1 - (i*i)/(variance*variance)*k1
        g_kernel_x.append(function)

    g_kernel_x = np.array(g_kernel_x, dtype = np.float32,ndmin=2)
    kernel = g_kernel_y.dot(g_kernel_x)
    mean = cv.mean(kernel)[0]
    kernel = cv.subtract(kernel,mean)
    temp_image = cv.filter2D(temp_image,-1,kernel)

    return temp_image

#Get Specified quantile value from Input image

def getQuantile(input_image, qtile):
    number_rows = input_image.shape[0]
    number_columns = input_image.shape[1]

    temp_image = np.reshape(input_image,(1,number_rows*number_columns))
    quantile = getPoints(temp_image,qtile)
    # print(quantile)
    output_image = thresholdlower(input_image,quantile)
    return output_image[1]

def getPoints(input_image, quantile):
    # print(input_image.size)
    size = input_image.size
    if (size == 0):
        return float(0)
    elif (size ==1 ):
        return input_image[0]
    elif (quantile<=0):
        return np.amin(input_image)
    elif (quantile>=1):
        return np.amax(input_image)

    pos = (size-1)*quantile
    index = np.uint32(pos)
    delta = pos - index

    temp_image = input_image
    temp_image_1 = np.partition(temp_image,index)
    i1 = temp_image_1[[0],index]
    i2 = np.amin(temp_image_1[[0],index +1:])
    # print(i2)
    return (i1*(1.0 - delta) + i2*delta)

#Threshold Image
def thresholdlower(input_image, threshold):
    # print(input_image)
    output_image = cv.threshold(input_image,threshold,0,cv.THRESH_TOZERO)
    # cv.imshow("Result",output_image[1])
    # cv.waitKey(0)
    return output_image


def getbinaryimage(input_image):
    # Binarize the input  image
    #Calculate Maximum  and minimum of an image
    maximum = np.amax(input_image)
    print(maximum)
    minimum = np.amin(input_image)
    thresh = (maximum - minimum)/2
    output_image = cv.threshold(input_image,thresh,1,cv.THRESH_BINARY)

    return output_image[1]


def getclearImage(thresholded_image):
    img_shape = thresholded_image.shape
    ncolumns = img_shape[1]

    approx = int(ncolumns*0.65)
    img_copy = thresholded_image

    #Make Right End zero
    img_copy[:,approx:] = 0
    img_copy[:,0:(ncolumns-approx)] = 0

    return img_copy

def getHoughLines(input_image):
    hspace, angles, dist = hough_line(binary_image)
    peak_hspace, angles, dist = hough_line_peaks(hspace, angles, dist)


    if(debug):
        for i in range(0, len(angles)):
            a = np.cos(angles[i])
            b = np.sin(angles[i])
            x0 = a*dist[i]
            y0 = b*dist[i]
            x1 = int(x0 + binary_image.shape[0]*(-b))
            y1 = int(y0 + binary_image.shape[0]*(a))
            x2 = int(x0 - binary_image.shape[0]*(-b))
            y2 = int(y0 - binary_image.shape[0]*(a))
        # print(x1)
        # print(y1)
            cv.line(gray_image,(x1,y1),(x2,y2),(0,0,255),1)
        cv.imshow("Result", gray_image)
        cv.waitKey(0)














#Take Input Test Image
input_image = cv.imread("/home/mohak/IPM_test_images/IPM_test_image_1.png")

# if(debug):
#     cv.imshow("Input_Image", input_image)
#     cv.waitKey(0)

#GrayScale Image
gray_image = cv.cvtColor(input_image,cv.COLOR_BGR2GRAY)

# if(debug):
#     cv.imshow("Gray_Image", gray_image)
#     cv.waitKey(0)

#Preprocess Image
filtered_image = FilterLines(gray_image,2,2,2.5,10)

if(debug):
    cv.imshow("Gray_Image", filtered_image)
    cv.waitKey(0)

#Threshold Image
thresholded_image = getQuantile(filtered_image,0.99)

# if(debug):
#     cv.imshow("Gray_Image", thresholded_image)
#     cv.waitKey(0)

#Clean Negetive parts of the Image
thresholded_image = thresholdlower(thresholded_image,0)[1]
# print(thresholded_image.shape)

#Clean Outer  Edges of the Images
clear_image = getclearImage(thresholded_image)

if(debug):
    # cv.imshow("Cleaned_Image", thresholded_image[:,150:])
    cv.imshow("Cleaned_Image", clear_image)
    cv.waitKey(0)

# Binarize the Image
binary_image = getbinaryimage(thresholded_image)

if(debug):
      cv.imshow("Result", binary_image)
      cv.waitKey(0)














