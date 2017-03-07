import numpy as np
import cv2 as cv
from skimage import feature
from collections import defaultdict
from skimage.transform import hough_line, hough_line_peaks
import scipy
import math as mt
import operator
from skimage import io

np.set_printoptions(threshold=np .nan)

debug = True
# debug = False

class Line:
    startpoint = []
    endpoint = []
    linescore = 0
    list_x_y_points = []
    def __init__(self, startpoint, endpoint, linescore):

        self.startpoint = startpoint
        self.endpoint = endpoint
        self.linescore = linescore

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

def getRansacLines(thresholded_image, lines):

    test_image = np.transpose(thresholded_image)
    non_zero_points = np.nonzero(test_image)
    list_x_y_points = []
    for i in range(0, len(non_zero_points[0])):
        point = (non_zero_points[0][i], non_zero_points[1][i])
        list_x_y_points.append(point)

    sorted_by_second = sorted(list_x_y_points, key=lambda tup: tup[1])
    # print(sorted_by_second)
    intializepointsinROI(sorted_by_second, lines)

    for i in range(0, len(lines)):
        # print(lines[i].list_x_y_points)
        data = np.array(lines[i].list_x_y_points)
        line = cv.fitLine(data,cv.DIST_FAIR,0,0.01,0.01)
        mult = max(gray_image.shape[0], gray_image.shape[1])
        startpoint = (int(line[2] - mult*line[0]), int(line[3] - mult*line[1]))
        endpoint = (int(line[2]  + mult*line[0]), int(line[3] + mult*line[1]))
        points = cv.clipLine((0,0,gray_image.shape[1],gray_image.shape[0]), startpoint, endpoint)
        x_limit_max = max(lines[i].startpoint[0], lines[i].endpoint[0])
        x_limit_min = min(lines[i].startpoint[0], lines[i].endpoint[0])
        points = [list(i) for i in points[1:]]
        # print(points[][0])
        for i in range(0, len(points)):
            if(points[i][0] < x_limit_min ):
                points[i][0] = x_limit_min
            elif(points[i][0] > x_limit_max):
                points[i][0] = x_limit_max
        # print(points)
        cv.line(gray_image, tuple(points[0]), tuple(points[1]),(0, 0, 255),2)

    # # print(line)
    # cv.imshow("Result", gray_image)
    # cv.waitKey(0)
    #Write Image
    cv.imwrite("/home/mohak/Lane_Detection_Result/image_8.png", gray_image)


def intializepointsinROI(x_y_points, lines):
    # for i in range(0, len(lines)):
    #     print(lines[i].startpoint)
    #     print(lines[i].endpoint)

    # threshold = x_limit_max
    for i in range(0, len(lines)):
        # print(lines[i].startpoint)
        # print(lines[i].endpoint)
        x_limit_max = max(lines[i].startpoint[0], lines[i].endpoint[0])
        x_limit_min = min(lines[i].startpoint[0], lines[i].endpoint[0])
        # threshold = x_limit_max - x_limit_min
        # print(threshold)
        search_range = np.arange(x_limit_min-1, x_limit_max+1)
        # print(search_rangenge)
        points = [x for x in x_y_points if  x[0] in search_range]
        lines[i].list_x_y_points = points
        # print(lines[i].list_x_y_points)


#The fucntion gets the specified quantile value
#from the input image
#Param:-
#input_image - Filtered_Image
#qtile - Quantile Threshold
#

def getQuantile(input_image, qtile):
    number_rows = input_image.shape[0]
    number_columns = input_image.shape[1]

    temp_image = np.reshape(input_image,(1,number_rows*number_columns))
    quantile = getPoints(temp_image,qtile)
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
    # print(maximum)
    minimum = np.amin(input_image)
    thresh = (maximum - minimum)/2
    output_image = cv.threshold(input_image,thresh,1,cv.THRESH_BINARY)

    return output_image[1]


def getclearImage(thresholded_image):
    img_shape = thresholded_image.shape
    ncolumns = img_shape[1]
    nrows = img_shape[0]

    approx_1 = int(nrows*0.75)
    approx = int(ncolumns*0.75)
    img_copy = thresholded_image

    #Make Right End zero
    img_copy[:,approx:] = 0
    img_copy[:,0:(ncolumns-approx)] = 0
    img_copy[0:(nrows-approx_1),:] = 0

    return img_copy

def getHoughLines(input_image):
    # cv.imshow("Intermediate",input_image)
    # cv.waitKey(0)
    hspace, angles, dist = hough_line(binary_image)
    maximum = np.amax(hspace)
    # print(maximum)
    peak_hspace, angles, dist = hough_line_peaks(hspace, angles,dist, threshold = maximum*0.38,min_distance = 20)
    # print(peak_hspace)

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
        io.imsave("/home/mohak/Process_Pipeline/Initial_Guess_For_Ransac.png",gray_image)
        cv.imshow("Result", gray_image)
        cv.waitKey(0)

    # print(peak_hspace)
    lines = groupLines(angles,dist,peak_hspace)
    # print(lines)
    lines = checklanewidth(lines)
    # print(lines)
    getRansacLines(input_image,lines)
    return hspace, angles, dist



def checklanewidth(lines):
    number_of_lanes = len(lines)
    #Sort Line Objects According to Starting Point
    lines.sort(key = lambda x:x.startpoint[0])
    # print(lines)

    min_distance = 10 #approx
    max_distance_two_side_lanes = 45 #approx
    max_distance_two_edge_lanes = 70 #approx
    x_points = []
    for i in range(0, number_of_lanes):
        x_max = max(lines[i].startpoint[0], lines[i].endpoint[0])
        x_points.append(x_max)

    x_points.sort()
    # print(x_points)

    if (number_of_lanes == 2):
        #Calculate Diff array
        diff_array = np.diff(x_points)
        for i in range(0, len(diff_array)):
            if(diff_array[i] < min_distance or diff_array[i] > max_distance_two_edge_lanes):
                lines.pop(i+1)
    elif(number_of_lanes == 3):
        diff_array = np.diff(x_points)
        for i in range(0, len(diff_array)):
            if (diff_array[i] < min_distance or diff_array[i] > max_distance_two_side_lanes):
                lines.pop(i+1)


    return lines




def getlocalMaxima(input_matrix,  threhold):
    rows = input_matrix.shape[0]
    columns = input_matrix.shape[1]
    localMaxima = []
    localMaximaLoc = []

    localMaximaLoc = feature.peak_local_max(input_matrix,min_distance=3,indices=True)
    for i,j in localMaximaLoc:
        localMaxima.append(input_matrix[i][j])
    data_dict = defaultdict(list)

    for i in range(0,len(localMaxima)):
        data_dict[localMaxima[i]].append(localMaximaLoc[i])
    data_dict = sorted(data_dict.items(),reverse = True)
    return data_dict


def getLineEndPoints(r, theta,img_size):
    startpoint = []
    endpoint = []
    if(mt.cos(theta) == 0):
        xup = int(img_size[0])
        xdown = int(img_size[0])
    else:
        xup = int(r/mt.cos(theta))
        xdown = int((r-img_size[1]*mt.sin(theta))/mt.cos(theta))

    if(mt.sin(theta ) ==0):
        yleft = int(img_size[1])
        yright = int(img_size[1])
    else:
        yleft = int(r/mt.sin(theta))
        yright = int((r-img_size[0]*mt.cos(theta))/mt.sin(theta))

    pts = [(xup,0),(xdown,img_size[1]),(0,yleft),(img_size[0],yright)]
    count = 0
    for i in range(0,4):
        if(isPointInside(pts[i],img_size)):
            startpoint.append((pts[i][0],pts[i][1]))
            count = i
            break
    # print(count )
    for i in range(count+1,4):
            if(isPointInside(pts[i],img_size)):
                endpoint.append((pts[i][0],pts[i][1]))
                break

    return startpoint, endpoint


def isPointInside(point, img_size):
    if(point[0] >= 0  and point[0] <= img_size[0]  and point[1] >= 0 and point[1] <= img_size[1]):
        return True
    else:
        return False





def groupLines(angles, dist, peak_hspace):

    startpoints = []
    endpoints = []
    for i in range(0, len(angles)):
        startpoint, endpoint = getLineEndPoints(dist[i], angles[i], (binary_image.shape[1],binary_image.shape[0]))
        startpoints.append(startpoint)
        endpoints.append(endpoint)

    number_of_lines = len(dist)
    lines = []
    # print(startpoints[0][0])
    for i in range(0, number_of_lines):
        line = Line(startpoints[i][0],endpoints[i][0], peak_hspace[0])
        lines.append(line)
    # print(lines)
    return lines








#Take Input Test Image
input_image = cv.imread("/home/mohak/IPM_test_images/IPM_test_image_1.png")

if(debug):
    cv.imshow("Input_Image", input_image)
    cv.waitKey(0)

#GrayScale Image
gray_image = cv.cvtColor(input_image,cv.COLOR_BGR2GRAY)

if(debug):
    cv.imshow("Gray_Image", gray_image)
    cv.waitKey(0)

#Preprocess Image
filtered_image = FilterLines(gray_image,2,2,2,10)

if(debug):
    cv.imshow("Gray_Image", filtered_image)
    cv.waitKey(0)
    io.imsave("/home/mohak/Process_Pipeline/filtered_image.png",filtered_image)

#Threshold Image
thresholded_image = getQuantile(filtered_image,0.985)

if(debug):
    io.imsave("/home/mohak/Process_Pipeline/thresholded_image.png",thresholded_image)
    cv.imshow("Gray_Image", thresholded_image)
    cv.waitKey(0)

# Clean Negetive parts of the Image
thresholded_image = thresholdlower(thresholded_image,0)[1]
# print(thresholded_image.shape)

#Clean Outer  Edges of the Images
clear_image = getclearImage(thresholded_image)

if(debug):
    cv.imshow("Cleaned_Image", thresholded_image[:,150:])
    cv.imshow("Cleaned_Image", clear_image)
    cv.waitKey(0)

# Binarize the Image
binary_image = getbinaryimage(thresholded_image)

if(debug):
      cv.imshow("Result", binary_image)
      cv.waitKey(0)

#Select ROI
image_height = binary_image.shape[0]
image_width = binary_image.shape[1]

ROI_height = int(0.45*image_height)
ROI_image = binary_image[ROI_height:,:]

if(debug):
      io.imsave("/home/mohak/Process_Pipeline/binary_image_after_ROI.png", ROI_image)
      cv.imshow("Result",  ROI_image)
      cv.waitKey(0)
hspace, angles, dist = getHoughLines(ROI_image)










