import cv2 as cv
import numpy as np
import math as mt
from skimage import feature
import operator
from collections import defaultdict
from skimage.transform import hough_line, hough_line_peaks



frameWidth = 480
frameHeight = 360

# frameWidth = 640
# frameHeight = 480

def FilterLines(input_image,width_kernel_x,width_kernel_y,sigmax,sigmay):
    temp_image = np.float32(input_image)
    temp_image = temp_image*1./255
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

def getQuantile(input_image, qtile):
    number_rows = input_image.shape[0]
    number_columns = input_image.shape[1]

    temp_image = np.reshape(input_image,(1,number_rows*number_columns))
    quantile = getPoints(temp_image,qtile)
    # print(quantile)
    output_image = thresholdlower(input_image,quantile)
    return output_image

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

    # print(index)

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
    minimum = np.amin(input_image)
    thresh = (maximum - minimum)/2
    output_image = cv.threshold(input_image,thresh,1,cv.THRESH_BINARY)

    return output_image[1]

# def getHoughLines(input_image):
#     #Define pho and theta values
#     rMin = 0
#     rMax = 120
#     rStep = 3

#     thetaMin = 80
#     thetaMax = 100
#     thetaStep = 1

#     rBins = int((rMax - rMin)/rStep)
#     thetaBins = int((thetaMax - thetaMin)/thetaStep)

#     print(rBins)
#     print(thetaBins)
#     houghSpace = np.zeros((rBins, thetaBins), np.float32)

#     r = rMin + rStep/2
#     rvalues = []
#     thvalues = []

#     for i in range(0,rBins):
#         rvalues.insert(i,r)
#         r = r + rStep

#     theta  = thetaMin
#     for i in range(0, thetaBins):
#         # thvalues[i] = theta
#         thvalues.insert(i,theta)
#         theta = theta + thetaStep

#     # print(rvalues)
#     # print(thvalues)
#     # print(cv.countNonZero(input_image))
#     nzCount = cv.countNonZero(input_image)
#     xypoints = np.zeros((nzCount,2), np.int32)
#     idx = 0

#     for i in range(0,input_image.shape[1]):
#         for j in range(0,input_image.shape[0]):
#             if(input_image[[j],i] > 0):
#                 xypoints[[idx],0]  = i
#                 xypoints[[idx],1] = j
#                 idx = idx + 1

#     # print(idx)
#     # print(xypoints)

#     for i in range(0,nzCount):
#         for j in range(0,thetaBins):
#             theta = thvalues[j]
#             rval = xypoints[[i],0]*mt.cos(theta) +  xypoints[[i],1]*mt.sin(theta)
#             r = int((rval -rMin)/rStep)

#             if(r>=0 and r<rBins):
#                 houghSpace[[r],j] = houghSpace[[r],j] + 1


#     #Smoth the Hough Space
#     houghSpace = cv.GaussianBlur(houghSpace,(3,3),0.95)



#     # print(houghSpace)
#     # print(houghSpace.shape)
#     # print(houghSpace[39][19])

#     maxLineScores = []
#     maxLineLocs = []
#     #get Local Maxima in Hough Space
#     dictionary  = getlocalMaxima(houghSpace,1)

#     #get the maximum value
#     temp_list = cv.minMaxLoc(houghSpace)
#     maxLineScore = temp_list[1]
#     maxLineLoc = temp_list[3]


#     for i in range(0, len(dictionary)):
#         maxLineScores.append(dictionary[i][0])
#         maxLineLocs.append(dictionary[i][1][0].tolist())

#     stop = False
#     groupThreshold = 15
#     while(!stop):
#         minDIst = groupThreshold + 5
#         dist = 0
#         minlloc = 0
#         minjloc = minlloc + 1
#         iscore = 0
#         for iloc in range(0,len(maxLineLocs)):
#             jloc = iloc + 1
#             for jscore in range(iscore + 1,len(maxLineScores)):
#                 t1 = thvalues[maxLineLocs[iloc][0]]<0 ? thvalues[maxLineLocs[iloc][0]]:thvalues[maxLineLocs[iloc][0]] + np.pi
#                 t2 = thvalues[maxLineLocs[jloc][0]]<0 ? thvalues[maxLineLocs[jloc][0]]:thvalues[maxLineLocs[jloc][0]] + np.pi

#                 distance = mt.fabs(rvalues[maxLineLocs[iloc][1]] - rvalues[maxLineLocs[jloc][1]]) + 0.1* mt.fabs(t1-t2)

#                 if(distance < minDIst):
#                     minDIst = distance
#                     minlloc = iloc
#                     minlscore = iscore
#                     minjloc = jloc
#                     minjscore = jscore

#                 jloc = jloc + 1
#             iscore = iscore + 1


#         if(minDIst >= groupThreshold):
#             stop = True
#         else:
#             x = (maxLineLocs[minlloc][0]*maxLineScores[minlscore]+ maxLineLocs[minjloc][0]*maxLineScores[minjscore])/(maxLineScores[minlscore] + maxLineScores[minjscore])
#             y = (maxLineLocs[minlloc][1]*maxLineScores[minlscore]+ maxLineLocs[minjloc][1]*maxLineScores[minjscore])/(maxLineScores[minlscore] + maxLineScores[minjscore])
#             maxLineLocs[minlloc][0] = int(x)
#             maxLineLocs[minlloc][1] = int(y)
#             maxLineScores[minlscore] = (maxLineScores[minjscore] + maxLineScores[minlscore] )/2






    # print(maxLineLocs)
    # for key,value in dictionary.items():
    #     print(key)
    #     print(value)
    # print(maxLineScores)
    # print(maxLineLocs)
    # print(dictionary)

# def IntersectLineRThetaWithBB(r, theta, bbox):
#     if(cos(theta) ==0):
#         xup = xdown =

def getlocalMaxima(input_matrix,  threhold):
    rows = input_matrix.shape[0]
    columns = input_matrix.shape[1]
    localMaxima = []
    localMaximaLoc = []

    # for i in range(1, rows - 1):
    #     for j in range(1, columns -1):
    #         val = input_matrix[i][j]
    #         if(val > input_matrix[i-1][j-1] and val >  input_matrix[i-1][j]  and val > input_matrix[i-1][j+1] and val >  input_matrix[i][j-1]  and val >  input_matrix[i][j+1] and val >  input_matrix[i+1][j-1] and val >  input_matrix[i+1][j]   and val >  input_matrix[i+1][j+1] ):
    #             print(val)

    localMaximaLoc = feature.peak_local_max(input_matrix,min_distance=3,indices=True)
    # print(localMaximaLoc)
    # print(localMaxima)
    for i,j in localMaximaLoc:
        localMaxima.append(input_matrix[i][j])
    # print(localMaxima)
    data_dict = defaultdict(list)
    for i in range(0,len(localMaxima)):
        data_dict[localMaxima[i]].append(localMaximaLoc[i])
    data_dict = sorted(data_dict.items(),reverse = True)

    return data_dict




img1 = cv.imread("/home/mohak/Downloads/roma/BDXD54/IMG00106.jpg",cv.IMREAD_COLOR)
img1 = cv.resize(img1,(frameWidth,frameHeight))


src = np.array([[0,200],[480,200],[480,360],[0,360]],np.float32)
dst = np.array([[0,150],[480,0],[300,360],[180,400]],np.float32)

M = cv.getPerspectiveTransform(src, dst)
# print(M)
warp = cv.warpPerspective(img1.copy(), M, (480, 360), cv.INTER_CUBIC | cv.WARP_INVERSE_MAP)


#GrayScaling the Image
warp = cv.cvtColor(warp,cv.COLOR_BGR2GRAY)
xScale = 0.01321799535191904
yScale = 0.026703855320215634
#Linewidth in mm
lineWidth = 2000
#Line Pixel  Height
lineHeight = 12*25.4

stopLinePixelWidth = lineWidth*xScale
stopLinePixelHeight = lineWidth*yScale

#Determine sigma in x direction and y direction
# sigmax = stopLinePixelWidth
# sigmay = stopLinePixelHeightd ]
# print(sigmax)
# print(sigmay)


# print(warp.dtype)

output_image = FilterLines(warp, 2,2,2.8,53)
# cv.imshow("Result",output_image)
# cv.waitKey(0)
output_image = getQuantile(output_image,0.985)

kernel = np.ones((5,5), np.float32)
# print(output_image[1])

closed_image = cv.morphologyEx(output_image[1], cv.MORPH_CLOSE,kernel)

#Clean Negetive Parts of the Image
closed_image = thresholdlower(output_image[1],0)

# print(closed_image)
# cv.imshow("Result",closed_image[1])
# cv.waitKey(0)

# stripHeight = closed_image[1].shape[0]

binary_image = getbinaryimage(closed_image[1])
# binary_image = cv.convertScaleAbs(binary_image)

#Scale and Convert binary_image to uint8

# binary_image = (binary_image*255).round().astype(np.uint8)

# getHoughLines(binary_image)

# print(binary_image)
# cv.imshow("Result",binary_image)
# cv.waitKey(0)
# print(binary_image.dtype)
#Get Hough Transform Lines
# print(binary_image.dtype)
# #Change Type
# int_image = np.uint8(binary_image)
# # print(int_image)

# cv.imshow("Result",int_image)
# cv.waitKey(0)
# binary_image = np.uint8(binary_image)


# lines = cv.HoughLinesP(binary_image,3,np.pi/180,15  )
# # print(lines)
# # print(lines[0])
# print(lines)
# a,b,c = lines.shape
# x_values = []
# y_values = []
# # print(lines)
# # x_values = set([])
# # y_values = set([])
# for i in range(a):
# #     cv.line(warp, (lines[i][0][0], lines[i][0][1]),  (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3,8)
#     x_values.append(lines[i][0][0])
#     x_values.append(lines[i][0][2])
#     y_values.append(lines[i][0][1])
#     y_values.append(lines[i][0][3])
#     # x_values.add(lines[i][0][0])
#     # x_values.add(lines[i][0][2])
#     # y_values.add(lines[i][0][1])
#     # y_values.add(lines[i][0][3])

# minimum_y = np.min(y_values)
# index_min_y = np.argmin(y_values)
# x_value_min_y = x_values[index_min_y]
# maximum_y = np.max(y_values)
# index_maximum_y = np.argmax(y_values)
# x_value_max_y = x_values[index_maximum_y]
# cv.line(warp,(x_value_max_y,maximum_y),(x_value_min_y,0),(0,255,0),2)
# # print(x_value_max_y)
# print(x_value_min_y)
# print(index_min_y)
# print(minimum_y)


# # print(lines.shape)
# # print(lines.size)
# # for i in range(0, )
# #Write Code for Hough Transform due to ROI deliminiation
# # print(lines[[1],])
# print(x_values)
# print(y_values)
# lines =  cv.HoughLines(binary_image,3,np.pi/180,30)
# a, b,c = lines.shape
# print(a)
# print(b)
# print(lines[0])
# print(lines.shape)
# print(lines[0][0])
# for rho, theta in lines:
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 480*(-b))
#     y1 = int(y0 + 480*(a))
#     x2 = int(x0 - 480*(-b))
#     y2 = int(y0 - 480*(a))

#     cv.line(warp,(x1,y1),(x2,y2),(0,0,255),2)
# # getHoughLines(binary_image)




# cv.imshow("Result",warp)
# cv.waitKey(0)
# # print(lines)
# print(closed_image.dtype)

# lines = cv.HoughLinesP(closed_image,)
# print(closed_image)
# binary_image = cv.Canny(closed_image,0.05,0.1)

# cv.destroyAllWindows()


