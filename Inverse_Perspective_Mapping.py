#Inverse Perspective Transform Inspired from  Caltech_LD

import cv2 as cv
import numpy as np
import math as mt

class IPMInfo:
    #min and max x-value on ground in world coordinates
    xLimits = []
    #min and max y-value on ground in world coordinates
    yLimits = []
    #conversion between mm in world coordinate on the ground in x-direction and pixel in image
    xScale = 0.0
    #conversion between mm in world coordinate on the ground in y-direction and pixel in image
    yScale = 0.0

    width = 0.0
    height = 0.0
    #portion of image height to add to y-coordinate of vanishing point
    vpPortion = 0.0
    #Left point in original image of region to make IPM for
    ipmLeft = 0.0
    #Right point in original image of region to make IPM for
    ipmRight = 0.0
    #Top point in original image of region to make IPM for
    ipmTop = 0.0
    #Bottom point in original image of region to make IPM for
    ipmBottom = 0.0
    #nterpolation to use for IPM (0: bilinear, 1:nearest neighbor)
    ipmInterpolation = 0

    def __init__(self,width, height, vpPortion, ipmLeft, ipmRight, ipmTop, ipmBottom, ipmInterpolation ):
        # self.xLimits = xLimits
        # self.yLimits = yLimits
        # self.xScale = xScale
        # self.yScale = yScale
        self.width = width
        self.height = height
        self.vpPortion = vpPortion
        self.ipmLeft = ipmLeft
        self.ipmRight = ipmRight
        self.ipmTop = ipmTopc
        self.ipmBottom = ipmBottom
        self.ipmInterpolation = ipmInterpolation
    @classmethod
    def intialize_limits(self, xLimits, yLimits, xScale, yScale):
        self.xLimits = xLimits
        self.yLimits = yLimits
        self.xScale = xScale
        self.yScale = yScale



class CameraInfo:
    focalLength = []
    opticalCenter = []
    cameraHeight = 0.0
    pitch = 0.0
    yaw = 0.0
    imageWidth = 0
    imageHeight = 0

    #Initialize Object
    def __init__(self, focalLength, opticalCenter, cameraHeight, pitch, yaw, imageWidth, imageHeight):
        self.focalLength = focalLength
        self.opticalCenter = opticalCenter
        self.cameraHeight = cameraHeight
        self.pitch = mt.radians(pitch)
        self.yaw = mt.radians(yaw)
        self.imagewidth = imageWidth
        self.imageHeight = imageHeight

def getVanishingPoint(cinfo):
    # print(cinfo.pitch)
    #Get  vanishing point in world coordiantes
    vp = np.array([[mt.sin(cinfo.yaw)/mt.cos(cinfo.pitch)],[mt.cos(cinfo.yaw)/mt.cos(cinfo.pitch)],[0]],dtype = np.float32)
    # print(vp)
    # print(vp.shape)
    #Transform from World to camera coordiantes
    #Rotation matrix for yaw
    tyaw = np.array([[mt.cos(cinfo.yaw),-mt.sin(cinfo.yaw),0],[mt.sin(cinfo.yaw),mt.cos(cinfo.yaw),0],[0,0,1]],dtype = np.float32)
    # print(tyaw)

    #Rotation Matrix for pitch
    tpitch = np.array([[1,0,0],[0,-mt.sin(cinfo.pitch),-mt.cos(cinfo.pitch)],[0,mt.cos(cinfo.pitch),-mt.sin(cinfo.pitch)]],dtype = np.float32)
    # print(tpitch)

    transform = tyaw.dot(tpitch)
    # print(transform)
    #Transformation from (xc,yc) in camera coordinates to (u,v) in image frame

    #Matrix to shift optical center and focal length
    shift_mat = np.array([[cinfo.focalLength[0],0,cinfo.opticalCenter[0]],[0, cinfo.focalLength[1],cinfo.opticalCenter[1]],[0,0,1]],dtype = np.float32)
    # print(shift_mat)
    transform = shift_mat.dot(transform)
    # print(transform)

    vp = transform.dot(vp)
    # print(vp)
    coordinates = [vp[0],vp[1]]
    return coordinates

def TransformImage2Ground(inpoints, cinfo):
    #add two roes to the iinput points
    # print(inpoints)
    rows = inpoints.shape[0]
    columns = inpoints.shape[1]
    zero_row = np.zeros((1,columns),np.float32)
    count = 0;
    while(count <2):
        inpoints = np.vstack((inpoints,zero_row))
        count = count + 1

    # print(inpoints)
    inpoints2 = inpoints[[0,1],:]
    inpoints3 = inpoints[[0,1,2],:]

    inpointsr3 = inpoints[[2],:]
    inpointsr4 = inpoints[[3],:]

    inpoints[[2],:] = 1
    # print(inpoints)
    inpoints2 = inpoints
    inpoints3 = inpoints[[0,1,2],:].astype(np.float32,copy=False)
    inpoints2 = inpoints2.astype(np.float32,copy=False)
    # # print(inpoints2)
    # print(inpoints3)
    # #Create Transformation Matrix
    c1 = mt.cos(cinfo.pitch)
    s1 = mt.sin(cinfo.pitch)
    c2 = mt.cos(cinfo.yaw)
    s2 = mt.sin(cinfo.yaw)

    matp =  np.array([[-cinfo.cameraHeight*c2/cinfo.focalLength[0],cinfo.cameraHeight*s1*s2/cinfo.focalLength[1],
                     (cinfo.cameraHeight*c2*cinfo.opticalCenter[0]/cinfo.focalLength[0]) -
                     (cinfo.cameraHeight*s1*s2*cinfo.opticalCenter[1]/cinfo.focalLength[1]) - cinfo.cameraHeight*c1*s2],


                     [cinfo.cameraHeight*s2/cinfo.focalLength[0], cinfo.cameraHeight*s1*c2/cinfo.focalLength[1],
                     (-cinfo.cameraHeight*s2*cinfo.opticalCenter[0]/cinfo.focalLength[0]) - (cinfo.cameraHeight*s1*c2*cinfo.opticalCenter[1]/cinfo.focalLength[1])
                     - cinfo.cameraHeight*c1*c2],

                     [0, cinfo.cameraHeight*c1/cinfo.focalLength[1],
                     (-cinfo.cameraHeight*c1*cinfo.opticalCenter[1]/cinfo.focalLength[1]) + cinfo.cameraHeight*s1],

                     [0,-c1/cinfo.focalLength[1],(c1*cinfo.opticalCenter[1]/cinfo.focalLength[1]) - s1]],dtype=np.float32)

    # print(matp)

    # print(inpoints3)
    # print(matp.shape)
    # print(inpoints3.shape)
    inpoints4 = matp.dot(inpoints3).astype(np.float32,copy=False)
    # print(inpoints4.astype(np.float32,copy=False))
    # print(inpoints4)
    # # inpoints4 = cv.convertPointsFromHomogeneous(inpoints4)
    # # print(inpoints4)
    inpointsr4 = inpoints4[[3],:]
    # # print(inpointsr4[0][0])
    # # print(inpoints4)
    # print(inpoints4)
    for i in range(0,columns):
        div = inpointsr4[0][i]
        # print(div)
        inpoints4[[0],i] = inpoints4[[0],i]/div
        inpoints4[[1],i] = inpoints4[[1],i]/div
        # i = i+1
    # print(inpoints4)
    inpoints2 = inpoints4[[0,1],:]
    # # print(inpoints2)

    return inpoints2

def TransformGround2Image(inPoints, cinfo):
    rows = inPoints.shape[0]
    columns = inPoints.shape[1]
    # print(columns)
    zero_row = np.zeros((1,columns),np.float32)

    count = 0;
    while(count < 1):
        inPoints = np.vstack((inPoints,zero_row))
        count = count + 1
    # print(inPoints)
    inPoints3 = inPoints
    # print(inPoints3)
    inPointsr3 = inPoints3[[2],:]
    # print(inPointsr3)
    inPoints3[[2],:] = -cinfo.cameraHeight
    # print(inPoints3)

    c1 = mt.cos(cinfo.pitch)
    s1 = mt.sin(cinfo.pitch)
    c2 = mt.cos(cinfo.yaw)
    s2 = mt.sin(cinfo.yaw)

    matp = np.array([[cinfo.focalLength[0]*c2 + c1*s2*cinfo.opticalCenter[0],-cinfo.focalLength[0]*s2+c1*c2*cinfo.opticalCenter[0],
                    -s1*cinfo.opticalCenter[0]],

                    [s2*(-cinfo.focalLength[1]*s1 +  c1*cinfo.opticalCenter[1]),c2*(-cinfo.focalLength[1]*s1 + c1*cinfo.opticalCenter[1]),
                    -cinfo.focalLength[1]*c1 - s1*cinfo.opticalCenter[1]],

                    [c1*s2,c1*c2,-s1]],dtype= np.float32)
    # print(matp)
    # print(inPoints3.shape)

    transform = matp.dot(inPoints3)
    # print(transform)
    inPointsr3 = transform[[2],:]
    # print(transform)
    for i in range(0,columns):
        div = inPointsr3[0][i]
        # print(div)
        transform[[0],i] = transform[[0],i]/div
        transform[[1],i] = transform[[1],i]/div

    # # print(transform)

    inPoints2 = transform[[0,1],:]
    # print(inPoints2)
    return inPoints2











# input_image = cv.imread("/home/mohak/Downloads/roma/BDXD54/IMG00002.jpg",0)
input_image = cv.imread("/home/mohak/Downloads/caltech-lanes/cordova1/f00000.png",0)

# print(input_image)
input_image = input_image*1./255
# print(input_image)
# cv.imshow("input",input_image)
# cv.waitKey(0)

#Get size of input image
image_shape = input_image.shape
# print(input_image.shape)
width  = float(image_shape[1])
height  = float(image_shape[0])
# print(height)


#Try  to get the vanishing point
#Initialize CameraInfo Object
cinfo = CameraInfo([309.4362,344.2161],[317.9034,256.5352],2179.8,14.0,0.0,640,480)
# print(cinfo.pitch)
# print(cinfo.yaw)
coordinates = getVanishingPoint(cinfo)
# print(coordinates)
coordinates[1] = max(0,coordinates[1])
# print(coordinates[0])
# cv.line(input_image,(0,0),(coordinates[0],coordinates[1]),(255,0,0),15)
# cv.imshow("input",input_image)
# cv.waitKey(0)

# print(coordinates)
#get extent of image in the xfyf plane
#Initialize obejct of class IPMInfo
ipmInfo = IPMInfo(160,120,0.2,85,550,50,380,0)

#Size for output Image
output_width = ipmInfo.width
output_height = ipmInfo.height
outImage = np.zeros((output_height,output_width),np.float32)
# print(outImage.shape)
# print(outImage)
# cv.imshow("Result",outImage)
# cv.waitKey(0)

eps = ipmInfo.vpPortion*height
# print(eps)
ipmInfo.ipmLeft = max(0,ipmInfo.ipmLeft)
ipmInfo.ipmRight = min(width-1,ipmInfo.ipmRight)
ipmInfo.ipmTop = max(coordinates[1] + eps,ipmInfo.ipmTop)
ipmInfo.ipmBottom = min(height-1, ipmInfo.ipmBottom)
# print(ipmInfo.ipmLeft)
# print(ipmInfo.ipmRight)
# print(ipmInfo.ipmTop)
# print(ipmInfo.ipmBottom)

uvLimits = np.array([[coordinates[0],ipmInfo.ipmRight,ipmInfo.ipmLeft,coordinates[0]],
                    [ipmInfo.ipmTop,ipmInfo.ipmTop,ipmInfo.ipmTop,ipmInfo.ipmBottom]],dtype = np.float32)
# print(uvLimits)

#Get points on the ground plane
xyLimits = TransformImage2Ground(uvLimits, cinfo)
# print(xyLimits)

#get extent  on the ground plane
row1 = xyLimits[[0]]
row2 = xyLimits[[1]]
# print(row1)
# # # print(row1.astype(np.float32,copy=False))
row1 = row1.astype(np.float32, copy=False)
# # print(row1)
row2 = row2.astype(np.float32,copy=False)
min_max_1 = cv.minMaxLoc(row1)
xfMin = min_max_1[0]
xfMax = min_max_1[1]
# print(xfMin)
# print(xfMax)
min_max_2 = cv.minMaxLoc(row2)
# print(row2)
yfMin = min_max_2[0]
yfMax = min_max_2[1]
# # print(row2)
# print(yfMin)
# print(yfMax)
outRow = int(ipmInfo.height)
outCol = int(ipmInfo.width)
# print(yfMax - yfMin)
stepRow = float((yfMax - yfMin)/outRow)
stepCol = float((xfMax - xfMin)/outCol)
# print(stepCol)
# print(stepRow)
xyGrid = np.zeros((2,outRow*outCol),np.float32)
# print(xyGrid.shape)
#fill it with x-y values on the ground plane in world frame
y = yfMax - 0.5*stepRow
x = xfMin + 0.5*stepCol
for i in range(0,outRow):
    for j in range(0,outCol):
        xyGrid[[0],i*outCol+j] = x
        xyGrid[[1],i*outCol+j] = y
        x = x + stepCol
        # j = j +1
    y = y - stepRow
    # i = i +1

# print(xyGrid)
# #Get pixel values in Image Frame

uvGrid = TransformGround2Image(xyGrid,cinfo)
# cv.imshow("Result",uvGrid)
# print(uvGrid)

# print(uvGrid)
# #Calculate mean of the input image
mean = cv.mean(input_image)[0]
# print(mean)
outpoints = []
# print(outCol)
count = 0;
# print(ipmInfo.ipmLeft)
# print(ipmInfo.ipmRight)
# print(ipmInfo.ipmTop)
# print(ipmInfo.ipmBottom)
for i in range(0,outRow):
    for j in range(0,outCol):
#         # print("J-value :" + str(j)  + "i-value : " + str(i))
        u_i = float(uvGrid[[0],i*outCol + j])
        v_i = float(uvGrid[[1],i*outCol + j])
        #Check if out of boundary
        # print(u_i)
        # print(v_i)
        if (u_i<ipmInfo.ipmLeft or u_i > ipmInfo.ipmRight  or v_i <ipmInfo.ipmTop or v_i > ipmInfo.ipmBottom):
            outImage[[i],j] = mean
            # print("in if conditions")
            count = count + 1
        else:
            if(ipmInfo.ipmInterpolation == 0):
                x1 = int(u_i)
                x2 = int(u_i + 1)
                y1 = int(v_i)
                y2 = int(v_i+1)
                x = u_i - x1
                y = v_i - y1
                val = input_image[[y1],x1]* (1-x)*(1-y) +  input_image[[y1],x2]*x*(1-y) + input_image[[y2],x1]*(1-x)*y + input_image[[y2],x2]*x*y
                outImage[[i],j] = val
            else:
                outImage[[i],j] = input_image[[int(v_i + 0.5)], int(u_i + 0.5)]

        if(u_i < ipmInfo.ipmLeft + 10 or u_i > ipmInfo.ipmRight - 10 or v_i < ipmInfo.ipmTop or v_i > ipmInfo.ipmBottom -2):
            outpoints.append((j,i))

        # j = j +1

    # i = i +1
# print(count)
# print(outCol*outRow)
# ipmInfo.xLimits[0] = xyGrid[[0],0]
# ipmInfo.xLimits[1] =xyGrid[[0],(outRow-1)*outCol +  outCol -1]
# ipmInfo.yLimits[1] = xyGrid[[1],0]
# ipmInfo.yLimits[0] = xyGrid[[1],(outRow-1)*outCol+outCol -1]
# ipmInfo.xScale = 1/stepCol
# ipmInfo.yScale = 1/stepRow
# ipmInfo.width = outCol
# ipmInfo.height = outRow
# print(outpoints)
xLimits = [[xyGrid[[0],0]],xyGrid[[0],(outRow-1)*outCol +  outCol -1]]
yLimits = [[ xyGrid[[1],(outRow-1)*outCol+outCol -1]],[xyGrid[[1],0]]]
xScale = 1/stepCol
yScale = 1/stepRow
print(xScale)
print(yScale)

ipmInfo.intialize_limits(xLimits, yLimits, xScale, yScale)
ipmInfo.width = outCol
ipmInfo.height = outRow
# # print(outImage)
# print(input_image)
# outimage = outImage
# print(outImage.size)
# print(input_image.shape)
# cv.imshow("Result",outImage)
# cv.waitKey(0)

