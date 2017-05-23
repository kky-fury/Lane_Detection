# Lane_Detection
The aim of this project is to develop and implement a Computer Vision based algorithm such that: <br />
	1. All visible lane boundaries are detected for urban/highway roads under close to idle weather conditions

<!--[//]:Dependencies:<br />
[//]:#	1. skimage <br />
[//]:#	2. OpenCV >= 2.4.6 <br />
[//]:	3. numpy

[//]:Code Files:<br />
[//]:	1. GenerateBEV.py - Code from KITTI Road Dataset to obtain IPM view of an Image using a calibrated camera. <br />
[//]:	2. Lane_Detection.py - Detects multiple Lanes taking the IPM image as input. <br />
[//]:	3. Inverse_Perspective_Mapping.py - Generic code for obtaining the BEV of an image. Calculates the homography matrix by using the extrinc and intrinsic camera parameters.
-->
Dependencies:<br />
	1. c++ Boost >= 1.7
	2. CUDA Toolkit >= 7.0
	3. OpenCV >= 2.4.6

Relevant Code Files: <br />
	1. GenerateBEV.cu - CUDA Acceralated Code to get Perspective-IPM and IPM-Perspective view of an input image.
	2. Pre_Processing.cu - CUDA Acceralated Code to pre_process the input image.
	3. rgb2gray.cu - CUDA Acceralated Code to convert rgb image to grayscale.
	4. Hough_Transform.cu - Fast Hough Transform using CUDA
	5. Line.cpp - Group identical lanes and eliminates outliers
	6. line_fitting.cpp - Using Ransac algorithm to fit a line to the detected lane. Bresenham's algorithm to plot the line pixels.
	7. fit_poly.cpp - Least Squares 2nd degree polynomial fitting for curved lanes

To Do: <br />
	Calculate Pixel Based Performance Metrics
#For detecting Curved Lanes - Curve fitting using Ransac


Process Pipeline: <br />

Original Image	|	Input IPM Image		|	Filtered IPM Image	|	Thresholded Image|	Binary Image After Selecting ROI	|	Initial Guess for Ransac	|	Lane Detected Image After Ransac and Eliminating False Lanes
:------------------:|:-------------------:|:-----------------------:|:-----------------------------:|:-------------------------------------:|:-------------------------------:|:--------------------------------:
![img_24](https://github.com/kky-fury/Lane_Detection/blob/master/Original_Images/img_1.png)	| ![img_25](https://github.com/kky-fury/Lane_Detection/blob/master/Test_Images/IPM_test_image_1.png)	|		![img_19](https://github.com/kky-fury/Lane_Detection/blob/master/Process_Pipeline/filtered_image.png)	|	![img_20](https://github.com/kky-fury/Lane_Detection/blob/master/Process_Pipeline/thresholded_image.png)	|	![img_21](https://github.com/kky-fury/Lane_Detection/blob/master/Process_Pipeline/binary_image_after_ROI.png)	| ![img_22](https://github.com/kky-fury/Lane_Detection/blob/master/Process_Pipeline/Initial_Guess_For_Ransac.png) | ![img_23](https://github.com/kky-fury/Lane_Detection/blob/master/Lane_Detected_Images/image_1.png)


Lane Detection for Test Images: <br />

Orginal Image	|	Input IPM Image (KITTI)		|	IPM Image After Lane Detection
:---------------------------------:|:----------------------:|:-------------------------------------:
![img_13](https://github.com/kky-fury/Lane_Detection/blob/master/Original_Images/img_0.png)	|	![img_1](https://github.com/kky-fury/Lane_Detection/blob/master/Test_Images/IPM_test_image_0.png) | ![img_2](https://github.com/kky-fury/Lane_Detection/blob/master/Lane_Detected_Images/image_0.png)
![img_14](https://github.com/kky-fury/Lane_Detection/blob/master/Original_Images/img_1.png)	|	![img_3](https://github.com/kky-fury/Lane_Detection/blob/master/Test_Images/IPM_test_image_1.png) | ![img_4](https://github.com/kky-fury/Lane_Detection/blob/master/Lane_Detected_Images/image_1.png)
![img_15](https://github.com/kky-fury/Lane_Detection/blob/master/Original_Images/img_3.png)	|	![img_5](https://github.com/kky-fury/Lane_Detection/blob/master/Test_Images/IPM_test_image_3.png) | ![img_6](https://github.com/kky-fury/Lane_Detection/blob/master/Lane_Detected_Images/image_3.png)
![img_16](https://github.com/kky-fury/Lane_Detection/blob/master/Original_Images/img_4.png)	|	![img_7](https://github.com/kky-fury/Lane_Detection/blob/master/Test_Images/IPM_test_image_4.png) | ![img_8](https://github.com/kky-fury/Lane_Detection/blob/master/Lane_Detected_Images/image_4.png)


Original Image (ROMA)	|	Input Approx IPM Image (Roma) 	|	IPM Image After Lane Detection
:--------------------------:|:----------------------------:|:-------------------------------:
![img_17](https://github.com/kky-fury/Lane_Detection/blob/master/Original_Images/img_5.jpg)	|	![img_9](https://github.com/kky-fury/Lane_Detection/blob/master/Test_Images/IPM_test_image_5.png)	| 	![img_10](https://github.com/kky-fury/Lane_Detection/blob/master/Lane_Detected_Images/approx_image_0.png)
![img_18](https://github.com/kky-fury/Lane_Detection/blob/master/Original_Images/img_8.jpg)	|	![img_11](https://github.com/kky-fury/Lane_Detection/blob/master/Test_Images/IPM_test_image_8.png)	|	![img_12](https://github.com/kky-fury/Lane_Detection/blob/master/Lane_Detected_Images/image_8.png)
	



	


