# Lane_Detection
The aim of this project is to develop and implement a Computer Vision based algorithm such that:
	1. All visible lane boundaries are detected for urban/highway roads under close to idle weather conditions

Dependencies:
	1. skimage
	2. OpenCV >= 2.4.6
	3. numpy

Code Files:
	1. GenerateBEV.py - Code from KITTI Road Dataset to obtain IPM view of an Image using a calibrated camera.
	2. Lane_Detection.py - Detects multiple Lanes taking the IPM image as input.
	3. Inverse_Perspective_Mapping.py - Generic code for obtaining the BEV of an image. Calculates the homography matrix by using the extrinc and intrinsic camera parameters.


Lane Detection for Test Images:

Input IPM Image		|	IPM Image After Lane Detection
:----------------------:|:-------------------------------------:
![img_1](https://github.com/kky-fury/Lane_Detection/blob/master/Test_Images/IPM_test_image_0.png) | ![img_2](https://github.com/kky-fury/Lane_Detection/blob/master/Lane_Detected_Images/image_0.png)

	



	


