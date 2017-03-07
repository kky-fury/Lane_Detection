# Lane_Detection
The aim of this project is to develop and implement a Computer Vision based algorithm such that: <br />
	1. All visible lane boundaries are detected for urban/highway roads under close to idle weather conditions

Dependencies:<br />
	1. skimage <br />
	2. OpenCV >= 2.4.6 <br />
	3. numpy

Code Files:<br />
	1. GenerateBEV.py - Code from KITTI Road Dataset to obtain IPM view of an Image using a calibrated camera. <br />
	2. Lane_Detection.py - Detects multiple Lanes taking the IPM image as input. <br />
	3. Inverse_Perspective_Mapping.py - Generic code for obtaining the BEV of an image. Calculates the homography matrix by using the extrinc and intrinsic camera parameters.


Lane Detection for Test Images:

Input IPM Image (KITTI)		|	IPM Image After Lane Detection
:----------------------:|:-------------------------------------:
![img_1](https://github.com/kky-fury/Lane_Detection/blob/master/Test_Images/IPM_test_image_0.png) | ![img_2](https://github.com/kky-fury/Lane_Detection/blob/master/Lane_Detected_Images/image_0.png)
![img_3](https://github.com/kky-fury/Lane_Detection/blob/master/Test_Images/IPM_test_image_1.png) | ![img_4](https://github.com/kky-fury/Lane_Detection/blob/master/Lane_Detected_Images/image_1.png)
![img_5](https://github.com/kky-fury/Lane_Detection/blob/master/Test_Images/IPM_test_image_3.png) | ![img_6](https://github.com/kky-fury/Lane_Detection/blob/master/Lane_Detected_Images/image_3.png)
![img_7](https://github.com/kky-fury/Lane_Detection/blob/master/Test_Images/IPM_test_image_4.png) | ![img_8](https://github.com/kky-fury/Lane_Detection/blob/master/Lane_Detected_Images/image_4.png)

<br />

Input Approx IPM Image (Roma) 	|	IPM Image After Lane Detection
:----------------------------:|:-------------------------------:
![img_9](https://github.com/kky-fury/Lane_Detection/blob/master/Test_Images/IPM_test_image_5.png) | ![img_10](https://github.com/kky-fury/Lane_Detection/blob/master/Lane_Detected_Images/approx_image_0.png)![img_11](https://github.com/kky-fury/Lane_Detection/blob/master/Test_Images/IPM_test_image_8.png) | ![img_12](https://github.com/kky-fury/Lane_Detection/blob/master/Lane_Detected_Images/image_8.png)
	



	


