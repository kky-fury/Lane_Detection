# Lane_Detection
The aim of this project is to develop and implement a Computer Vision based algorithm such that:
	a.) All visible lane boundaries are detected for urban/highway roads under close to idle weather conditions


Data Sets
	A major bottleneck for evaluating lane detection algorithms using pixel based metrics is the presence of standardized data sets with ground truth information. Generating ground truth information for sample images is time consuming, since the most common method is manual annotation. Moreover, camera calibration values required for different feometric transformations are rarely mentioned. To overcome this, the following data sets are used to evaluate our approach:
	
	1. ROMA (Road Markings)
		a.) Comprises of 116 images of diverse road scenes
		b.) Contains camera calibration parameters which can be used for Image Transformations
	2. Caltech Lanes DataSet
		a.) Comprises of 1225 frames from urban road enviornment
		b.) Contains camera calibration parameters which can be used for Image Transformations
