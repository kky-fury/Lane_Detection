#include"main.hpp"


int main(int argc, char* argv[])
{

	//Mat input_image =  imread("/home/nvidia/Lane_Detection/Original_Images/img_2.png");
	Mat input_image = imread("/home/nvidia/image_2/um_000002.png");

	unsigned char* h_rgb_img = input_image.data;

	/*
	Mat gray_image(IMAGE_HEIGHT_RGB, IMAGE_WIDTH_RGB, CV_8UC1);
	unsigned char* img = gray_image.data;
	*/
	/*
	
	for(int i =0;i<IMAGE_HEIGHT_RGB;i++)
	{
		for(int j = 0;j<IMAGE_WIDTH_RGB;j++)
		{

			*(img + i*IMAGE_WIDTH_RGB + j) = *(h_grayImage + i*IMAGE_WIDTH_RGB + j);
		}
	}
	
	imshow("Gray_Image", gray_image);
	waitKey(0);
	*/
	/*Gererate IPM View*/

	float bev_res = 0.1;
	tuple_int bev_xRange_minMax = {-10,10};
	tuple_int bev_zRange_minMax = {6, 46};
	float invalid_value = -numeric_limits<float>::infinity(); 

	BirdsEyeView bev(bev_res, invalid_value,bev_xRange_minMax, bev_zRange_minMax);
	/*Projection matrix for left color camera in rectified coordinates*/
	//For Image_0
	matrix_t intrinsic_matrix
	{
		{7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02, 4.485728000000e+01},
		{0.000000000000e+00 ,7.215377000000e+02 ,1.728540000000e+02 ,2.163791000000e-01},
		{0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 2.745884000000e-03}
	};
	/*Initialize Rotation Matrix (3x3) */
	matrix_t rotation_matrix
	{
		{9.999239000000e-01, 9.837760000000e-03, -7.445048000000e-03},
		{-9.869795000000e-03, 9.999421000000e-01,-4.278459000000e-03},
		{7.402527000000e-03, 4.351614000000e-03, 9.999631000000e-01}
	};
	/*Rigid transformation from (non-rectified) camera to road coordinates* (3x4)*/
	matrix_t Tr_cam_to_road
	{
		{9.999044710077e-01,-1.170165577363e-02, -7.360826724365e-03, 1.911984983337e-02},
		{1.160251828357e-02,9.998432738993e-01,-1.336987011872e-02,-1.562198078590e+00},
		{7.516122576373e-03, 1.328318612284e-02, 9.998834806284e-01,2.752775890648e-01}
	
	};
	/*For Image_1*/
	/*
	matrix_t Tr_cam_to_road
	{
		{9.998780444958e-01, -1.379252251645e-02, -7.329673303654e-03, 2.259126679386e-02},
		{1.371927193990e-02, 9.998563329041e-01, -9.951851093180e-03, -1.576782598490e+00},
		{7.465881263639e-03, 9.850077017074e-03, 9.999235703691e-01,2.807170564751e-01}
	
	};
	*/
	
	/*For Image_2*/
	/*
	matrix_t Tr_cam_to_road
	{
		{9.998783653356e-01,-1.376905762475e-02,-7.330023107842e-03,2.258724696694e-02},
		{1.370891707988e-02,9.998724359945e-01,-8.192773298801e-03,-1.578818275890e+00},
		{7.441894688477e-03,8.091287506981e-03, 9.999395278003e-01,2.835072170308e-01},
	};
	*/
	
	/*For Image_13*/
	/*
	matrix_t Tr_cam_to_road
	{
		{9.993974242913e-01, -3.399222060168e-02, -7.027004866705e-03, 5.404177016000e-02},
		{3.393449925061e-02, 9.993905818164e-01, -8.176738594524e-03,-1.562421412154e+00},
		{7.300667829702e-03, 7.933351013649e-03, 9.999418343980e-01, 2.836494496477e-01}
	};
	*/
	bev.setup(intrinsic_matrix, rotation_matrix, Tr_cam_to_road);
	bev.initialize();

	//auto begin = std::chrono::high_resolution_clock::now();
	unsigned char* h_grayImage =  rgb2gray(h_rgb_img);


	Mat gray_image(IMAGE_HEIGHT_RGB, IMAGE_WIDTH_RGB, CV_8UC1);
	unsigned char* img = gray_image.data;

	unsigned char* ipm_image = bev.computeLookUpTable(h_grayImage);
	
	Mat gray_IPM_image(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1);
	unsigned char* o_im  = gray_IPM_image.data;
	for(int i =0 ;i< IMAGE_HEIGHT;i++)
	{
		for(int j =0;j<IMAGE_WIDTH;j++)
		{
			gray_IPM_image.at<unsigned char>(i,j) = *(ipm_image + i*IMAGE_WIDTH + j);
			//cout<<(int)*(ipm_image + i*IMAGE_WIDTH + j)<<"\t";
		}
	}
	
	//imshow("Result", gray_IPM_image);
	//waitKey(0);

	
	unsigned char* bin_image = convert2fp(ipm_image);

	/*
	Mat output_image(ROI_IMAGE_HEIGHT, ROI_IMAGE_WIDTH, CV_8UC1);
	unsigned char* poutputimage = output_image.data;
	for(int i  =0;i<ROI_IMAGE_HEIGHT;i++)
	{
		for(int j  =0;j<ROI_IMAGE_WIDTH;j++)
		{
			*(poutputimage + i*ROI_IMAGE_WIDTH + j) = *(bin_image + i*ROI_IMAGE_WIDTH +j);

		}
	}
	
	imshow("Result", output_image);
	waitKey(0);
	*/

	float rMin = 0;
	float rMax = (IMG_WIDTH + IMG_HEIGHT)*2 + 1;
	float rStep = 1.0;

	float thetaMax = 180;
	float thetaMin = 0;
	float thetaStep = 1;

	const int numangle = std::round((thetaMax - thetaMin)/thetaStep);
	const int numrho = std::round(rMax/rStep);

	lines_w_non_zero* values = houghTransform(bin_image, numangle, numrho, thetaStep, rStep);

	int line_count = values->hough_lines->countlines;
	//cout<<"Line Count \t"<<line_count<<endl;

	/*
	for(int i  =0;i<line_count;i++)
	{
		//float theta_line = (hough_lines->lines + i)->y;
		//float rho = (hough_lines->lines + i)->x;

		float theta_line = (values->hough_lines->lines + i)->y;
		float rho = (values->hough_lines->lines + i)->x;
		double a = cos(theta_line);
		double b = sin(theta_line);

		double x0 = a*rho;
		double y0 = b*rho;

		cv::Point pt1, pt2;
		pt1.x = (int)(x0 + 400*(-b));
		pt1.y = (int)(y0 + 400*(a));
		pt2.x = (int)(x0 - 400*(-b));
		pt2.y = (int)(x0 - 400*(a));

		line(gray_IPM_image, pt1,pt2, (255,0,0),1);
	}
	imshow("Inital Guess After Hough", gray_IPM_image);
	waitKey(0);
	*/

	vector<Line> line_objects(line_count);
	getLineObjects(line_objects, values->hough_lines,values->votes,IMAGE_WIDTH, IMAGE_HEIGHT);
	
	/*
	Linepoint startpoint, endpoint;
	for(int i =0;i<line_objects.size();i++)
	{
		startpoint = line_objects[i].getstartpoint();
		endpoint = line_objects[i].getendpoint();
	//	cout<<"StartPoint  \t"<<"X_coordinate \t"<<startpoint.x<<"\t"<<"Y_Coordinate \t"<<startpoint.y<<endl;
	//	cout<<"EndPoint \t"<<"X_coordinate \t"<<endpoint.x<<"\t"<<"Y_Coordinate \t"<<endpoint.y<<endl;
		cv::Point pt1, pt2;
		pt1.x = startpoint.x;
		pt1.y = startpoint.y;

		pt2.x = endpoint.x;
		pt2.y = endpoint.y;

		line(gray_IPM_image, pt1,pt2, (255,0,0),1);
	
	}
	imshow("Result", gray_IPM_image);
	waitKey(0);
	*/

	initializePoints(line_objects, values->clist, values->count);	

	/*
	for(int i =0;i<line_objects.size();i++)
	{
		for(int j = 0;j<line_objects[i].x_y_points.size();j++)
		{
			cout<<"X Coordinate \t"<<line_objects[i].x_y_points[j].x<<"\t"<<"Y Coordinate \t"<<line_objects[i].x_y_points[j].y<<"\t";

		}

		cout<<endl;
	}
	*/


	fit_line(line_objects, gray_IPM_image);
	/*	
	unsigned char* line_detected_image = gray_IPM_image.data;
	unsigned char* perspective_image = bev.getperspectiveView(line_detected_image);

	Mat gray_IPM_image_detected(IMAGE_HEIGHT_RGB, IMAGE_WIDTH_RGB, CV_8UC1);
	unsigned char* p_im_pointer = gray_IPM_image_detected.data;

	for(int i =0;i<IMAGE_HEIGHT_RGB;i++)
	{
		for(int j =0;j<IMAGE_WIDTH_RGB;j++)
		{
		
			*(p_im_pointer + i*IMAGE_WIDTH_RGB + j) = *(perspective_image + i*IMAGE_WIDTH_RGB + j);

		}
	}


	imshow("Result", gray_IPM_image_detected);
	waitKey(0);
	//auto end = std::chrono::high_resolution_clock::now();
	//cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() << "ns" << std::endl;
	*/
	imshow("Result", gray_IPM_image);
	waitKey(0);
	


}
