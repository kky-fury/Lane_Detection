#include"Lane_Detection.hpp"
bool debug=false;
//bool debug = true;

/*Convert Image to Gray on GPU*/
void convertrgb2Gray(const gpu::GpuMat& src, gpu::GpuMat& dst)
{
	gpu::cvtColor(src, dst,CV_RGB2GRAY);	
	
}

/*Filter Image*/
void filterImage(const gpu::GpuMat& src, gpu::GpuMat& dst, float width_kernel_x, float width_kernel_y,float sigmax, float sigmay)
{
	src.convertTo(dst, CV_32F, 1.0/255.0,0);	
	
	Mat g_kernel_y = Mat::zeros(2*width_kernel_y + 1,1,CV_32F);
	Mat g_kernel_x = Mat::zeros(1,2*width_kernel_x + 1,CV_32F);

	float variance_y, variance_x;
	variance_y = sigmay*sigmay;
	variance_x = sigmax*sigmax;

	float k1, k2,function;

	/*Calculate g_kernel_y*/
	for(int i = -width_kernel_y;i<=width_kernel_y;i++)
	{
		k1 = exp((-0.5/variance_y)*i*i);
		g_kernel_y.at<float>(i+width_kernel_y,0) = k1; 	
	}

	/*Calculate g_kernel_x*/
	for(int i=-width_kernel_x;i<=width_kernel_x;i++)	
	{
		k2 = exp(-i*i*0.5/variance_x);
		function = (1/variance_x)*k2 - (i*i)/(variance_x*variance_x)*k2;
		g_kernel_x.at<float>(0,i+width_kernel_x) = function;
	}

	if(debug)
	{
		cout << "g_kernel_y = " << endl << " " << g_kernel_y << endl << endl;
		cout << "g_kernel_x = " << endl << " " << g_kernel_x << endl << endl;	
	}

	/*Initialize Kernel*/
	Mat kernel(2*width_kernel_x+1, 2*width_kernel_y +1,CV_32F);
	kernel = g_kernel_y*g_kernel_x;	
	
	if(debug)
	{
		cout<< "Kernel = " << endl<< " "<<kernel<<endl<<endl; 
	}
	
	Scalar mean_kernel;
	mean_kernel = mean(kernel)(0);
	subtract(kernel,mean_kernel,kernel);

	/*Filter Image on gpu*/
	gpu::filter2D(dst,dst,-1,kernel);
	
}

/*Get quantile value for Thresholding*/
void getQuantile(const gpu::GpuMat& src,gpu::GpuMat& dst, float  qtile)
{
	int number_rows, number_columns;
	number_rows = src.rows;
	number_columns = src.cols;
	
	Mat temp_image;
	src.download(temp_image);

	Mat array = temp_image.reshape(1,number_rows*number_columns);
	float quantile = getPoints(array, qtile); 

	if(debug)
	{
		cout<<quantile<<endl;
	}
	thresholdlower(src,dst,quantile);	


}

float getPoints(Mat& input_image, float quantile)
{
	Size size = input_image.size();
	int num_elements = size.width*size.height;
	double min, max;
	minMaxLoc(input_image, &min,&max);
		
	if(num_elements == 0)
	{
		return float(0);
	}
	else if(num_elements ==1)
	{
		return input_image.at<float>(0,0);
	}
	else if(quantile <=0)
	{
		return float(min);
	}
	else if (quantile >=1)
	{
		return float(max);
	}
	
	double pos =(num_elements-1)*quantile;
	unsigned int index = pos;
	double delta = pos - index;



	vector<float> w(num_elements);
	w.assign((float*)input_image.datastart, (float*)input_image.dataend);
	
	if(debug)
	{
		for(auto i = w.begin(); i !=w.end();++i)
		{
			cout<<*i<<' ';

		}
	}

	nth_element(w.begin(),w.begin() + index,w.end());
	float i1 = *(w.begin() + index);
	float i2 = *min_element(w.begin() + index +1,w.end());
	return (float)(i1*(1.0 - delta) + i2*delta);
	
}

void thresholdlower(const gpu::GpuMat& src, gpu::GpuMat& dst,double threshold)
{
	double retval;
	double maxval = 0.0;
	retval = gpu::threshold(src,dst,threshold,maxval,THRESH_TOZERO);
	if(debug)
	{	
		cout<<retval<<endl;
	}	
}

void getclearImage(gpu::GpuMat& src, gpu::GpuMat& dst)
{
	int number_rows, number_columns;

	number_rows = src.rows;
	number_columns = src.cols;
	
	/*
	Mat image_to_process;
	src.download(image_to_process);

	if(debug)
	{
		cout<<number_rows<<endl;
		cout<<number_columns<<endl;
	}

	int column_index = (int)(number_columns*0.75);
	int row_index = (int)(number_rows*0.75);

	//Clear Outer Columns to Zero
	for(int i = 0;i<number_rows;i++)
	{
		for(int j = column_index;j<number_columns;j++)
		{
			image_to_process.at<float>(i,j) = float(0);
			image_to_process.at<float>(i,(j-column_index)) = float(0);

		}
	}
    
	for(int i = 0;i<number_columns;i++)
	{
		for(int j =0;j<(number_rows-row_index);j++)
		{
			image_to_process.at<float>(j,i) = float(0); 

		}

	}

	dst.upload(image_to_process);		
	*/

	//More Optimized Code Without Copying Image to host
	
	int column_index = (int)(number_columns*0.75);
	int row_index = (int)(number_rows*0.75);

	gpu::GpuMat temp_roi;

	temp_roi = gpu::GpuMat(src, Rect(0,0,(number_columns-column_index),number_rows-1));
	temp_roi.setTo(Scalar(0));

	temp_roi = gpu::GpuMat(src, Rect(column_index,0,(number_columns-column_index-1),number_rows-1));
	temp_roi.setTo(Scalar(0));
	
	temp_roi = gpu::GpuMat(src, Rect(0,0,number_columns-1,(number_rows- row_index)));
	temp_roi.setTo(Scalar(0));

	dst = src.clone();

	if(debug)
	{
		Mat dst_host;
		src.download(dst_host);
		imshow("Result", dst_host);
		waitKey(0);
	}	

}

void getbinaryImage(const gpu::GpuMat& src, gpu::GpuMat& dst)
{
	double min, max ;	
	gpu::minMax(src,&min,&max);

	double threshold = (max - min)/2;
	gpu::threshold(src, dst, threshold,1, THRESH_BINARY);


}

void selectROI(const gpu::GpuMat& src, gpu::GpuMat& dst)
{
	int number_rows, number_columns;

	number_rows = src.rows; //400 ()
	number_columns = src.cols;//200
	
	int roi_height = int(0.45*number_rows);
	cout<<roi_height<<endl;	
	dst = gpu::GpuMat(src, Rect(0,roi_height,number_columns-1, number_rows-roi_height));
	
	if(1)
	{
		Mat dst_host;
		dst.download(dst_host);
		imshow("Result", dst_host);
		waitKey(0);
	}



}
























int main(int argc, char* argv[])
{
	/*Load Image*/
	Mat src_host;
	src_host = imread("/home/nvidia/Lane_Detection/Test_Images/IPM_test_image_0.png");
	
	gpu::GpuMat input_image, gray_image;
		
	/*Upload Image on Gpu*/
	input_image.upload(src_host);
	
	/*Convert Image to Gray*/
	convertrgb2Gray(input_image, gray_image);


	if(debug)
	{
		Mat dst_host;
		gray_image.download(dst_host);
		imshow("Result",dst_host);
		waitKey(0);
	}
	
	/*Filter Image using 2-D Gaussian Kernel*/
	
	gpu::GpuMat filtered_image;
	filterImage(gray_image, filtered_image,2,2,2,10);
	
	if(debug)
	{
		Mat dst_host;
		filtered_image.download(dst_host);
		imshow("Result",dst_host);
		waitKey(0);
	}
	
	/*Threshold Image*/
	gpu::GpuMat thresholded_image ;
	getQuantile(filtered_image, thresholded_image,0.985);

	if(debug)
	{
		Mat dst_host;
		thresholded_image.download(dst_host);
		imshow("Result",dst_host);
		waitKey(0);

	}

	/*Clean Negetive Parts Of an Image*/
	gpu::GpuMat clear_thresholded_image ;
	thresholdlower(thresholded_image, clear_thresholded_image, 0);

	if(debug)
	{
		Mat dst_host;
		clear_thresholded_image.download(dst_host);
		imshow("Result", dst_host);
		waitKey(0);

	}

	/*Clear Image (Select ROI)*/
	gpu::GpuMat cleared_image;
	getclearImage(clear_thresholded_image, cleared_image);

	if(debug)
	{
		Mat dst_host;
		cleared_image.download(dst_host);
		imshow("Result", dst_host);
		waitKey(0);
	}

	gpu::GpuMat binary_image;
	getbinaryImage(cleared_image, binary_image);
		
	if(debug)
	{
		Mat dst_host;
		binary_image.download(dst_host);
		imshow("Result", dst_host);
		waitKey(0);
	}

	gpu::GpuMat roi_image;
	selectROI(binary_image, roi_image);
	

}

