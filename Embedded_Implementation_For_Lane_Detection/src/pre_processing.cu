#include "preprocess.hpp"
#include"cuda_error_check.hpp"

cudaEvent_t start, stop;
bool debug_pre_process=false;


void  matrix_multiplication(float* arr1, int arr1_rows, int arr1_cols, float* arr2, int arr2_rows, int arr2_cols, float* r_arr)
{	
	for(int i = 0;i<arr1_rows;i++)
	{
		for(int j = 0;j < arr2_cols;j++)
		{
			*(r_arr + i*arr2_cols + j) = 0;
			for(int k =0;k<arr1_cols;k++)
			{	
				*(r_arr + i*arr2_cols + j) = *(r_arr + i*arr2_cols +j) + (*(arr1+ i*arr1_cols + k))*(*(arr2 + k*arr2_cols + j));		
			}

		}
	}
}


/*Using Thrust to Calculate Max/Min in an Image*/
void find_min_max(thrust::device_ptr<float> &dbeg, thrust::device_ptr<float> &dend, float* min, float* max)
{
	thrust::pair<thrust::device_vector<float>::iterator,thrust::device_vector<float>::iterator> tuple;
	//tuple = thrust::minmax_element(dev_vec.begin(), dev_vec.end());
	tuple = thrust::minmax_element(dbeg, dend);
	*(min) = *(tuple.first);
	*(max) = *(tuple.second);

}

__global__ void gaussian_blur_tiled(const float* const grayImage, float*
		filteredImage, int numRows, int numCols, const float* const filter,const int filterWidth)
{

	const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,blockIdx.y * blockDim.y + threadIdx.y);

	const int i = thread_2D_pos.y * numCols + thread_2D_pos.x;
	
	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
		return;

	int temp_x, temp_y;
	extern __shared__ float s_input[];


	int halfW = filterWidth/2;
	int smem_w = blockDim.x + halfW*2;
	int smem_h = blockDim.y + halfW*2;
	int smem_offset = smem_w*halfW + halfW;
	int smem_idx = smem_offset + smem_w*threadIdx.y + threadIdx.x;
	
	s_input[smem_idx] = grayImage[i];
	
	if(threadIdx.x < halfW)
	{
		
		temp_x = min(max(thread_2D_pos.x - halfW, 0), numCols-1);
		temp_y = thread_2D_pos.y;
		s_input[smem_idx-halfW] = grayImage[temp_y*numCols + temp_x];
		
		if(threadIdx.y < halfW)
		{
			temp_y = min(max(thread_2D_pos.y - halfW, 0),numRows-1);
			s_input[smem_idx-(halfW*smem_w)-halfW] =
			grayImage[temp_y*numCols + temp_x];
		
		}
		else if((threadIdx.y >= (blockDim.y-halfW)) ||((thread_2D_pos.y+halfW)>= numRows))
		{

			temp_y = min(max(thread_2D_pos.y + halfW, 0), numRows-1);
			s_input[smem_idx+(halfW*smem_w)-halfW] = grayImage[temp_y*numCols + temp_x];
		}
		
	}
	else if ((threadIdx.x >= (blockDim.x-halfW)) || ((thread_2D_pos.x+halfW) >= numCols))
	{
		temp_x = min(max(thread_2D_pos.x + halfW, 0), numCols-1);
		temp_y = thread_2D_pos.y;
		s_input[smem_idx+halfW] = grayImage[temp_y*numCols + temp_x];

		if (threadIdx.y < halfW)
		{
				temp_y = min(max(thread_2D_pos.y - halfW, 0),numRows-1);
				s_input[smem_idx-(halfW*smem_w)+halfW] = grayImage[temp_y*numCols + temp_x];
		
		}
		else if ((threadIdx.y >= (blockDim.y-halfW)) || ((thread_2D_pos.y+halfW) >= numRows))
		{
			temp_y = min(max(thread_2D_pos.y + halfW, 0), numRows-1);
			s_input[smem_idx+(halfW*smem_w)+halfW] = grayImage[temp_y*numCols + temp_x];

		}	
		
	}

	if (threadIdx.y < halfW)
	{
		temp_x = thread_2D_pos.x;
		temp_y = min(max(thread_2D_pos.y - halfW, 0), numRows-1);
		s_input[smem_idx-(halfW*smem_w)] = grayImage[temp_y*numCols + temp_x];
	}
	
	else if ((threadIdx.y >= (blockDim.y-halfW)) || ((thread_2D_pos.y+halfW) >= numRows))
	{
		temp_x = thread_2D_pos.x;
		temp_y = min(max(thread_2D_pos.y + halfW, 0), numRows-1);
		s_input[smem_idx+(halfW*smem_w)] = grayImage[temp_y*numCols + temp_x];

	}
	
	__syncthreads();

	int idx_x, idx_y;

	float result = 0.0f;
	for (idx_y=-halfW; idx_y<=halfW; idx_y++)
	{
		for(idx_x=-halfW; idx_x<=halfW; idx_x++)
		{
			result += filter[(idx_y+halfW)*filterWidth + (idx_x+halfW)]*s_input[smem_idx + idx_y*smem_h + idx_x];
		}
	}

	__syncthreads();


	filteredImage[i] = result;


}


/*Scales down an image to 32f*/
__global__  void uchar2fp(const unsigned char* const d_grayImage, float* d_grayImage_32f)
{

	int rgb_x = blockIdx.x*blockDim.x + threadIdx.x;
	int rgb_y = blockIdx.y*blockDim.y + threadIdx.y;

	if((rgb_x >= IMAGE_WIDTH) || (rgb_y >=IMAGE_HEIGHT))
	{
		return;
	}
	
	/*Scaling Image to 32f*/
	*(d_grayImage_32f + rgb_y*IMAGE_WIDTH + rgb_x) = *(d_grayImage +rgb_y*IMAGE_WIDTH + rgb_x)*(1.0f/255.0f);

}



__global__ void thresh_to_zero(float* d_image, const float threshold)
{

	int img_x = blockIdx.x*blockDim.x +  threadIdx.x;
	int img_y = blockIdx.y*blockDim.y + threadIdx.y;

	if((img_x >= IMAGE_WIDTH) || (img_y >= IMAGE_HEIGHT))
	{
			return;
	}

	if(!(*(d_image +img_y*IMAGE_WIDTH + img_x) >= threshold))
	{

			*(d_image +img_y*IMAGE_WIDTH + img_x) = 0;
	}



}

__global__ void thresh_binary(float* d_image,unsigned char* binary_image, const float threshold, const float max_val)
{
	int img_x = blockIdx.x*blockDim.x +  threadIdx.x;
	int img_y = blockIdx.y*blockDim.y + threadIdx.y;

	if((img_x >= ROI_IMAGE_WIDTH) || (img_y >= ROI_IMAGE_HEIGHT))
	{
			return;
	}

	if(*(d_image + img_y*ROI_IMAGE_WIDTH + img_x) >= threshold)
	{
		*(d_image + img_y*ROI_IMAGE_WIDTH + img_x) = max_val;
		
	}
	else
	{
		*(d_image +  img_y*ROI_IMAGE_WIDTH + img_x) = float(0);

	}

	__syncthreads();
	/*Downscale to uchar*/	
	*(binary_image + img_y*ROI_IMAGE_WIDTH + img_x) = (unsigned char)(*(d_image + img_y*ROI_IMAGE_WIDTH + img_x)*255.0); 

}

__global__ void selectROI(float* input_image, float* roi_selected_image)
{

	int img_x = blockIdx.x*blockDim.x +  threadIdx.x;
	int img_y = blockIdx.y*blockDim.y + threadIdx.y;

	/*Assuming ROI Image is 192x224*/
	
	if((img_x < ROI_IMAGE_WIDTH) && ((IMAGE_HEIGHT - ROI_IMAGE_HEIGHT) <= img_y < IMAGE_HEIGHT))
	{	
		*(roi_selected_image+(img_y -(IMAGE_HEIGHT-ROI_IMAGE_HEIGHT))*ROI_IMAGE_WIDTH + img_x) = *(input_image + img_y*IMAGE_WIDTH + img_x);
	}
	
}

__global__ void clearImage(float* input_image)
{

	int img_x = blockIdx.x*blockDim.x + threadIdx.x;
	int img_y = blockIdx.y*blockDim.y + threadIdx.y;

	if((img_x >= IMAGE_WIDTH) || (img_y >= IMAGE_HEIGHT))
	{
		return;
	}


	if(img_y < (IMAGE_HEIGHT - ROW_INDEX))
	{
	
		*(input_image + img_y*IMAGE_WIDTH + img_x) = float(0);	

	}
	else
	{
		
		if((img_x < (IMAGE_WIDTH - COLUMN_INDEX)) || (img_x > COLUMN_INDEX  && img_x < IMAGE_WIDTH))
		{
			*(input_image + img_y*IMAGE_WIDTH + img_x) = float(0);	

		}

	}
}


unsigned char* threshold_image(float* d_image, float* h_thresholded_image, float threshold, float min, float max)
{
	

	const dim3 blockSize(THREAD_X, THREAD_Y);
	const dim3 gridSize((IMAGE_WIDTH + blockSize.x -1)/blockSize.x, (IMAGE_HEIGHT + blockSize.y -1)/blockSize.y);
	
	thresh_to_zero<<<gridSize, blockSize>>>(d_image, threshold);
	cudaDeviceSynchronize();
	CudaCheckError();
	
	
	clearImage<<<gridSize, blockSize>>>(d_image);
	cudaDeviceSynchronize();
	CudaCheckError();
	
	if(debug_pre_process)
	{
		cudaMemcpy(h_thresholded_image, d_image, NUMPIX*sizeof(float), cudaMemcpyDeviceToHost);
		CudaCheckError();
		Mat output_image(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32F);
		for(int i =0;i<IMAGE_HEIGHT;i++)
		{
			for(int j = 0;j<IMAGE_WIDTH;j++)
			{
				output_image.at<float>(i,j) = *(h_thresholded_image + i*IMAGE_WIDTH + j);
			
			}
		}
	
		imshow("Result", output_image);
		waitKey(0);
	}
	
	float* h_roi_selectedImage = (float*)malloc(ROI_IMAGE_WIDTH*ROI_IMAGE_HEIGHT*sizeof(float));
	float* d_roi_selectedImage;

	cudaMalloc((void**)&d_roi_selectedImage,ROI_IMAGE_WIDTH*ROI_IMAGE_HEIGHT*sizeof(float));
	CudaCheckError();
	
	cudaMemset(d_roi_selectedImage,0, ROI_IMAGE_HEIGHT*ROI_IMAGE_WIDTH*sizeof(float));
	CudaCheckError();

	selectROI<<<gridSize,blockSize>>>(d_image, d_roi_selectedImage);
	cudaDeviceSynchronize();
	CudaCheckError();

	if(debug_pre_process)
	{	
		cudaMemcpy(h_roi_selectedImage,d_roi_selectedImage,sizeof(float)*ROI_IMAGE_HEIGHT*ROI_IMAGE_WIDTH, cudaMemcpyDeviceToHost);
		CudaCheckError();
		Mat roi_image(ROI_IMAGE_HEIGHT, ROI_IMAGE_WIDTH, CV_32F);
		for(int i =0;i<ROI_IMAGE_HEIGHT;i++)
		{
			for(int j = 0;j<ROI_IMAGE_WIDTH;j++)
				{
					roi_image.at<float>(i,j) = *(h_roi_selectedImage + i*ROI_IMAGE_WIDTH + j);
				
				}
		}
	
		//cout<<roi_image<<endl;
		cout<<roi_image.rows<<"\t"<<roi_image.cols;
		imshow("Result", roi_image);
		waitKey(0);

	}

	float bin_thresh = (max)/2;
	unsigned char* d_bin_image;
	cudaMalloc((void**)&d_bin_image, ROI_IMAGE_WIDTH*ROI_IMAGE_HEIGHT*sizeof(unsigned char));
	CudaCheckError();
	
	cudaMemset(d_bin_image, 0, ROI_IMAGE_WIDTH*ROI_IMAGE_HEIGHT*sizeof(unsigned char));
	CudaCheckError();

	dim3 gridSize_roi((ROI_IMAGE_WIDTH + blockSize.x -1)/blockSize.x, (ROI_IMAGE_HEIGHT + blockSize.y -1)/blockSize.y);

	thresh_binary<<<gridSize_roi, blockSize>>>(d_roi_selectedImage, d_bin_image,bin_thresh, 1);
	cudaDeviceSynchronize();
	CudaCheckError();

	if(debug_pre_process)
	{
		cudaMemcpy(h_roi_selectedImage,d_roi_selectedImage,sizeof(float)*ROI_IMAGE_HEIGHT*ROI_IMAGE_WIDTH, cudaMemcpyDeviceToHost);
		//CudaCheckError();
		Mat roi_image(ROI_IMAGE_HEIGHT, ROI_IMAGE_WIDTH, CV_32F);
		for(int i =0;i<ROI_IMAGE_HEIGHT;i++)
		{
			for(int j = 0;j<ROI_IMAGE_WIDTH;j++)
			{
				roi_image.at<float>(i,j) = *(h_roi_selectedImage + i*ROI_IMAGE_WIDTH + j);
				
			}
		}
	
	
		//cout<<roi_image<<endl;
		imshow("Result", roi_image);
		waitKey(0);
	}
	unsigned char* h_bin_image = (unsigned char*)malloc(ROI_IMAGE_WIDTH*ROI_IMAGE_HEIGHT*sizeof(unsigned char));
	cudaMemcpy(h_bin_image, d_bin_image,ROI_IMAGE_WIDTH*ROI_IMAGE_HEIGHT*sizeof(unsigned char), cudaMemcpyDeviceToHost);
	CudaCheckError();

	/*
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout<<"Time \t"<<milliseconds<<endl;
	*/

	if(1)
	{
		cout<<(int)*(h_bin_image)<<endl;
		Mat bin_image(ROI_IMAGE_HEIGHT, ROI_IMAGE_WIDTH, CV_8UC1);
		for(int i = 0;i<ROI_IMAGE_HEIGHT;i++)
		{
			for(int j = 0;j<ROI_IMAGE_WIDTH;j++)
			{
				bin_image.at<unsigned char>(i,j) = *(h_bin_image + i*ROI_IMAGE_WIDTH + j);
			}
		}
		imshow("Result", bin_image);
		waitKey(0);
	
	
	}

	return h_bin_image;

}


unsigned char* getQuantile(float* d_filteredImage,float* h_filtered_Image, float qtile)
{
	thrust::device_ptr<float> dbeg(d_filteredImage);
	thrust::device_ptr<float> dend = dbeg + NUMPIX;

	float min, max;
	find_min_max(dbeg, dend, &min, &max);
	
	float quantile_value = 0;

	if(NUMPIX == 0)
	{
		quantile_value = float(0);

	}
	else if(NUMPIX == 1)
	{
		quantile_value = *(d_filteredImage +0);

	}
	else if(qtile <=0)
	{
		cout<<"Third If Statement"<<endl;
		quantile_value = min;
	}
	else if (qtile >=1)
	{
		quantile_value =  max;
	}
	else
	{
		float pos = (NUMPIX - 1)*qtile;
		unsigned int index = pos;
		float delta = pos - index;	
		vector<float> w(h_filtered_Image, h_filtered_Image + NUMPIX);
		nth_element(w.begin(), w.begin() +index, w.end());
		float i1 = *(w.begin() + index);
		float i2 = *min_element(w.begin() + index +1, w.end());
		quantile_value = (float)(i1*(1.0 - delta) + i2*delta);

	}

	float* h_thresholded_image = (float*)malloc(sizeof(float)*NUMPIX);
	
	unsigned char* bin_image = threshold_image(d_filteredImage, h_thresholded_image, quantile_value, min,max);
	
	return bin_image;

}


unsigned char* filterImage(const float* const grayImage, int width_kernel_x, int  width_kernel_y, float sigmax, float sigmay)
{
	
	int nrow_kernel_y = 2*width_kernel_x + 1;
	int ncol_kernel_x = 2*width_kernel_y + 1;

	float variance_y, variance_x;
	variance_y = sigmay*sigmay;
	variance_x = sigmax*sigmax;

	float k1, k2, function;
	
	float* kernel_x = (float*)malloc(sizeof(float)*ncol_kernel_x);
	float* kernel_y = (float*)malloc(sizeof(float)*nrow_kernel_y);
	float* kernel = (float*)malloc(sizeof(float)*ncol_kernel_x*nrow_kernel_y);

	for(int i = -width_kernel_y;i<=width_kernel_y;i++)
	{		
		k1 = exp((-0.5/variance_y)*i*i);
		*(kernel_y + i + width_kernel_y) = k1;  
	}

	for(int i = -width_kernel_x;i<=width_kernel_x;i++)
	{
		k2 = exp(-i*i*0.5/variance_x);
		function = (1/variance_x)*k2 - (i*i)/(variance_x*variance_x)*k2;
		*(kernel_x + i + width_kernel_x) =  function;

	}

	matrix_multiplication(kernel_y, nrow_kernel_y,1,kernel_x, 1,ncol_kernel_x,
			kernel);

	if(debug_pre_process)
	{
		for(int i  =0;i<5;i++)
		{
			for(int j =0;j<5;j++)
			{
				cout<<*(kernel + i*5 +j)<<"\t";

			}
		cout<<endl;
		}
	}

	float mean = 0;
	for(int i =0;i<nrow_kernel_y;i++)
	{
		for(int j =0;j<ncol_kernel_x;j++)
		{
				mean += *(kernel + i*ncol_kernel_x + j);

		}
	}

	mean /=nrow_kernel_y*ncol_kernel_x;


	/*Subtract Mean Value from Kernel*/
	for(int i =0;i<nrow_kernel_y;i++)
	{
		for(int j =0;j<ncol_kernel_x;j++)
		{		
				*(kernel + i*ncol_kernel_x + j) -= mean;
		}	
		//cout<<endl;
	}


	float* d_kernel, *d_filter_Image;
	float* h_filter_Image = (float*)malloc(NUMPIX*sizeof(float));	

	cudaMalloc((void**)&d_kernel,nrow_kernel_y*ncol_kernel_x*sizeof(float));
	CudaCheckError();

	cudaMemcpy(d_kernel,kernel,nrow_kernel_y*ncol_kernel_x*sizeof(float),cudaMemcpyHostToDevice);
	CudaCheckError();

	const int filterWidth = CU_FILTER_WIDTH;

	cudaMalloc((void**)&d_filter_Image, NUMPIX*sizeof(float));
	CudaCheckError();

	const dim3 blockSize(THREAD_X, THREAD_Y);	
	const dim3 gridSize((IMAGE_WIDTH + blockSize.x -1)/blockSize.x, (IMAGE_HEIGHT + blockSize.y -1)/blockSize.y);

	size_t smemSize =(blockSize.x + ((filterWidth/2) * 2))*(blockSize.y +((filterWidth/2) * 2))*sizeof(float); 

	gaussian_blur_tiled<<<gridSize,blockSize,smemSize>>>(grayImage,d_filter_Image, IMAGE_HEIGHT, IMAGE_WIDTH,d_kernel, filterWidth);
	cudaDeviceSynchronize();
	CudaCheckError();
	
	cudaMemcpy(h_filter_Image, d_filter_Image, NUMPIX*sizeof(float), cudaMemcpyDeviceToHost);
	unsigned char* bin_image = getQuantile(d_filter_Image, h_filter_Image,0.985);
	
	if(debug_pre_process)
	{
		Mat output_image(IMAGE_HEIGHT,IMAGE_WIDTH, CV_32F);
		for(int i = 0;i<IMAGE_HEIGHT;i++)
		{
			for(int j =0;j<IMAGE_WIDTH;j++)
			{
				output_image.at<float>(i,j) = *(h_filter_Image + i*IMAGE_WIDTH + j);

			}
		}

		imshow("Result", output_image);
		waitKey(0);
	}

	return bin_image;


}



unsigned char* convert2fp(const unsigned char* const h_grayImage)
{
	unsigned char* d_grayImage;
	float* d_grayImage_32f;

	cudaMalloc((void**)&d_grayImage, NUMPIX*sizeof(unsigned char));
	CudaCheckError();

	cudaMalloc((void**)&d_grayImage_32f, NUMPIX*sizeof(float));
	CudaCheckError();

	cudaMemcpy(d_grayImage, h_grayImage, NUMPIX*sizeof(unsigned char), cudaMemcpyHostToDevice);
	CudaCheckError();

	cudaMemset(d_grayImage_32f, 0, NUMPIX*sizeof(float));
	CudaCheckError();

	const dim3 blockSize(THREAD_X, THREAD_Y);
	const dim3 gridSize((IMAGE_WIDTH + blockSize.x -1)/blockSize.x,(IMAGE_HEIGHT + blockSize.y -1)/blockSize.y);
	
	uchar2fp<<<gridSize,blockSize>>>(d_grayImage, d_grayImage_32f);

	unsigned char* bin_image = filterImage(d_grayImage_32f, 2, 2 ,1.95,10);

	
	
	if(debug_pre_process)
	{
		float* h_grayImage_32f;
		h_grayImage_32f = (float*)malloc(sizeof(float)*NUMPIX);

		cudaMemcpy(h_grayImage_32f, d_grayImage_32f, NUMPIX*sizeof(float), cudaMemcpyDeviceToHost);
		CudaCheckError();
		Mat gray_image_32f(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32F);
		
		for(int i =0;i<IMAGE_HEIGHT;i++)
		{
			for(int j =0;j<IMAGE_WIDTH;j++)
			{

					gray_image_32f.at<float>(i,j) = *(h_grayImage_32f + i*IMAGE_WIDTH + j);

			}

		}
		cout<<gray_image_32f<<endl;
	}

	return bin_image; 
}






