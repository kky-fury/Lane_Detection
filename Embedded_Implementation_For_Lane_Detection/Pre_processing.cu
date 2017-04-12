#include "preprocess.hpp"
#include"cuda_error_check.hpp"

bool debug = false;

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


/*Converts rgb2gray and Scales down an image to 32f*/
__global__  void rgb2gray(unsigned char* d_grayImage, const unsigned char* const
		d_rgbImage, float* d_grayImage_32f)
{

	int rgb_x = blockIdx.x*blockDim.x + threadIdx.x;
	int rgb_y = blockIdx.y*blockDim.y + threadIdx.y;

	if((rgb_x >= IMAGE_WIDTH) || (rgb_y >=IMAGE_HEIGHT))
	{
		return;
	}

		
	unsigned char blue = float(*(d_rgbImage + 3*IMAGE_WIDTH*rgb_y + 3*rgb_x))*0.114f;
	unsigned char green = float(*(d_rgbImage + 3*IMAGE_WIDTH*rgb_y + 3*rgb_x +1))*0.587f;
	unsigned char red = float(*(d_rgbImage + 3*IMAGE_WIDTH*rgb_y + 3*rgb_x +2))*0.299f;
	
	*(d_grayImage + rgb_y*IMAGE_WIDTH + rgb_x) = uchar(blue +  green + red);
	*(d_grayImage_32f + rgb_y*IMAGE_WIDTH + rgb_x) = float(uchar(blue + green +red))*(1.0f/255.0f);

}



void filterImage(const float* const grayImage, int width_kernel_x, int  width_kernel_y, float sigmax, float sigmay)
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

	if(debug)
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
	cout<<"Mean Value \t"<<mean<<endl;


	/*Subtract Mean Value from Kernel*/
	for(int i =0;i<nrow_kernel_y;i++)
	{
		for(int j =0;j<ncol_kernel_x;j++)
		{		
				*(kernel + i*ncol_kernel_x + j) -= mean;
				cout<<*(kernel + i*ncol_kernel_x + j)<<"\t";
		}	
		cout<<endl;
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

	thrust::device_ptr<float> dbeg(d_filter_Image);
	thrust::device_ptr<float> dend = dbeg + NUMPIX;
	
	float min, max;
	find_min_max(dbeg, dend, &min, &max);



	cudaMemcpy(h_filter_Image, d_filter_Image, NUMPIX*sizeof(float), cudaMemcpyDeviceToHost);

	if(debug)
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


}



void convert2Gray(const unsigned char* const rgbImage, unsigned char* grayImage,
		float* h_grayImage_32f)
{
	unsigned char* d_rgbImage;
	unsigned char* d_grayImage;
	float* d_grayImage_32f;
	
	cudaMalloc((void**)&d_rgbImage, 3*NUMPIX*sizeof(unsigned char));
	CudaCheckError();

	
	cudaMalloc((void**)&d_grayImage, NUMPIX*sizeof(unsigned char));
	CudaCheckError();
	
	cudaMalloc((void**)&d_grayImage_32f, NUMPIX*sizeof(float));
	CudaCheckError();

	cudaMemset(d_grayImage, 0, sizeof(unsigned char)*NUMPIX);
	CudaCheckError();
	
	cudaMemset(d_grayImage_32f, 0, sizeof(float)*NUMPIX);
	CudaCheckError();

	cudaMemcpy(d_rgbImage, rgbImage, 3*sizeof(unsigned char)*NUMPIX,cudaMemcpyHostToDevice);
	CudaCheckError();
	
	
	dim3 blockSize(THREAD_X, THREAD_Y);
	dim3 gridSize((IMAGE_WIDTH + blockSize.x -1)/blockSize.x, (IMAGE_HEIGHT + blockSize.y -1)/blockSize.y);
	
	rgb2gray<<<gridSize, blockSize>>>(d_grayImage, d_rgbImage, d_grayImage_32f);
	cudaDeviceSynchronize();
	CudaCheckError();
	
	cudaMemcpy(grayImage, d_grayImage, sizeof(unsigned char)*NUMPIX,cudaMemcpyDeviceToHost);
	CudaCheckError();
	
	cudaMemcpy(h_grayImage_32f,d_grayImage_32f, sizeof(float)*NUMPIX, cudaMemcpyDeviceToHost);
	CudaCheckError();

	filterImage(d_grayImage_32f, 2, 2 ,2,10);

	

	if(debug)
	{
		Mat gray_image(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1);
		unsigned char* img = gray_image.data;
		for(int i =0;i<IMAGE_HEIGHT;i++)
		{
			for(int j = 0;j<IMAGE_WIDTH;j++)
			{
				*(img + i*IMAGE_WIDTH + j) = *(grayImage + i*IMAGE_WIDTH + j);

			}

		}
		
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

}










int main(int argc, char* argv[])
{

	Mat src_host;
	src_host = imread("/home/nvidia/Lane_Detection/Test_Images/IPM_test_image_0.png", CV_LOAD_IMAGE_COLOR);
		
	unsigned char* h_rgb_img = src_host.data;
	unsigned char* h_gray_img = (unsigned char*)malloc(NUMPIX*sizeof(unsigned char));
	float* h_grayImage_32f = (float*)malloc(NUMPIX*sizeof(float));
	convert2Gray(h_rgb_img, h_gray_img, h_grayImage_32f);
	

}
