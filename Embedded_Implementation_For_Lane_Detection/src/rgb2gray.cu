#include"rgb2gray.hpp"
#include"cuda_error_check.hpp"

__global__ void rgb_2_gray(unsigned char* d_grayImage, const unsigned char* const d_rgbImage)
{

	int rgb_x = blockIdx.x*blockDim.x + threadIdx.x;
	int rgb_y = blockIdx.y*blockDim.y + threadIdx.y;

	if((rgb_x >= IMAGE_WIDTH_RGB) || (rgb_y >=IMAGE_HEIGHT_RGB))
	{
		return;
	}

	unsigned char blue = float(*(d_rgbImage + 3*IMAGE_WIDTH_RGB*rgb_y + 3*rgb_x))*0.114f;
	unsigned char green = float(*(d_rgbImage + 3*IMAGE_WIDTH_RGB*rgb_y + 3*rgb_x +1))*0.587f;
	unsigned char red = float(*(d_rgbImage + 3*IMAGE_WIDTH_RGB*rgb_y + 3*rgb_x +2))*0.299f;
	
	*(d_grayImage + rgb_y*IMAGE_WIDTH_RGB + rgb_x) = uchar(blue +  green + red);


}
















unsigned char* rgb2gray(const unsigned char* const rgbImage)
{

	unsigned char* h_grayImage = (unsigned char*)malloc(NUMPIX_RGB*sizeof(unsigned char));
	unsigned char* d_rgbImage;
	unsigned char* d_grayImage;

	cudaMalloc((void**)&d_rgbImage, 3*NUMPIX_RGB*sizeof(unsigned char));
	CudaCheckError();

	cudaMalloc((void**)&d_grayImage, NUMPIX_RGB*sizeof(unsigned char));
	CudaCheckError();

	cudaMemset(d_grayImage, 0, sizeof(unsigned char)*NUMPIX_RGB);
	CudaCheckError();

	cudaMemcpy(d_rgbImage, rgbImage, 3*sizeof(unsigned char)*NUMPIX_RGB,cudaMemcpyHostToDevice);
	CudaCheckError();

	dim3 blockSize(THREAD_X_RGB, THREAD_Y_RGB);
	dim3 gridSize((IMAGE_WIDTH_RGB + blockSize.x -1)/blockSize.x, (IMAGE_HEIGHT_RGB +blockSize.y -1)/blockSize.y);

	rgb_2_gray<<<gridSize, blockSize>>>(d_grayImage, d_rgbImage);
	cudaDeviceSynchronize();
	CudaCheckError();

	cudaMemcpy(h_grayImage, d_grayImage, sizeof(unsigned char)*NUMPIX_RGB,cudaMemcpyDeviceToHost);
	CudaCheckError();

	return h_grayImage;


}



























