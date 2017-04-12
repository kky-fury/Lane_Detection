#ifndef PREPROCESS_HPP
#define PREPROCESS_HPP

#include<iostream>
#include<cuda_runtime_api.h>
#include<cuda.h>
#include<vector>
#include<math.h>
#include<cstring>
#include<limits>
#include<tuple>
#include <iomanip>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"

#define IMAGE_WIDTH 200
#define IMAGE_HEIGHT 400
#define NUMPIX	(IMAGE_WIDTH*IMAGE_HEIGHT)
#define THREAD_X 16
#define THREAD_Y 16

#define CU_FILTER_WIDTH (5)


using namespace std;
using namespace cv;
	

void  matrix_multiplication(float* arr1, int arr1_rows, int arr1_cols, float* arr2, int arr2_rows, int arr2_cols, float* r_arr);
void find_min_max(thrust::device_ptr<float> &dbeg, thrust::device_ptr<float> &dend, float* min, float* max);

__global__ void rgb2gray(unsigned char* d_grayImage, const unsigned char* const d_rgbImage, float* d_grayImage_32f);
__global__ void gaussian_blur_tiled(const float* const grayImage, float* filteredImage, int numRows, int numCols, const float* const filter,const int filterWidth);

void convert2Gray(const unsigned char* const rgbImage, unsigned char* grayImage, float* h_grayImage_32f);
void filterImage(const float* const grayImage, int width_kernel_x, int width_kernel_y,float sigmax, float sigmay);




















#endif
