#ifndef RGB2GRAY_HPP
#define RGB2GRAY_HPP

#include<iostream>
#include<cuda_runtime_api.h>
#include<cuda.h>

#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"

#define IMAGE_WIDTH_RGB 1242
#define IMAGE_HEIGHT_RGB 375
#define NUMPIX_RGB IMAGE_WIDTH_RGB*IMAGE_HEIGHT_RGB
#define THREAD_X_RGB 16
#define THREAD_Y_RGB 16

using namespace std;
using namespace cv;



__global__ void rgb_2_gray(unsigned char* d_grayImage, const unsigned char* const d_rgbImage);
unsigned char* rgb2gray(const unsigned char* const rgbImage);




#endif
