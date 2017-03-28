#include<iostream>
#include<cstdio>
#include"opencv2/opencv.hpp"
#include"opencv2/highgui.hpp"
#include"opencv2/gpu/gpu.hpp"
#include<cuda_runtime_api.h>
#include<cuda.h>
#include<vector>
#include<math.h>
#include<cstring>
#include<string>

#define IMG_WIDTH	192
#define IMG_HEIGHT 224
#define IMG_SIZE (IMG_WIDTH*IMG_HEIGHT)

#define HS_STEP		1
#define HS_ANGLES	(90/HS_STEP + 1)
#define HS_1_WIDTH	(IMG_HEIGHT + IMG_WIDTH + IMG_HEIGHT)
#define HS_1_SIZE	(HS_1_WIDTH * HS_ANGLES)
#define HS_2_WIDTH	(IMG_WIDTH + IMG_HEIGHT + IMG_WIDTH)
#define HS_2_SIZE	(HS_2_WIDTH * HS_ANGLES)


/*defintion to expand macros*/
#define STR(x)   #x
#define SHOW_DEFINE(x) printf("%s=%s\n", #x, STR(x))


using namespace std;
using namespace cv;

/*Function Definitions*/

void print_array(float *, int);
void print_image(unsigned char *image, int height, int width);

__global__ void getNonzeroEdgepoints(unsigned char const* const,unsigned int* const);

__global__ void getLines(unsigned int* const, const int, int*, const float, const float theta, const int numrho );
//__global__ void Hough(unsigned char const* const, unsigned int const, unsigned int* const, unsigned int* const);

//void houghTransform(unsigned char const * const,  unsigned int const,unsigned int* const,unsigned int* const );

void houghTransform(unsigned char const* const, const int, const int, float, float);

void print_houghspace(unsigned int* const array, int width);

int getMaximum(unsigned int* const array, int width);
