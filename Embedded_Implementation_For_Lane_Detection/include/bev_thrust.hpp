#ifndef BEV_THRUST_HPP
#define BEV_THRUST_HPP

#include<iostream>
#include<cuda_runtime_api.h>
#include<cuda.h>
#include<vector>
#include<math.h>
#include<cstring>
#include<limits>
#include <iomanip>



/*Thrust Definitions*/

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/copy.h>

#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;
using namespace thrust;

#define TILE_WIDTH 2
#define IMAGE_WIDTH 1242
#define IMAGE_HEIGHT 375

typedef vector<float> row_t;
typedef vector<row_t> matrix_t;

typedef host_vector<float> h_row_t;
typedef host_vector<float> h_matrix_t;

typedef device_vector<float> d_row_t;
typedef device_vector<float> d_matrix_t;

typedef thrust::tuple<float, float> tuple_t;
typedef thrust::device_vector<float>::iterator floatIterator;
typedef thrust::tuple<floatIterator, floatIterator> floatIteratorTuple;
typedef thrust::zip_iterator<floatIteratorTuple> zipIterator;
typedef thrust::device_vector<tuple_t>::iterator tupleIterator;



typedef struct coord{

	int a;
	int b;
}tuple_int;

void print1dvector(row_t vector);
void print2dvector(matrix_t vector);

__global__ void matrix_mul(float* d_A, float* d_B, float* d_C, int numARows, int numAColumns, int numBRows, int numBColumns,
		int numCRows, int numCColumns);

float* getMatrix(matrix_t, float*, int, int);

/*Function to fo matrix_multiplication*/
matrix_t matrix_multiplication(matrix_t const& vec_a, matrix_t const& vec_b);

/*Functions to obtain inverse of a matrix*/
void getCofactor(matrix_t &vec_a, matrix_t &vec_b, int p, int q, int vec_a_rows);

double determinant(matrix_t &vec_a, int n);
matrix_t adjoint(matrix_t &vec_a);
matrix_t inverse(matrix_t &vec_a);


class BevParams
{   
	public:
		tuple_int bev_size;
		float bev_res;
		tuple_int bev_xLimits;
		tuple_int bev_zLimits;
		tuple_int imSize;
		/*Constructor*/ 
		BevParams(float bev_res, tuple_int bev_xLimits, tuple_int bev_zLimits, tuple_int imSize);

};


class Calibration
{
	public:
		matrix_t P2;
		matrix_t R0_Rect;
		matrix_t Tr_cam_to_road;
		matrix_t Tr33;
		matrix_t Tr;
		/*Constructor*/
		Calibration();
		void setup_calib(matrix_t P2, matrix_t R0_Rect, matrix_t Tr_cam_to_road);
		matrix_t get_matrix33();

};




class BirdsEyeView
{	  
	public:
		tuple_int imSize; /*RGB Image*/
		BevParams* bevParams;	
		Calibration* calib;
		float invalid_value;
		float bev_res;
		tuple_int bev_xRange_minMax;
		tuple_int bev_zRange_minMax;
		matrix_t Tr33;
		matrix_t uvMat;
		float* h_B;
		int numBRows;
		int numBColumns;
		void computeLookUpTable();	
		float* xi_1;
		float* yi_1;
		vector<int> z_index_vec;
		vector<int> x_index_vec;


		BirdsEyeView(float bev_res,double invalid_value, tuple_int bev_xRange_minMax, tuple_int bev_zRange_minMax);
		void setup(matrix_t P2, matrix_t R0_Rect, matrix_t Tr_cam_to_road);
		void set_matrix33(matrix_t Tr33);
		void initialize(Mat& image);


};

#endif
