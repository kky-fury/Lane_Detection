#ifndef BEV_HPP
#define BEV_HPP

#include<iostream>
#include<cuda_runtime_api.h>
#include<cuda.h>
#include<vector>
#include<math.h>
#include<cstring>
#include<limits>
#include<tuple>
#include <iomanip> 
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"


#define IMAGE_WIDTH	1242
#define IMAGE_HEIGHT 375


using namespace std;
using namespace cv;
/*Declaring a 2D Vector of double*/

typedef vector<double> row_t;
typedef vector<row_t> matrix_t;

void print1dvector(row_t vector);
void print2dvector(matrix_t vector);

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
		tuple<int, int> bev_size;
		float bev_res;
		tuple<int, int> bev_xLimits;
		tuple<int, int> bev_zLimits;
		tuple<int,int> imSize;
		/*Constructor*/
		BevParams(float bev_res, tuple<int, int> bev_xLimits, tuple<int, int> bev_zLimits, tuple<int, int> imSize);




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
		tuple<int, int> imSize; /*RGB Image*/
		BevParams* bevParams;
		Calibration* calib;
		double invalid_value;
		float bev_res;
		tuple<int, int> bev_xRange_minMax;
		tuple<int, int> bev_zRange_minMax;
		matrix_t Tr33;
		row_t xi_1;
		row_t yi_1;

		/*Constructor*/		
		BirdsEyeView(float bev_res,double invalid_value, tuple<int, int> bev_xRange_minMax, tuple<int, int> bev_zRange_minMax);
		void setup(matrix_t P2, matrix_t R0_Rect, matrix_t Tr_cam_to_road);
		void set_matrix33(matrix_t Tr33);
		void compute(const Mat& image);
		void computeBEVLookUpTable();
		void world2image(row_t x_world, row_t z_world);
		matrix_t world2image_uvMat(matrix_t& uvMat);

};












#endif
