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

using namespace std;

/*Declaring a 2D Vector of double*/

typedef vector<double> row_t;
typedef vector<row_t> matrix_t;

void print2dvector(matrix_t vector);



class BevParams
{
	public:
		tuple<int, int> bev_size;
		float bev_res;
		tuple<int, int> bev_xLimits;
		tuple<int, int> bev_zLimits;
		int imSize;
		/*Constructor*/
		BevParams(float bev_res, tuple<int, int> bev_xLimits, tuple<int, int> bev_zLimits, int imSize);




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

	


};





class BirdsEyeView
{
	public:
		int imSize;
		BevParams* bevParams;
		Calibration* calib;
		float invalid_value;
		float bev_res;
		tuple<int, int> bev_xRange_minMax;
		tuple<int, int> bev_zRange_minMax;
	
		/*Constructor*/		
		BirdsEyeView(float bev_res,float invalid_value, tuple<int, int> bev_xRange_minMax, tuple<int, int> bev_zRange_minMax);
		void setup(matrix_t P2, matrix_t R0_Rect, matrix_t Tr_cam_to_road);

};












#endif
