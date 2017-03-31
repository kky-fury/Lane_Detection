#include"bev.hpp"

bool debug = false;

/*Util Functions*/

void print2dvector(matrix_t vector)
{
	for(const auto& row : vector)
	{
		for(const auto& elem : row)
		{
			cout<<elem<<"\t"; 

		}
		cout<<endl;
	}



}

/*Matrix Multiplication*/

matrix_t  matrix_multiplication(matrix_t const& vec_a, matrix_t  const& vec_b)
{



}

/*Implementation of Class Functions*/
/*Class BevParams Functions*/
BevParams::BevParams(float bev_res, tuple<int, int> bev_xLimits, tuple<int, int> bev_zLimits, int imSize)
{

	this->bev_size = make_tuple(round((get<1>(bev_zLimits) - get<0>(bev_zLimits))/bev_res),round((get<1>(bev_xLimits) -get<0>(bev_xLimits))/bev_res));
	this->bev_res = bev_res;
	this->bev_xLimits = bev_xLimits;
	this->bev_zLimits = bev_zLimits;
	this->imSize = imSize;

}

/*Class Calibration Functions*/

Calibration::Calibration()
{

}

void Calibration::setup_calib(matrix_t P2, matrix_t R0_rect, matrix_t Tr_cam_to_road)
{
	this->P2 = P2;
	//matrix_t R0_rect_raw(4, row_t(4,0));
	(this->R0_Rect).resize(4,row_t(4,0));

	if(debug)
	{
		print2dvector(this->P2);
	
	}
	
/*
	R0_rect_raw = R0_Rect;
	print2dvector(R0_rect_raw);	

*/
	matrix_t::iterator row,i;
	row_t::iterator column,j;


	for(row = (this->R0_Rect).begin(), i = R0_rect.begin() ; row != (this->R0_Rect).end() - 1;++row,++i)
	{
		for(column = row->begin(), j = i->begin(); column != row->end()-1;++column,++j)
		{
			*column = *j; 

		}

	}

	(this->R0_Rect).at(3).at(3) = 1.0;

	if(debug)
	{
		print2dvector(this->P2);	
		print2dvector(this->R0_Rect);
	
	}




}


/*Class BirdsEyeView Functions */


BirdsEyeView::BirdsEyeView(float bev_res, float invalid_value, tuple<int,int> bev_xRange_minMax, tuple<int,int> bev_zRange_minMax)
{

	this->calib = new Calibration();
	this->bev_res = bev_res;
	this->invalid_value = invalid_value;
	this->bev_xRange_minMax = bev_xRange_minMax;
	this->bev_zRange_minMax = bev_zRange_minMax;
	this->bevParams = new BevParams(bev_res, bev_xRange_minMax, bev_zRange_minMax, this->imSize);

}

void  BirdsEyeView::setup(matrix_t P2, matrix_t R0_rect, matrix_t Tr_cam_to_road)
{
	(this->calib)->setup_calib(P2, R0_rect, Tr_cam_to_road);

}











































int main(int argc, char* argv[])
{

	/*define Parameters*/
	float bev_res = 0.1;
	tuple<int, int> bev_xRange_minMax(make_tuple(-10,10));
	tuple<int, int> bev_zRange_minMax(make_tuple(6, 46));
	float invalid_value = numeric_limits<float>::infinity();

	BirdsEyeView bev(bev_res, invalid_value,bev_xRange_minMax, bev_zRange_minMax);

	if(debug)
	{
		cout<<bev.bev_res<<endl;
		cout<<"Accessing Object of BevParams \t"<<(bev.bevParams)->bev_res<<endl;
	}


	/*Projection matrix for left color camera in rectified coordinates*/
	/*3x4*/
	matrix_t P2
	{
		{7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02, 4.485728000000e+01},
		{0.000000000000e+00 ,7.215377000000e+02 ,1.728540000000e+02 ,2.163791000000e-01},
		{0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 2.745884000000e-03}

	};

	/*Initialize Rotation Matrix (3x3) */
	matrix_t R0_rect
	{
		{9.999239000000e-01, 9.837760000000e-03, -7.445048000000e-03},
		{-9.869795000000e-03, 9.999421000000e-01,-4.278459000000e-03},
		{7.402527000000e-03, 4.351614000000e-03, 9.999631000000e-01}
	};

	/*Rigid transformation from (non-rectified) camera to road coordinates (3x4)*/
	matrix_t Tr_cam_to_road
	{	
		{9.998675805558e-01, -1.466259288355e-02, -7.059878200710e-03, 2.879062998184e-02},
		{1.469236542096e-02, 9.998832652489e-01, 4.183808189280e-03, -1.630891383620e+00},
		{6.997709379545e-03, -4.286980067163e-03, 9.999662905842e-01,3.368200142169e-01},

	};


	bev.setup(P2, R0_rect, Tr_cam_to_road);

}

