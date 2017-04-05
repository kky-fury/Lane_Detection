#include"bev.hpp"

bool debug = false;

/*Util Functions*/
void print1dvector(row_t vector)
{
	for(const auto& elem: vector)
		cout<<setprecision(11)<<elem<<"\t";
	
	cout<<endl;

}

void print2dvector(matrix_t vector)
{
	for(const auto& row : vector)
	{
		for(const auto& elem : row)
		{
			cout<<setprecision(11)<<elem<<"\t"; 

		}
		cout<<endl;
	}

	cout<<endl;
}

/*Matrix Multiplication*/

matrix_t  matrix_multiplication(matrix_t const& vec_a, matrix_t  const& vec_b)
{
	int vec_a_rows = vec_a.size();
	int vec_a_columns = vec_a[0].size();

	int vec_b_rows = vec_b.size();
	int vec_b_columns  = vec_b[0].size();

	if(debug)
	{
		cout<<vec_a_rows<<"\t"<<vec_a_columns<<endl;
		cout<<vec_b_rows<<"\t"<<vec_b_columns<<endl;
	}
	
	matrix_t R2_1(vec_a_rows, row_t(vec_b_columns,0));
	
	for(int i =0;i<vec_a_rows;++i)
	{
		for(int j =0;j<vec_b_columns;++j)
		{
			R2_1[i][j] = 0;
			for(int k = 0;k<vec_a_columns;++k)
			{
				//R2_1.at(i).at(j) = R2_1.at(i).at(j) + (vec_a.at(i).at(k)*vec_b.at(k).at(j));
				R2_1[i][j] = R2_1[i][j] + (vec_a[i][k]*vec_b[k][j]);

			}
			

		}
	}
		
	return R2_1;	


}

/*Matrix Inverse*/

void getCofactor(matrix_t &vec_a, matrix_t &vec_b, int p,int q, int vec_a_rows)
{
	int i = 0, j = 0;

	for( int row = 0; row < vec_a_rows; row++)
	{
		for( int col = 0; col < vec_a_rows ; col++)
		{
			
			if( row != p && col !=q)
			{
				//vec_b.at(i).at(j++) = vec_a.at(row).at(col);
				vec_b[i][j++] = vec_a[row][col];
				if(j == vec_a_rows -1)
				{
					j = 0;
					i++;
				}

			}


		}

	}

}


double determinant(matrix_t &vec_a, int n)
{
	double D = 0.0;
	
	if(n==1)
		return (double) vec_a[0][0];
	
	matrix_t temp(4, row_t(4));
	//print2dvector(temp);
	int sign = 1;

	for( int f = 0;f< n;f++)
	{
		getCofactor(vec_a, temp,0,f,n);
		//D += sign*(vec_a.at(0).at(f))*(determinant(temp,n-1));
		D += sign*(vec_a[0][f])*(determinant(temp,n-1));	
		sign = -sign;

	}


	return D;


}



matrix_t adjoint(matrix_t &vec_a)
{
	int vec_a_rows = vec_a.size();
	int vec_a_columns = vec_a[0].size();

	int sign = 1;
	matrix_t temp(vec_a_rows, row_t(vec_a_columns,0));
	matrix_t adj(vec_a_rows, row_t(vec_a_rows,0));

	for(int i = 0; i<vec_a_rows;i++)
	{
		for(int j = 0;j<vec_a_columns;j++)
			{
					
				getCofactor(vec_a, temp, i, j, vec_a_rows);
				sign  = ((i+j)%2==0)? 1: -1;

				adj[j][i] = (sign)*(determinant(temp, vec_a_rows -1));
				
			}
	}

	return adj;

}


matrix_t inverse(matrix_t &vec_a)
{
	int vec_a_rows = vec_a.size();
	//cout<<vec_a_rows<<endl;
	
	double det = determinant(vec_a,vec_a_rows);
	
	//out<<setprecision(11)<<det<<endl;

	
	matrix_t adj(vec_a_rows, row_t(vec_a_rows,0));
	matrix_t inverse_matrix(vec_a_rows, row_t(vec_a_rows,0));

	adj = adjoint(vec_a);

	for(int i = 0;i<vec_a_rows;i++)
	{
		for( int j = 0;j<vec_a_rows;j++)
		{
			
			inverse_matrix[i][j] = adj[i][j]/det;

		}
	}	
	
	return inverse_matrix;

}





/*Implementation of Class Functions*/
/*Class BevParams Functions*/
BevParams::BevParams(float bev_res, tuple<int, int> bev_xLimits, tuple<int, int> bev_zLimits, tuple<int, int> imSize)
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
	
	matrix_t::iterator row,i;
	row_t::iterator column,j;


	for(row = (this->R0_Rect).begin(), i = R0_rect.begin() ; row != (this->R0_Rect).end() - 1;++row,++i)
	{
		for(column = row->begin(), j = i->begin(); column != row->end()-1;++column,++j)
		{
			*column = *j; 

		}

	}

	(this->R0_Rect)[3][3] = 1.0;

	if(debug)
	{
		print2dvector(this->P2);	
		print2dvector(this->R0_Rect);
	
	}
	
	matrix_t R2_1 = matrix_multiplication(this->P2, this->R0_Rect);


	Tr_cam_to_road.resize(4, row_t(4,0));
	Tr_cam_to_road[3][3] = 1;
	this->Tr_cam_to_road = Tr_cam_to_road;

	/*
	this->Tr  = matrix_multiplication(R2_1, this->Tr_cam_to_road);
	print2dvector(this->Tr);
	*/

	//inverse(this->Tr_cam_to_road);

	matrix_t Tr_cam_to_road_inverse	= inverse(this->Tr_cam_to_road);
	this->Tr = matrix_multiplication(R2_1, Tr_cam_to_road_inverse);


	unsigned columntoDelete = 1;
	for(unsigned i = 0;i<(this->Tr).size();++i)
	{
		if((this->Tr)[i].size() > columntoDelete)
		{
			(this->Tr)[i].erase((this->Tr)[i].begin() + columntoDelete);		

		}

	}

	if(debug)
	{
		print2dvector(this->Tr);
	}

	this->Tr33 = this->Tr;
	


}

matrix_t Calibration::get_matrix33()
{
	return this->Tr33;

}


/*Class BirdsEyeView Functions */


BirdsEyeView::BirdsEyeView(float bev_res, double invalid_value, tuple<int,int> bev_xRange_minMax, tuple<int,int> bev_zRange_minMax)
{

	this->e1 = getTickCount();
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
	this->set_matrix33((this->calib)->get_matrix33());	
	
}


void BirdsEyeView::set_matrix33(matrix_t Tr33)
{
	this->Tr33 = Tr33;

}

void BirdsEyeView::compute(const Mat& image)
{

	Size size = image.size();
	this->imSize = make_tuple(size.width, size.height);
	this->computeBEVLookUpTable();
	this->transformImage2BEV(image);
}

void BirdsEyeView::computeBEVLookUpTable()
{
	float res = (this->bevParams)->bev_res;
	
	int x_vec_length = ((get<1>((this->bevParams)->bev_xLimits)) - (get<0>((this->bevParams)->bev_xLimits) + res/2))/res + 1;
	int z_vec_length = ((get<1>((this->bevParams)->bev_zLimits) - res/2) - (get<0>((this->bevParams)->bev_zLimits)))/res + 1;

	cout<<"Size is \t"<<z_vec_length<<endl;
	
	double init_value_x =  get<0>((this->bevParams)->bev_xLimits) + res/2;
	double init_value_z = (get<1>((this->bevParams)->bev_zLimits) - res/2);
	cout<<"Initial_Value"<<init_value_z<<endl;

	row_t x_vec(x_vec_length), z_vec(z_vec_length);	
	for(int i = 0 ;i<x_vec_length;i++)
	{
		x_vec[i] = init_value_x;
		init_value_x += res;
		
	}
		
	
	for(int i = 0 ;i<z_vec_length;i++)
	{
		z_vec[i] = init_value_z;
		init_value_z -= res;
	}

	
	if(debug)
	{
		print1dvector(x_vec);
		print1dvector(z_vec);
		cout<<"Size Z_vec"<<z_vec.size()<<endl;
		cout<<"Size X_vec"<<x_vec.size()<<endl;
	}

	/*Populate Z and X mesh*/
	int size = get<0>((this->bevParams)->bev_size)*get<1>((this->bevParams)->bev_size);

	row_t z_mesh_vec(size), x_mesh_vec;
	
	for(int i = 0;i<size;i++)
	{
		int temp = i%z_vec_length;
		//z_mesh_vec.push_back(z_vec[temp]);	
		z_mesh_vec[i] = z_vec[temp];
	}
	

	int i = 0;
	while(i < x_vec_length)
	{	
		for(int j = 0;j< z_vec_length ;j++)
		{
			x_mesh_vec.push_back(x_vec[i]);
			//x_mesh_vec[j] 
		}
		i++;
	}
	
	double e1 = getTickCount();

	this->world2image(x_mesh_vec, z_mesh_vec);
	
	double e2 = getTickCount();
	double time = (e2 -e1)/getTickFrequency();
	cout<<"time world 2 image"<<time<<endl;
	/*Copy all values except ones which are infinity*/
	int length = this->xi_1.size();
	row_t x_select(length), y_select(length);
	
	
	auto it = copy_if((this->xi_1).begin(), (this->xi_1).end(), x_select.begin(), [] (double i){return !(i == -numeric_limits<double>::infinity()) ;});
	(x_select).resize(distance(x_select.begin(), it));


	auto it_1 =  copy_if((this->yi_1).begin(), (this->yi_1).end(), y_select.begin(), [] (double i){return !(i == -numeric_limits<double>::infinity()) ;});
	(this->im_v).resize(distance(y_select.begin(), it_1));

	if(debug)
	{
		print1dvector(x_select);
		print1dvector(y_select);
	}

	vector<int> z_index_vec, x_index_vec;
	
	vector<int> values_z(get<0>((this->bevParams)->bev_size));
	iota(values_z.begin(), values_z.end(), 1);

	//print1dvector(values);

	if(debug)
	{
		for(const auto& i : values_z)
			cout<<i<<"\t";
		cout<<endl;
	}

	for(int i = 0; i< x_vec_length ;i++)
	{
		z_index_vec.insert(z_index_vec.end(), values_z.begin(), values_z.end());

	}

	vector<int> values_x(get<1>((this->bevParams)->bev_size));
	iota(values_x.begin(), values_x.end(),1 );

	int index = 0;
	while(index < x_vec_length)
	{
		for(int i =0; i<z_vec_length ;i++)
			x_index_vec.push_back(values_x[index]);
		index++;	
	}

	vector<int> z_index_vec_sel, x_index_vec_sel;
	int size_counter = 0;
	
	for(int i = 0;i<(this->xi_1).size();i++)
	{
		if(!((this->xi_1)[i] == (this->invalid_value)))
		{
			//cout<<"index_value \t"<<i<<endl;
			//z_index_vec.erase(z_index_vec.begin() + i);		
			//cout<<"z_index_value at that index \t"<<z_index_vec[i]<<endl;
			z_index_vec_sel.push_back(z_index_vec[i]);
			//this->bev_z_index.push_back(z_index_vec[i]);
			x_index_vec_sel.push_back(x_index_vec[i]);
			//this->bev_x_index.push_back(x_index_vec[i]);
			size_counter++;
		}

	}

	this->im_u = x_select;
	this->im_v = y_select;
	this->bev_x_index = x_index_vec_sel;
	this->bev_z_index = z_index_vec_sel;


}

void BirdsEyeView::world2image(row_t x_world, row_t z_world)
{
	int size = x_world.size();
	row_t y_world(size, 1.0);	
	matrix_t uv_mat(3, row_t(size, 0));

	/*Populate UV Mat*/

	double e3 = getTickCount();	
	copy(x_world.begin(), x_world.end(), uv_mat[0].begin());
	copy(z_world.begin(), z_world.end(), uv_mat[1].begin());
	copy(y_world.begin(), y_world.end(), uv_mat[2].begin());
	double e4 = getTickCount();
	double time_copy =  (e4-e3)/getTickFrequency();
	cout<<"Time Taken for copy"<<time_copy<<endl;
	

	matrix_t test = this->world2image_uvMat(uv_mat);
	
	if(debug)
		print2dvector(test);

	(this->xi_1).resize(test[0].size());
	(this->yi_1).resize(test[1].size());

	copy(test[0].begin(), test[1].end(), (this->xi_1).begin());
	copy(test[1].begin(), test[1].end(), (this->yi_1).begin());

	if(debug)
	{
		print1dvector(this->xi_1);
		print1dvector(this->yi_1);
	}

	row_t::iterator i,j;

	double e1 = getTickCount();

	for(i = (this->xi_1).begin(), j = (this->yi_1).begin(); i<(this->xi_1).end(); i++, j++)
	{
		if(!((*j >=1) & (*i>=1) & (*j <= get<1>(this->imSize)) & (*i <= get<0>(this->imSize))))
		{
			//cout<<*i<<"\t"<<*j<<endl;
			*i = this->invalid_value;
			*j = this->invalid_value;
		}
			
	}
	

	double e2 = getTickCount();

	double time = (e2 -e1)/getTickFrequency();

	cout<<"Time for loop"<<time<<endl;

	if(debug)
	{
		print1dvector(this->xi_1);
		print1dvector(this->yi_1);
	}


}

matrix_t BirdsEyeView::world2image_uvMat(matrix_t& uvMat)
{
	double e1 = getTickCount();
	matrix_t result = matrix_multiplication(this->Tr33, uvMat);
	double e2 = getTickCount();
	double time = (e2 -e1)/getTickFrequency();
	cout<<"Time for Multiplication"<<time<<endl;
	if(debug)
	{
		//print2dvector(result);
		cout<<result.size()<<endl;
		cout<<result[0].size()<<endl;

	}

	/*Convert to non - homogeneous*/
	int rows = result.size();
	int columns = result[0].size();
	//row_t last_row(columns,0); 	

	//copy(result[2].begin(), result[2].end(), last_row.begin());
	matrix_t result_non_homogeneous(rows, row_t(columns, 0));
	

	/*divide each row of result by last_row and store in result_non_homogeneous*/
	e1 = getTickCount();
	transform(result[0].begin(), result[0].end(), result[2].begin(), result_non_homogeneous[0].begin(), divides<double>());
	transform(result[1].begin(), result[1].end(), result[2].begin(), result_non_homogeneous[1].begin(), divides<double>());
	transform(result[2].begin(), result[2].end(), result[2].begin(), result_non_homogeneous[2].begin(), divides<double>());
	e2 = getTickCount();
	time = (e2 -e1)/getTickFrequency();
	cout<<"Time for Transform Operations"<<time<<endl;
	return result_non_homogeneous;


}

void BirdsEyeView::transformImage2BEV(const Mat& image)
{
	double e3,e4;
	e3 = getTickCount();
	row_t::const_iterator i,j;
	//cout<<get<0>((this->bevParams)->bev_size)<<endl;
	Mat output_image(get<0>((this->bevParams)->bev_size), get<1>((this->bevParams)->bev_size),CV_8UC1);
	vector<int>::const_iterator m,k;
	
	unsigned char* i_im = image.data;
	unsigned char* o_im = output_image.data; 
	
	for(i = (this->im_u).begin(), j = (this->im_v).begin(), m = (this->bev_x_index).begin(), k = (this->bev_z_index).begin(); i != (this->im_u).end(); i++,j++,m++,k++)
	{
		int row = (int)*j -1;
		int column = (int)*i -1;

		int row_output_image  = (int)*k -1;
		int column_output_image = (int)*m -1;	
		*(o_im + row_output_image*200 + column_output_image) = *(i_im + row*1242 + column);

	}
	
	if(debug)
	{
		cout<<(this->im_u).size();
		cout<<(this->im_v).size();
		cout<<(this->bev_z_index).size();
		cout<<(this->bev_x_index).size();
	}
	this->e2  = getTickCount();
	double time = (e2 -e1)/getTickFrequency();
	double time_for_generating_op = (e2 -e3)/getTickFrequency();
	cout<<"Time in sec \t"<<time<<endl;
	imshow("result", output_image);
	waitKey(0);



}




























int main(int argc, char* argv[])
{


	Mat test_image = imread("/home/mohak/Downloads/Lane_Detection-master/Original_Images/img_0.png", CV_LOAD_IMAGE_GRAYSCALE);

	if(debug)
	{	
		imshow("Test_image", test_image);
		waitKey(0);
	}

	/*define Parameters*/
	float bev_res = 0.1;
	
	tuple<int, int> bev_xRange_minMax(make_tuple(-10,10));
	tuple<int, int> bev_zRange_minMax(make_tuple(6, 46));
	double invalid_value = -numeric_limits<double>::infinity();

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
		{9.999044710077e-01,-1.170165577363e-02, -7.360826724365e-03, 1.911984983337e-02},
		{1.160251828357e-02,9.998432738993e-01,-1.336987011872e-02,-1.562198078590e+00},
		{7.516122576373e-03, 1.328318612284e-02, 9.998834806284e-01,2.752775890648e-01}
	};


	bev.setup(P2, R0_rect, Tr_cam_to_road);
	bev.compute(test_image);

}

