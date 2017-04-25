#include"bev_thrust.hpp"

bool debug_bev = false;


/*Functor For Stream Compaction*/

/*
struct checkSanity
{
	__host__ __device__
		bool operator()(tuple_t x)
		{
			return(thrust::get<0>(x) >= 1 && thrust::get<1>(x) >=1 &&
					thrust::get<0>(x) <= IMAGE_WIDTH && thrust::get<1>(x) <= IMAGE_HEIGHT);
		}

};
*/

/*
struct checkSanity
{
	__host__ __device__
		bool operator()(const float x)
		{
			return (x >=  1 && x <= IMAGE_WIDTH);
		}


};
*/
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



matrix_t  matrix_multiplication(matrix_t const& vec_a, matrix_t  const& vec_b)
{
	int vec_a_rows = vec_a.size();
	int vec_a_columns = vec_a[0].size();
	
	int vec_b_rows = vec_b.size();
	int vec_b_columns  = vec_b[0].size();
			
	if(debug_bev)
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
				R2_1[i][j] = R2_1[i][j] + (vec_a[i][k]*vec_b[k][j]);
				
			}
		}			
	}
	return R2_1;

}

void getCofactor(matrix_t &vec_a, matrix_t &vec_b, int p,int q, int vec_a_rows)
{
	int i = 0, j = 0;
	for( int row = 0; row < vec_a_rows; row++)
	{
		for( int col = 0; col < vec_a_rows ; col++)	
		{
			if( row != p && col !=q)
			{
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
 	
	double det = determinant(vec_a,vec_a_rows);
	
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


__global__ void matrix_mul(float* d_A, float* d_B, float* d_C, int numARows, int numAColumns, int numBRows, int numBColumns,
		int numCRows, int numCColumns)
{
	__shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
 	__shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];	

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int Row = by*TILE_WIDTH + ty;
	int Col = bx*TILE_WIDTH + tx;

	float Pvalue = 0;

	#pragma unroll
	for (int m = 0; m < (numAColumns-1)/TILE_WIDTH+1; ++m)
	{
		if (Row < numARows && m*TILE_WIDTH+tx < numAColumns)
		{
			ds_M[ty][tx] = d_A[Row*numAColumns + m*TILE_WIDTH+tx];
	
		}
		else
		{
			ds_M[ty][tx] = 0;

		}

		if (Col < numBColumns && m*TILE_WIDTH+ty < numBRows)
		{
			ds_N[ty][tx] = d_B[(m*TILE_WIDTH+ty)*numBColumns+Col];
		}
		else
		{
			ds_N[ty][tx] = 0;
		}

		__syncthreads();

		for (int k = 0; k < TILE_WIDTH; ++k)
		{
			Pvalue += ds_M[ty][k] * ds_N[k][tx];
		}

		__syncthreads();
	
	}

	if (Row < numCRows && Col < numCColumns)
		d_C[Row*numCColumns+Col] = Pvalue;

}


float* getMatrix(matrix_t Tr33, float* h_B, int numBRows, int numBColumns)
{
	//double e1 = getTickCount();

	float* h_A;
	float* h_C; 
	float* d_A;
	float* d_B;
	float* d_C;

	int numARows = Tr33.size();
	int numAColumns = Tr33[0].size();

	int numCRows = numARows;
	int numCColumns = numBColumns;

	h_A = (float*)malloc(numARows*numAColumns*sizeof(float));

	/*Populate h_A and h_B */
	for(int i = 0 ;i<numARows;i++)
	{	
		for(int j =0;j<numAColumns;j++)
		{
			*(h_A + i*numAColumns + j) = Tr33[i][j];

		}

	}
	
	h_C = (float*)malloc(numCRows*numCColumns*sizeof(float));

	cudaMalloc((void**)&d_A, sizeof(float) * numARows * numAColumns);
	cudaMalloc((void**)&d_B, sizeof(float) * numBRows * numBColumns);
	cudaMalloc((void**)&d_C, sizeof(float) * numCRows * numCColumns);

	cudaMemcpy(d_A, h_A, sizeof(float) * numARows * numAColumns, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, sizeof(float) * numBRows * numBColumns, cudaMemcpyHostToDevice);

	int dim_grid_x = (numCColumns - 1)/TILE_WIDTH +1;
	int dim_grid_y = (numCRows -1)/TILE_WIDTH + 1;


	dim3 dimGrid(dim_grid_x, dim_grid_y);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

	matrix_mul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
	cudaDeviceSynchronize();
	cudaMemcpy(h_C, d_C, sizeof(float) * numCRows * numCColumns, cudaMemcpyDeviceToHost);

	//cudaDeviceSynchronize();

	if(debug_bev)
	{
		for(int i = 0 ;i<numCRows;i++)
		{
			for(int j  = 0;j<numCColumns;j++)
			{
				cout<<*(h_C + i*numCRows + j)<<"\t";
			}
		cout<<endl;
		}
	}

	float* row_ptr_0 = h_C;
	float* row_ptr_1 = h_C + numCColumns;
	float* row_ptr_2 = h_C + 2*numCColumns;

	for(int i =0;i<numCColumns;i++)
	{
		*(row_ptr_0 +  i) =  *(row_ptr_0 + i)/(*(row_ptr_2 + i));
		*(row_ptr_1 + i) = *(row_ptr_1 + i)/(*(row_ptr_2 + i));
		*(row_ptr_2 + i) = *(row_ptr_2 + i)/(*(row_ptr_2 + i));

	}


	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	
	return h_C;



}

BevParams::BevParams(float bev_res, tuple_int bev_xLimits, tuple_int bev_zLimits, tuple_int imSize)
{
	
	this->bev_size = {static_cast<int>(std::round((bev_zLimits.b - bev_zLimits.a)/bev_res)), static_cast<int>(std::round((bev_xLimits.b - bev_xLimits.a)/bev_res))};	
	this->bev_res = bev_res;
	this->bev_xLimits = bev_xLimits;
	this->bev_zLimits = bev_zLimits;
	this->imSize = imSize;

};






Calibration::Calibration()
{
	


};


void Calibration::setup_calib(matrix_t P2, matrix_t R0_rect, matrix_t Tr_cam_to_road)
{
	this->P2 = P2;
	(this->R0_Rect).resize(4,row_t(4,0));
	
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
	
	matrix_t R2_1 = matrix_multiplication(this->P2, this->R0_Rect);
	
	Tr_cam_to_road.resize(4, row_t(4,0));
	Tr_cam_to_road[3][3] = 1;
	this->Tr_cam_to_road = Tr_cam_to_road;

	matrix_t Tr_cam_to_road_inverse = inverse(this->Tr_cam_to_road);
	this->Tr = matrix_multiplication(R2_1, Tr_cam_to_road_inverse);
	
	unsigned columntoDelete = 1;

	for(unsigned i = 0;i<(this->Tr).size();++i)
	{
		if((this->Tr)[i].size() > columntoDelete)
		{
			(this->Tr)[i].erase((this->Tr)[i].begin() + columntoDelete);
		}


	}
	this->Tr33 = this->Tr;

}



matrix_t Calibration::get_matrix33()
{   
	return this->Tr33;

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

void BirdsEyeView::initialize()
{
	
	this->imSize = {IMAGE_WIDTH_GRAY, IMAGE_HEIGHT_GRAY};

	float res = (this->bevParams)->bev_res;

	int x_vec_length = ((this->bevParams)->bev_xLimits.b - ((this->bevParams)->bev_xLimits.a + res/2))/res + 1;
	int z_vec_length = ((this->bevParams)->bev_zLimits.b - res/2 - (this->bevParams)->bev_zLimits.a)/res + 1;
		
	double init_value_x = (this->bevParams)->bev_xLimits.a + res/2;
	double init_value_z = (this->bevParams)->bev_zLimits.b - res/2;

	
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

	
	int vec_size = (this->bevParams)->bev_size.a*(this->bevParams)->bev_size.b;
	row_t z_mesh_vec(vec_size), x_mesh_vec;
	
	for(int i = 0;i<vec_size;i++)
	{
		int temp = i%z_vec_length;
		z_mesh_vec[i] = z_vec[temp];
	}
		int i = 0;
	while(i < x_vec_length)
	{
		for(int j = 0;j< z_vec_length ;j++)
		{
			x_mesh_vec.push_back(x_vec[i]);
			
		}
		i++;
	}
	
	row_t y_world(vec_size, 1.0);
	matrix_t uv_mat(3, row_t(vec_size, 0));

	row_t::iterator row_x, row_z, row_y, row_uv_0, row_uv_1,row_uv_2;
	
	//double e3 = getTickCount();
	
	for(row_x = x_mesh_vec.begin(), row_z = z_mesh_vec.begin(), 
		row_y  = y_world.begin(), row_uv_0 = uv_mat[0].begin(), row_uv_1 = uv_mat[1].begin(),
		row_uv_2 = uv_mat[2].begin(); row_x != x_mesh_vec.end(); row_x++, row_z++, row_y++, row_uv_0++, row_uv_1++, row_uv_2++)
	{
		*row_uv_0 = *row_x;
		*row_uv_1 = *row_z;
		*row_uv_2 = *row_y;
	}
	

	this->uvMat = uv_mat;
	
	this->numBRows = uv_mat.size();
	this->numBColumns = uv_mat[0].size();

	this->h_B = (float*)malloc(numBRows*numBColumns*sizeof(float));

	for(int i = 0;i<numBRows;i++)
	{
		for(int j =0;j<numBColumns;j++)
		{
			*(this->h_B + i*numBColumns + j) = this->uvMat[i][j]; 

		}
	}


	vector<int> values_z((this->bevParams)->bev_size.a);
	iota(values_z.begin(), values_z.end(), 1);

	for(int i = 0; i< x_vec_length ;i++)
	{
		(this->z_index_vec).insert(z_index_vec.end(), values_z.begin(),
		values_z.end());
	
	}

	vector<int> values_x((this->bevParams)->bev_size.b);
	iota(values_x.begin(), values_x.end(),1 );

	int index = 0;
		
	while(index < x_vec_length)
	{
		for(int i =0; i<z_vec_length ;i++)
			(this->x_index_vec).push_back(values_x[index]);
		index++;
	}

	
	if(debug_bev)
	{
		for(const auto& i : x_index_vec)
			cout<<i<<"\t";
    	cout<<endl;
	}

	float* d_B;
	cudaMalloc((void**)&d_B, sizeof(float)*this->numBRows*this->numBColumns);
	cudaFree(d_B);

	s_h_B = (float*)malloc(numBRows*numBColumns*sizeof(float));
	s_h_B = this->h_B;

}


float* BirdsEyeView::s_h_B = NULL;




BirdsEyeView::BirdsEyeView(float bev_res, double invalid_value, tuple_int bev_xRange_minMax, tuple_int bev_zRange_minMax)
{			
		
	this->calib = new Calibration();
	this->bev_res = bev_res;
	this->invalid_value = invalid_value;
	this->bev_xRange_minMax = bev_xRange_minMax;
	this->bev_zRange_minMax = bev_zRange_minMax;
	this->bevParams = new BevParams(bev_res, bev_xRange_minMax, bev_zRange_minMax, this->imSize);
		
}

unsigned char* BirdsEyeView::computeLookUpTable(unsigned char* image)
{

	float* result = getMatrix(this->Tr33, getWorld(),this->numBRows, this->numBColumns);
	
	if(debug_bev)
	{
		for(int i =0 ;i<3;i++)
		{
			for(int j  = 0;j<80000;j++)
			{
				cout<<*(result  + i*80000 + j)<<"\t";
			}
			
			cout<<endl;
		}
	}
	
	int numCol = this->numBColumns;

	row_t xi_1(numCol), yi_1(numCol);

	float* result_row_0 = result;
	float* result_row_1 = result + 1*numCol;
	
	vector<int> x_index_vec_copy = this->x_index_vec;
	vector<int> z_index_vec_copy = this->z_index_vec;
	
	vector<int> z_vec_sel(numCol), x_vec_sel(numCol);

	int count  = 0;

	for(int i =0;i<numCol;i++)
	{
		if((*(result_row_1 + i) >=1) & (*(result_row_0 +i) >=1) & (*(result_row_1 + i) <= this->imSize.b) & (*(result_row_0 + i) <= this->imSize.a))
		{
			xi_1[count] = *(result_row_0 + i);
			yi_1[count] = *(result_row_1 + i);
			z_vec_sel[count] = z_index_vec_copy[i];
			x_vec_sel[count] = x_index_vec_copy[i];
			count++;	
		}

	}

	z_vec_sel.resize(count);
	x_vec_sel.resize(count);
	xi_1.resize(count);
	yi_1.resize(count);
	

	if(debug_bev)
	{
		for(const auto& i: xi_1)
			cout<<i<<"\t";
		cout<<endl;
	}
	
	if(debug_bev)
	{
		for(const auto& i : z_vec_sel)
			cout<<i<<"\t";
    	cout<<endl;
	}
	
	
	
	vector<int>::const_iterator m,k;
	row_t::const_iterator i,j;

	unsigned char* i_im = image;

	unsigned char* o_im  = (unsigned char*)malloc((this->bevParams)->bev_size.a*(this->bevParams)->bev_size.b);

	for(i = xi_1.begin() , j = yi_1.begin(), m = x_vec_sel.begin(), k = z_vec_sel.begin();i != xi_1.end();i++,j++,m++,k++)
	{
		int row = (int)*j -1;
		int column = (int)*i -1;

		int row_output_image = (int)*k -1;
		int column_output_image = (int)*m -1;
	
		//cout<<"Row \t"<<row<<"Column \t"<<column<<endl;

		*(o_im + row_output_image*200 + column_output_image) = *(i_im + row*IMAGE_WIDTH_GRAY + column);

	}
	
	return o_im;
}





