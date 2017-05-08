#include"line.hpp"
#include"spline.hpp"

bool debug_spline = false;

void  matrix_multiplication_spline(float* arr1, int arr1_rows, int arr1_cols, float* arr2, int arr2_rows, int arr2_cols, float* r_arr)
{   
	for(int i = 0;i<arr1_rows;i++)
	{	
		for(int j = 0;j < arr2_cols;j++)
		{
			*(r_arr + i*arr2_cols + j) = 0;
			for(int k =0;k<arr1_cols;k++)
			{   
				*(r_arr + i*arr2_cols + j) = *(r_arr + i*arr2_cols +j) + (*(arr1+ i*arr1_cols + k))*(*(arr2 + k*arr2_cols + j));        
			}
	
		}
	}

}








void getRansacSplines(vector<Line>& line_objects, vector<Spline>& spline_objects, Mat& gray_ipm_image)
{
	for(int i  = 0;i<line_objects.size();i++)
	{
		
		Spline spline = getLine2Spline(line_objects[i], DEGREE);
		spline_objects[i] = spline;		
		
		
	}

	//fitSpline(spline_objects);
	drawSpline(gray_ipm_image, spline_objects);

}


void drawSpline(Mat& gray_ipm_image, vector<Spline>& spline_objects)
{
	for(int i =0;i<spline_objects.size();i++)
	{
			getSplinePoints(spline_objects[i],.05);			

	}


}

void getSplinePoints(Spline& spline, float resolution)
{
	float* tangents = (float*)malloc(2*2*sizeof(float));
	float* points = evaluateSpline(spline, resolution, tangents);	

	if(debug_spline)
	{
		for(int i = 0;i<2;i++)
		{
			for(int j = 0;j<2;j++)
			{
				cout<<*(tangents + i*2 + j)<<"\t";
			}
			cout<<endl;
		}
	}
	if(debug_spline)
	{
		for(int i =0;i<20;i++)
		{
			for(int j =0;j<2;j++)
			{
				cout<<*(points + i*20 +j)<<"\t";
			}
			cout<<endl;
		}
	}

}

float* evaluateSpline(Spline& spline, float resolution, float* tangents)
{
	int n = (int)(1./resolution) + 1;
	
	float* points = (float*)malloc(n*2*sizeof(float));

	float M3 [] = {-1,3,3,1,3,-6,3,0,-3,3,0,0,1,0,0,0};
	
	float* spline_points = (float*)malloc((spline.degree + 1)*2*sizeof(float));
	float P[2], dP[2], ddP[2], dddP[2];
	float h2 = resolution*resolution;
	float h3 = resolution*h2;

	for(int i =0;i<(spline.degree +1);i++)
	{
		*(spline_points + i*2) = (float) spline.points[i].x;
		*(spline_points + i*2 + 1) = (float) spline.points[i].y;

	}
	
	if(debug_spline)
	{
		for(int i = 0;i<(spline.degree + 1);i++)
		{
			for(int j =0;j<2;j++)
			{
				cout<<"Coordinates \t"<<*(spline_points +i*(2) + j);

			}
			cout<<endl;
		}
	}

	float* spline_control_points = (float*)malloc((spline.degree + 1)*2*sizeof(float));
	
	float *M = (float*)malloc(sizeof(float)*4*4);
	memcpy(M, M3, sizeof(float)*4*4);

	if(debug_spline)
	{
		for(int i = 0;i<4;i++)
		{
			for(int j =0;j<4;j++)
			{
				cout<<*(M + i*4 + j)<<"\t";

	
			}
		cout<<endl;
		}
	}
	matrix_multiplication_spline(M, 4, 4, spline_points, (spline.degree + 1), 2, spline_control_points);
	
	if(debug_spline)
	{
		for(int i = 0;i<4;i++)
		{
			for(int j =0;j<2;j++)
			{
				cout<<"Points \t"<<*(spline_control_points + i*2 + j)<<"\t";
			
			}
			cout<<endl;
		}
	}

	int p_col = 2;

	P[0] = *(spline_control_points + 3*p_col);
	P[1] = *(spline_control_points + 3*p_col + 1);

	dP[0] = *(spline_control_points + 2*p_col)*resolution + *(spline_control_points + 1*p_col)*h2 + *(spline_control_points)*h3;
	dP[1] = *(spline_control_points + 2*p_col + 1)*resolution + *(spline_control_points + 1*p_col + 1)*h2 + *(spline_control_points + 1)*h3;

	dddP[0] = 6*(*(spline_control_points))*h3;
	dddP[1] = 6*(*(spline_control_points + 1))*h3;

	ddP[0] = 2*(*(spline_control_points + 1*p_col))*h2 + dddP[0];
	ddP[1] = 2*(*(spline_control_points + 1*p_col + 1))*h2 + dddP[1];

	

	if(debug_spline)
	{
		cout<<"Number of Points \t"<<n<<endl;
	}

	for(int i = 0;i<n;i++)
	{
		*(points + i*p_col) = P[0];
		*(points + i*p_col + 1) = P[1];
		
		P[0] += dP[0];
		P[1] += dP[1];
		dP[0] += ddP[0];
		dP[1] += ddP[1];
		ddP[0] += dddP[0];
		ddP[1] += dddP[1];

	}

	if(tangents)
	{
		*(tangents) = *(spline_control_points + 2*p_col);
		*(tangents + 1) = *(spline_control_points + 2*p_col +1);
		*(tangents + 1*p_col) =  3*(*(spline_control_points)) + 2*(*(spline_control_points + 1*p_col)) + *(spline_control_points + 2*p_col);
		*(tangents + 1*p_col) = 3*(*(spline_control_points + 1)) + 2*(*(spline_control_points + 1*p_col + 1)) + *(spline_control_points + 2*p_col + 1);
	
	}
		

	return points;

}

/*
void fitSpline(vector<Spline>& spline_objects)
{
	int count_spline_objects = spline_objects.size();
	for(int i  = 0;i<count_spline_objects;i++)
	{
		fitbezierSpline(spline_objects[i].spline_x_y_points);

	}
	




}
*/

Spline getLine2Spline(Line& line_object, int degree)
{
	
	Spline spline;
	spline.degree = 3;


	spline.points[0] = line_object.startpoint;
	spline.points[degree] = line_object.endpoint;

	Linepoint direction = getDirection(line_object.endpoint, line_object.startpoint);
	
	for(int i = 1;i<degree;i++)
	{
		Linepoint point;
		float t = i/(float) degree;
		point.x = line_object.startpoint.x + t*direction.x;
		point.y = line_object.startpoint.y + t*direction.y;

		spline.points[i] = point;

	}
	spline.spline_x_y_points = line_object.x_y_points;

	if(debug_spline)
	{
		for(int i =0;i<4;i++)
		{
			cout<<"Coordinates \t"<<spline.points[i].x<<"\t"<<spline.points[i].y<<endl;

		}	
	}

	return spline;


}


Linepoint getDirection(const Linepoint& v1, const Linepoint& v2)
{	
	Linepoint dir = {v1.x - v2.x, v1.y - v2.y};
	return dir;

}

/*
void fitbezierSpline(vector<Linepoint>& spline_x_y_points, int degree)
{
	





}
*/





