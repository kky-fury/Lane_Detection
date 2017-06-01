#include"line.hpp"
#include"spline.hpp"
#include"line_fitting.hpp"

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



	fitSpline(spline_objects);
	//drawSpline(gray_ipm_image, spline_objects);

}


void drawSpline(Mat& gray_ipm_image, vector<Spline>& spline_objects)
{

	for(int i =0;i<spline_objects.size();i++)
	{
			vector<Linepoint> rpoints = getSplinePoints(spline_objects[i],.05);			
			for(int i =0;i<rpoints.size() - 1;i++)
			{	
				Point pt1, pt2;
				pt1.x = rpoints[i].x;
				pt1.y = rpoints[i].y;
				pt2.x = rpoints[i+1].x;
				pt2.y = rpoints[i+1].y;

				cv::line(gray_ipm_image, pt1, pt2, (0,255,0),2);


			}

	}

	imshow("Result", gray_ipm_image);
	waitKey(0);


}

vector<Linepoint> getSplinePoints(Spline& spline, float resolution)
{

	float* tangents = (float*)malloc(2*2*sizeof(float));
	float* points = evaluateSpline(spline, resolution, tangents);	

	int n = (int)(1./resolution) + 1;
	

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
		for(int i =0;i<n;i++)
		{
			for(int j =0;j<2;j++)
			{
				cout<<*(points + i*2 +j)<<"\t";
			}
			cout<<endl;
		}
	}
	
	vector<int> inpoints;
	vector<int>::iterator it;	
	int lastin = -1;
	int nummin = 0;
	
	for(int i =0 ;i<n;i++)
	{
		*(points + i*2) = round(*(points + i*2));
		*(points + i*2 + 1) = round(*(points + i*2 +1));

		if(*(points + i*2) >= 0 && *(points + i*2) >= (spline.x_limit_min - 1) && *(points + i*2) <= (spline.x_limit_max + 1) && *(points + i*2 + 1) >= 0 && *(points + i*2 + 1) < 400)
		{
			if(lastin < 0 || (lastin >=0 && !(*(points + lastin*2 + 1) == *(points + i*2 + 1) && *(points + lastin*2) == *(points +i*2))))
			{
				inpoints.push_back(i);
				lastin = i;
				nummin++;
			}

		}
	
	}

	if(debug_spline)
	{
		for(int i =0;i<n;i++)
		{
			for(int j =0;j<2;j++)
			{
				cout<<*(points + i*2 +j)<<"\t";
			}
			cout<<endl;
		}
	}

	int p0 = inpoints.front();
	
	cout<<p0<<endl;
	Line line_obj;
	Point pt1, pt2;
	pt1.x = *(points + p0*2) -10*(*(tangents));
	pt1.y = *(points + p0*2 + 1) - 10*(*(tangents + 1));

	pt2.x = *(points + p0*2);
	pt2.y = *(points + p0*2 + 1);
	
	//boundline(200, 400, pt1, pt2);
	
	line_obj.startpoint.x = pt1.x;
	line_obj.startpoint.y = pt1.y;
	line_obj.endpoint.x = pt2.x;
	line_obj.endpoint.y = pt2.y;


	//cout<<"Coordinates \t"<<line_obj.startpoint.x<<"\t"<<line_obj.startpoint.y<<endl;
	//cout<<"Coordinates \t"<<line_obj.endpoint.x<<"\t"<<line_obj.endpoint.y<<endl;

	getLineIntersection(line_obj, spline.x_limit_max +1 , 400);
	


	cout<<"Coordinates \t"<<line_obj.startpoint.x<<"\t"<<line_obj.startpoint.y<<endl;
	cout<<"Coordinates \t"<<line_obj.endpoint.x<<"\t"<<line_obj.endpoint.y<<endl;


	//int* rpoints = (int*)malloc(sizeof(int)*nummin*2);
	vector<Linepoint> rpoints(nummin);
	int ri = 0;
	for(it = inpoints.begin(); it != inpoints.end();ri++,it++)
	{
		//*(rpoints + ri*2) = (int) *(points + *(it)*2);
		//*(rpoints + ri*2 + 1) = (int) *(points + *(it)*2 + 1);
		rpoints[ri].x =(int) *(points + *(it)*2);
		rpoints[ri].y =(int) *(points + *(it)*2 + 1);

	}

	if(debug_spline)
	{
		vector<Linepoint>::iterator iter;
		for(iter = rpoints.begin();iter!= rpoints.end();iter++)
		{
			cout<<iter->x<<"\t"<<iter->y<<endl;
		}
	
	}

	return rpoints;


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

void fitSpline(vector<Spline>& spline_objects)
{
	int count_spline_objects = spline_objects.size();
	for(int i  = 0;i<count_spline_objects;i++)
	{
		fitbezierSpline(spline_objects[i],spline_objects[i].spline_x_y_points, DEGREE);

	}
	




}

Spline getLine2Spline(Line& line_object, int degree)
{
	
	Spline spline;
	spline.degree = 3;


	//spline.points[0] = (Linepoint_f) line_object.startpoint;
	spline.points[0] = {(float) line_object.startpoint.x, (float) line_object.endpoint.x};
	//spline.points[degree] = (Linepoint_f) line_object.endpoint;
	spline.points[degree] = {(float) line_object.endpoint.x, (float) line_object.endpoint.y};
	Linepoint direction = getDirection(line_object.endpoint, line_object.startpoint);
	
	for(int i = 1;i<degree;i++)
	{
		Linepoint point;
		float t = i/(float) degree;
		point.x = line_object.startpoint.x + t*direction.x;
		point.y = line_object.startpoint.y + t*direction.y;

		//spline.points[i] = (Linepoint_f) point;
		spline.points[i] = {(float) point.x, (float) point.y};
	}
	spline.spline_x_y_points = line_object.x_y_points;

	if(debug_spline)
	{
		for(int i =0;i<4;i++)
		{
			cout<<"Coordinates \t"<<spline.points[i].x<<"\t"<<spline.points[i].y<<endl;

		}	
	}

	spline.x_limit_max = max(line_object.startpoint.x, line_object.endpoint.x);
	spline.x_limit_min = min(line_object.startpoint.x, line_object.endpoint.x);

	return spline;


}


Linepoint getDirection(const Linepoint& v1, const Linepoint& v2)
{	
	Linepoint dir = {v1.x - v2.x, v1.y - v2.y};
	return dir;

}

void fitbezierSpline(Spline& prevSpline, vector<Linepoint>& spline_x_y_points, int degree)
{
	int numSamples =  10;
	int numIterations = 1;
	int numGoodFit = 4;

	int count = spline_x_y_points.size();
	cout<<"Number of Points \t"<<count<<endl;
	float* weights = (float*)malloc(count*sizeof(float));
	
	for(int i = 0;i<count;i++)
	{
		*(weights + i) = 1.f;

	}
	
	getCumSum(weights, weights, count);

	int* randIndex =  (int*)malloc(sizeof(int)*numSamples);
	//int* samplePoints = (int*)malloc(sizeof(int)*numSamples*2);
	vector<Linepoint> samplePoints(numSamples);
	int* pointIndex = (int*)malloc(sizeof(int)*count);

	Spline curSpline, bestSpline;
	bestSpline.degree = 0;
	float bestScore = 0;

	//vector<Spline>::iterator prevSpline;
	
	for(int i = 0;i<numIterations;i++)
	{

		for(int i = 0; i<count;i++)
		{
			*(pointIndex + i) =0;
		}
		
		calculatenew_weights(weights, numSamples, randIndex, count);
		for(int j = 0;j<numSamples;j++)
		{
			int p = *(randIndex + j);
			*(pointIndex + p) = 1;

			//*(samplePoints + j*2) = spline_x_y_points[p].x;
			//*(samplePoints + j*2 + 1) = spline_x_y_points[p].y;
			samplePoints[j] = {spline_x_y_points[p].x, spline_x_y_points[p].y};
		}

		if(debug_spline)
		{
		
			for(int i = 0;i<numSamples;i++)
			{
				cout<<*(randIndex + i)<<endl;
				cout<<"Coordinates of Sample Points \t"<<samplePoints[i].x<<"\t"<<samplePoints[i].y<<endl;
			
			}
			for(int k = 0;k<count;k++)
			{
				cout<<*(pointIndex + k)<<endl;
			}
		}

		curSpline = fitsplinewithRansac(samplePoints, DEGREE, numSamples);
		/*
		for(int i = 0;i<curSpline.degree;i++)
		{
			cout<<curSpline.points[i].x<<"\t"<<curSpline.points[i].y<<endl;
		}

		cout<<endl;
		*/
	}

	

	if(debug_spline)
	{
		for(int i = 0;i<count;i++)
		{
			cout<<"\t"<<*(weights + i);
		}
		cout<<endl;
		
		for(int i  =0;i<spline_x_y_points.size();i++)
		{	
			cout<<"Coordintes \t"<<spline_x_y_points[i].x<<"\t"<<spline_x_y_points[i].y<<endl;

		}
		

	}







}

void getLineIntersection(Line& line_obj, int width, int height)
{

	bool startInside, endInside;
	startInside = isPointInside(line_obj.startpoint, width, height);
	endInside = isPointInside(line_obj.endpoint, width, height);
		
	

	if(!(startInside && endInside))
	{
		float deltax, deltay;
		deltax = line_obj.endpoint.x - line_obj.startpoint.x;
		deltay =  line_obj.endpoint.y - line_obj.startpoint.y;
	
		float t[4] = {2,2,2,2};
		float xup, xdown, yleft, yright;

		if(deltay == 0)
		{
			xup = xdown = width + 2;
		}
		else
		{
			t[0] =  -line_obj.startpoint.y/deltay;
			xup = line_obj.startpoint.x + t[0]*deltax;
			t[1] = (height - line_obj.startpoint.y)/deltay;
			xdown = line_obj.endpoint.x + t[1]*deltax;
		}


		if(deltax == 0)
		{
			yleft = yright = height + 2;
			
		}
		else
		{
			t[2] = -line_obj.startpoint.x/deltax;
			yleft = line_obj.startpoint.y + t[2]*deltay;
			t[3] = (width - line_obj.startpoint.x)/deltax;
			yright = line_obj.startpoint.y + t[3]*deltay;
		}

		Linepoint points[4];
		points[0] = {(int)xup, 0};
		points[1] = {(int)xdown, (int)height};
		points[2] = {0, (int) yleft};
		points[3] = {(int) width, (int) yright};
		
		if(debug_spline)
		{
			for(int i =0;i<4;i++)
			{	
				cout<<"Value of t array \t"<<t[i]<<endl;
				cout<<"Coordinates \t"<<points[i].x<<"\t"<<points[i].y<<endl;
			}
		}
		int i;
		if(!startInside)
		{
			bool cont = true;
			for(i = 0;i<4 && cont;i++)
			{
				if(t[i] >=0 && t[i] <= 1 && isPointInside(points[i], width, height) && !(points[i].x == line_obj.endpoint.x && points[i].y == line_obj.endpoint.y))
				{
					line_obj.startpoint.x = points[i].x;
					line_obj.startpoint.y = points[i].y;
					t[i] = 2;
					cont = false;
				}
			
			}

			if(cont)
			{
				for(i = 0;i<4 && cont;i++)
				{
					if(t[i] >=0 && t[i] <= 1 && isPointInside(points[i], width, height))
					{
						line_obj.startpoint.x = points[i].x;
						line_obj.startpoint.y = points[i].y;
						t[i] = 2;
						cont = false;

					}

				}


			}
		

		}
		if(!endInside)
		{
			bool cont = true;
			for(i  = 0;i<4 && cont;i++)
			{
				if(t[i] >= 0 && t[i] <= 1 && isPointInside(points[i], width, height) && !(points[i].x == line_obj.startpoint.x && points[i].y == line_obj.startpoint.y))
				{
					line_obj.endpoint.x = points[i].x;
					line_obj.endpoint.y = points[i].y;
					t[i] = 2;
					cont = false;

				}

			}

			if(cont)
			{
				for(i = 0;i<4 && cont;i++)
				{
					if(t[i] >= 0 && t[i] <= 1 && isPointInside(points[i], width, height))
					{
						line_obj.endpoint.x = points[i].x;
						line_obj.endpoint.y = points[i].y;
						t[i] = 2;
						cont = false;

					}

				}

			}
		}


	
	}


	
}

void getCumSum(float* in_arr, float* out_arr, int count)
{
	for(int i =1 ; i<count;i++)
	{
		*(out_arr + i) += *(out_arr + i-1);

	}

}

void calculatenew_weights(float* weights, int numSamples, int* randIndex, int count)
{
	int i =0, j;
	srand((int)time(0));
	
	if(numSamples >= count)
	{
		for(;i<numSamples;i++)
		{
			*(randIndex +i) = i;

		}

	}
	else
	{
		while(i<numSamples)
		{

			j = rand()%count;
			*(randIndex + i) = j;
			i++;

		}



	}

	if(debug_spline)
	{
		for(int i = 0;i<numSamples;i++)
		{
	
			cout<<"Index Values \t"<<*(randIndex + i)<<endl;
		}
	}
}

Spline fitsplinewithRansac(vector<Linepoint>& samplePoints, int degree, int count)
{

	Spline spline;
	spline.degree = degree;

	int n = count;
	
	sort(samplePoints.begin(), samplePoints.end(), [] (const Linepoint& lhs, const Linepoint& rhs){return lhs.y < rhs.y;});

	if(debug_spline)
	{
		for(int i = 0;i<count;i++)
		{
			cout<<"Coordinates \t"<<samplePoints[i].x<<"\t"<<samplePoints[i].y<<endl;

		}
	}
	
	
	Linepoint_f p0 = {(float) samplePoints[0].x, (float) samplePoints[0].y};
	
	float diff = 0.f;
	float* us = (float*)malloc(sizeof(float)*count);
	
	us[0] = 0;
	
	for(int i =1;i<count;++i)
	{
		float dx = samplePoints[i].x - samplePoints[i-1].x;
		float dy = samplePoints[i].y - samplePoints[i-1].y;

		*(us + i) = sqrt(dx*dx + dy*dy) + us[i-1];

	}

	diff = *(us + count -1);
	//cout<<"Difference Value \t"<<diff<<endl;

	float M3[] = {-1,3,-3,1,3,-6,3,0,-3,3,0,0,1,0,0,0};
	
	float *M = (float*)malloc(sizeof(float)*4*4);
	memcpy(M, M3, sizeof(float)*4*4);
		
	float u = 0.f;

	float* B = (float*)malloc(sizeof(float)*n*4);
	for(int i =0;i<4;i++)
	{
		u = *(us + i)/diff;
		
		*(B + i*4 + 3) = 1;
		*(B + i*4 + 2) = u;
		*(B + i*4 + 1) = u*u;
		*(B + i*4) = u*u*u;

	}

	matrix_multiplication_spline(B, n, 4, M, 4, 4,B);

	if(1)
	{
		for(int i =0;i<n;i++)
		{
			for(int j = 0;j<4;j++)
			{
				cout<<*(B + i*4 + j)<<"\t";
			}
		cout<<endl;
		}
	
	}

	Mat B_arr = Mat(n, 4, CV_32F, B);

	/*
	cout<<endl;
	for(int i =0;i<n;i++)
	{
		for(int j = 0;j<4;j++)
		{
			cout<<B_arr.at<float>(i,j)<<"\t";
		}
		cout<<endl;
	}
	cout<<endl;
	*/
	Mat points(n, 2, CV_32F);

	for(int i =0;i<count;i++)
	{
		points.at<float>(i,0) = (float) samplePoints[i].x;
		points.at<float>(i,1) = (float) samplePoints[i].y;
	}


	if(debug_spline)
	{
		for(int i = 0;i<count;i++)
		{
			cout<<points.at<float>(i,0)<<"\t"<<points.at<float>(i,1)<<endl;
		}
	}

	if(debug_spline)
	{
		for(int i =0;i<4;i++)
		{
			for(int j =0;j<4;j++)
			{
			//	cout<<M_arr.at<float>(i,j)<<"\t";
				cout<<B_arr.at<float>(i,j)<<"\t";
			}

			cout<<endl;
		}
	}
	
	Mat sp = Mat(degree + 1,2,CV_32F);
	
	solve(B_arr, points, sp, DECOMP_LU);

	if(debug_spline)
	{
		for(int i =0;i<4;i++)
		{
			for(int j = 0;j<2;j++)
			{
				cout<<"Knot points \t"<<sp.at<float>(i,j)<<"\t";
			}
			cout<<endl;

		}
		cout<<endl;
	}

	for(int i = 0 ;i< (degree + 1);i++)
	{
		spline.points[i].x = sp.at<float>(i,0);
		spline.points[i].y = sp.at<float>(i,1);
	}

	B_arr.release();
	sp.release();
	delete [] us;

 return spline;

}
