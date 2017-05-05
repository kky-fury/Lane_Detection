#include"line_fitting.hpp"


void fit_line(vector<Line>& line_objects, Mat& gray_ipm_image)
{
	int count_line_objects =  line_objects.size();
	
	/*
	for(int i = 0;i<line_objects.size();i++)
	{
	
		for(int j = 0;j<line_objects[i].x_y_points.size();j++)
		{
			cout<<"X Coordinate \t"<<line_objects[i].x_y_points[j].x<<"\t"<<"Y Coordinate \t"<<line_objects[i].x_y_points[j].y<<"\t";	
		}
	}
	*/

	float* line = (float*)malloc(4*sizeof(float));
	int mult = std::max(400, 200);
	for(int i = 0;i<count_line_objects;i++)
	{
		
		fitline2D(line_objects[i].x_y_points, line);
		//cout<<"Line Parameters"<<line[0]<<"\t"<<line[1]<<"\t"<<line[2]<<"\t"<<line[3]<<endl;
		Point pt1, pt2;	
		pt1.x = (int)(line[2] - mult*line[0]);
		pt1.y = (int)(line[3] - mult*line[1]);
		pt2.x = (int)(line[2] + mult*line[0]);
		pt2.y = (int)(line[3] + mult*line[1]);
		
		boundline(200, 400, pt1, pt2);
		int x_limit_max = std::max(line_objects[i].startpoint.x, line_objects[i].endpoint.x);
		int x_limit_min = std::min(line_objects[i].startpoint.x, line_objects[i].endpoint.x);
		
		if(pt1.x < x_limit_min)
			pt1.x = x_limit_min - 1;
		else if(pt1.x > x_limit_max)
			pt1.x = x_limit_max - 1;

		if(pt2.x < x_limit_min)
			pt2.x = x_limit_min - 3;
		else if(pt2.x > x_limit_max)
			pt2.x = x_limit_max - 3;
		
		//cout<<"Line Coordinates \t"<<"Point 1 \t"<<"("<<pt1.x<<","<<pt1.y<<")"<<endl;
		//cout<<"Line Coordinates \t"<<"Point 2 \t"<<"("<<pt2.x<<","<<pt2.y<<")"<<endl;	
	
		/*final check*/
	//	int dist_between_points = fabs(pt1.x - pt2.x);
		cv::line(gray_ipm_image, pt1, pt2, (0,255,0),2);
		
	}




};


void fitline2D(vector<Linepoint>& x_y_points, float* line)
{
	int count = x_y_points.size();
	double eps = count*STOP_C;

	float* weights =  (float*)malloc(count*sizeof(float));
	float* dist = (float*)malloc(count*sizeof(float));
	
	float delta_1 = 0.01f;
	float delta_2 = 0.01f;
	
	float min_sum_dist = DBL_MAX;

	srand((int)time(0));
	int i, j, k;
	float _line[4], _lineprev[4];
	
	double sum_dist = 0;


	for(k = 0;k<10;k++)
	{
		int first = 1;
		for(i = 0;i<count;i++)
		{
			*(weights + i) = 0.f;
		}
		
		for(i  = 0;i<std::min(count, 10);)
		{
			j  = rand()%count;
			if(weights[j] < STOP_C)
			{	
				weights[j] = 1.f;
				i++;
			}

		}

		fitlinels(x_y_points, count, weights,_line);
		for(i=0;i<15;i++)
		{
			double sum_w = 0;
			if(first)
				first = 0;
			else
			{
				double t = _line[0]*_lineprev[0] + _line[1]*_lineprev[1];
				t = std::max(t, -1.);
				t = std::min(t, 1.);
				if(fabs(acos(t)) < delta_2)
				{
					float x,y,d;

					x = (float)fabs(_line[2] - _lineprev[2]);
					y = (float)fabs(_line[3] - _lineprev[3]);

					d = x>y?x:y;
					
					if(d<delta_1)
						break;
				}

			}
		
			sum_dist = calcdist2D(x_y_points, count, _line, dist);
			
			if(sum_dist < eps)
			{
				break;
			}
			calculateweights(dist, count, weights);
			
			for(j =0;j<count;j++)
				sum_w += weights[j];
			if(fabs(sum_w) > STOP_C)
			{
				sum_w = 1./sum_w;
				for(j = 0;j<count;j++)
				{
					weights[j] = (float)(weights[j]*sum_w);

				}
	
			}else
			{
				for(j = 0;j<count;j++)
				{
					weights[j] = 1.f;
				}

			}

			memcpy(_lineprev, _line, 4*sizeof(float));
			fitlinels(x_y_points, count, weights, _line);	
		
		}
		if(sum_dist < min_sum_dist)
		{
			min_sum_dist = sum_dist;
			memcpy(line, _line, 4*sizeof(float));
			if(sum_dist < eps)
				break;

		}


	}

}

void fitlinels(vector<Linepoint>& x_y_points, int count, float* weights, float* line)
{
	double x = 0, y = 0, x2 = 0, y2 =0, xy = 0, w = 0;
	double dx2, dy2, dxy;

	int i;
	int count_t = count;
	float t;

	if(weights == 0)
	{
		for(i = 0;i<count_t;i++)
		{
			x +=x_y_points[i].x;
			y +=x_y_points[i].y;

			x2 +=x_y_points[i].x*x_y_points[i].x;
			y2 +=x_y_points[i].y*x_y_points[i].y;
			xy +=x_y_points[i].x*x_y_points[i].y;
			
		}

		w = (float)count;

	}else
	{
		for(int i =0;i<count_t;i++)
		{
			x += weights[i]*x_y_points[i].x;
			y += weights[i]*x_y_points[i].y;
			x2 += weights[i]*x_y_points[i].x*x_y_points[i].x;
			y2 += weights[i]*x_y_points[i].y*x_y_points[i].y;
			xy += weights[i]*x_y_points[i].x*x_y_points[i].y;
			w += weights[i];
		
		}
	}

	x /= w;
	y /= w;
	x2 /=w;
	y2 /=w;
	xy /=w;

	dx2 = x2 -x*x;
	dy2 = y2 - y*y;
	dxy = xy -x*y;
	
	//cout<<"Value of dxy"<<dxy<<"\t";
	//cout<<"Value of dx2"<<dx2<<"\t";
	//cout<<"Value of dy2"<<dy2<<endl;

	t = (float)atan2(2*dxy, dx2-dy2)/2;
	line[0] = (float)cos(t);
	line[1] = (float)sin(t);

	line[2] = (float)x;
	line[3] = (float)y;


}

double calcdist2D(vector<Linepoint>& x_y_points, int count, float* line, float* dist)
{
	float px = line[2];
	float py = line[3];
	float nx = line[1];
	float ny = -line[0];

	double sum_dist = 0.;
	
	for(int j = 0 ;j<count;j++)
	{
		float x, y;

		x = x_y_points[j].x -px;
		y = x_y_points[j].y - py;

		dist[j] = (float)fabs(nx*x + ny*y);
		sum_dist += dist[j];

	}

	return sum_dist;
	


}

void calculateweights(float* dist, int count, float* weights)
{
	const float c  = 1 / 1.3998f;
	for(int i =0;i<count;i++)
	{
		weights[i] = 1 / (1 + dist[i] * c);

	}


}

void boundline(int width, int height, Point& pt1, Point& pt2)
{
	int x1,y1,x2,y2;
	int c1, c2;
	int right = width - 1;
	int bottom =  height - 1;

	x1 = pt1.x;
	y1 = pt1.y;
	x2 = pt2.x;
	y2 = pt2.y;

	c1 = (x1 < 0) + (x1 > right) * 2 + (y1 < 0) * 4 + (y1 > bottom) * 8;
	c2 = (x2 < 0) + (x2 > right) * 2 + (y2 < 0) * 4 + (y2 > bottom) * 8;

	if( (c1 & c2) == 0 && (c1 | c2) != 0 )
	{
		int a;
		if(c1 & 12)
		{
			a = c1 < 8 ? 0 : bottom;
			x1 +=  (a - y1) * (x2 - x1) / (y2 - y1);
			y1 = a;
			c1 = (x1 < 0) + (x1 > right) * 2;
		}
		if(c2 & 12)
		{
			a = c2 < 8 ? 0 : bottom;
			x2 += (a - y2) * (x2 - x1) / (y2 - y1);
			y2 = a;
			c2 = (x2 < 0) + (x2 > right) * 2;


		}
		if( (c1 & c2) == 0 && (c1 | c2) != 0 )
		{
			if(c1)
			{
				a = c1 == 1 ? 0 : right;
				y1 += (a - x1) * (y2 - y1) / (x2 - x1);
				x1 = a;
				c1 = 0;
			}
			if(c2)
			{
				a = c2 == 1 ? 0 : right;
				y2 += (a - x2) * (y2 - y1) / (x2 - x1);
				x2 = a;
				c2 = 0;
			}
		}

		pt1.x = (int)x1;
		pt1.y = (int)y1;
		pt2.x = (int)x2;
		pt2.y = (int)y2;
	
	
	}


}


























