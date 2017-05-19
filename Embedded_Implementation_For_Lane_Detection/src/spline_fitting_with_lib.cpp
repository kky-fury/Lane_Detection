#include"spline.h"
#include"line.hpp"
#include"line_fitting.hpp"
#include"spline.hpp"

bool debug_spline = false;



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

Spline getLine2Spline(Line& line_object, int degree)
{
	
	Spline spline;
	spline.degree = 3;
	
	spline.points[0] = {(float) line_object.startpoint.x, (float) line_object.endpoint.x};
	spline.points[degree] = {(float) line_object.endpoint.x, (float) line_object.endpoint.y};

	Linepoint direction = getDirection(line_object.endpoint, line_object.startpoint);
	for(int i = 1;i<degree;i++)
	{
		Linepoint point;
		float t = i/(float) degree;
		point.x = line_object.startpoint.x + t*direction.x;
		point.y = line_object.startpoint.y + t*direction.y;
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



void fitSpline(vector<Spline>& spline_objects)
{
	int count_spline_objects = spline_objects.size();
	for(int i  = 0;i<count_spline_objects;i++)
	{
		fitbezierSpline(spline_objects[i],spline_objects[i].spline_x_y_points, DEGREE);
	}

}

void fitbezierSpline(Spline& prevSpline, vector<Linepoint>& spline_x_y_points, int degree)
{
	int count = spline_x_y_points.size();
	vector<double> X(count), Y(count);
	
	sort(spline_x_y_points.begin(), spline_x_y_points.end(), [] (const Linepoint& lhs, const Linepoint& rhs){return lhs.x < rhs.x; });

	if(debug_spline)
	{
		for(int i = 0;i < spline_x_y_points.size();i++)
		{
			cout<<"Coordinates \t"<<spline_x_y_points[i].x<<"\t"<<spline_x_y_points[i].y<<endl;
	
		}
	}
	
	for(int i  =0;i < count;i++)
	{
		X[i] = (double) spline_x_y_points[i].x;
		Y[i] = (double) spline_x_y_points[i].y;

	}
	/*
	tk::spline s;
	s.set_points(X,Y);

	double value = s(88);
	cout<<"value \t"<<value<<endl;
	*/




}
