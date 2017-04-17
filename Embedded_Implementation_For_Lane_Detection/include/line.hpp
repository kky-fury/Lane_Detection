#ifndef LINE_HPP
#define LINE_HPP

#include<iostream>
#include<cstdio>
#include"opencv2/opencv.hpp"
#include"opencv2/highgui.hpp"
#include"opencv2/gpu/gpu.hpp"
#include<vector>
#include<math.h>
#include<cstring>
#include<string>


using namespace std;
using namespace cv;

typedef struct point
{
	int x;
	int y;

}Linepoint;

class Line
{

	public:
		Linepoint startpoint;
		Linepoint endpoint;
		vector<tuple<int,int>> x_y_points;

		Linepoint getstartpoint();
		Linepoint getendpoint();

		Line(Linepoint startpoint, Linepoint endpoint);


}






#endif
