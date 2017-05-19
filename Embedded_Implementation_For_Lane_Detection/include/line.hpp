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
#include<algorithm>
#include <numeric>
#include <functional> 


#include"hough.hpp"

using namespace std;
using namespace cv;

typedef struct point
{
	int x;
	int y;

}Linepoint;


typedef struct point_f
{
	float x;
	float y;

}Linepoint_f;

/*
typedef struct int3_t
{
	int x;
	int y;
	int z;

}int3;
*/
typedef struct line_coord_t
{
	Linepoint startpoint;
	Linepoint endpoint;

}line_coord;

typedef struct coor_vote_t
{
	int coordinate;
	int votes;
}coor_vote;

class Line
{

	public:
		Linepoint startpoint;
		Linepoint endpoint;
		vector<Linepoint> x_y_points;
		int votes;
		Linepoint getstartpoint();
		Linepoint getendpoint();
		void setPoints(line_coord* coordinates);
};


void checklanewidth(vector<Line>& line_objects, int line_count);
void getLineObjects(vector<Line>& line_objects, lin_votes* hough_lines,int* votes, int image_width, int image_height);
line_coord* getLineEndPoints(float rho, float theta_line, int image_width, int image_height);
bool isPointInside(Linepoint point, int image_width, int image_height);
vector<Linepoint> initializePoints(vector<Line>& line_objects, unsigned int* clist, int count);
void initializeLinePoints(vector<Linepoint>& x_y_points, vector<Line>& line_objects);
void print_int_vector(vector<int>& vec);

#endif
