#ifndef LINE_FITTING_HPP
#define LINE_FITTING_HPP

#include<iostream>
#include<cstdio>
#include"opencv2/opencv.hpp"
#include"opencv2/highgui.hpp"
#include<vector>
#include<math.h>
#include<cstring>
#include<string>
#include<algorithm>
#include<numeric>
#include<functional>
#include<cstdlib>
#include<ctime>
#include<cfloat>
#include"line.hpp"
#include<chrono>

#define STOP_C (1e-15)

using namespace std;
using namespace cv;



void fit_line(Line& line_objects, Mat& gray_ipm_image);
void fitline2D(vector<Linepoint>& x_y_points, float* line);
void fitlinels(vector<Linepoint>& x_y_points, int count, float* weights, float* line);
double calcdist2D(vector<Linepoint>& x_y_points, int count, float* line, float* dist);
void calculateweights(float* dist, int count, float* weights);
void boundline(int width, int height, Point& pt1, Point& pt2);
void getLinePixels(Line& line_obj, Mat& gray_ipm_image);



















#endif
