#ifndef SPLINE_HPP
#define SPLINE_HPP

using namespace std;
using namespace cv;

#define DEGREE 3

class Spline
{
	public:
		int degree;
		Linepoint_f points[4];
		vector<Linepoint> spline_x_y_points;
		int x_limit_min;
		int x_limit_max;


};

void  matrix_multiplication_spline(float* arr1, int arr1_rows, int arr1_cols, float* arr2, int arr2_rows, int arr2_cols, float* r_arr);
void getRansacSplines(vector<Line>& line_objects, vector<Spline>& spline_objects, Mat& gray_ipm_image);
Spline getLine2Spline(Line& line_object, int degree);
Linepoint getDirection(const Linepoint& v1, const Linepoint& v2);
void drawSpline(Mat& gray_ipm_image, vector<Spline>& spline_objects);
vector<Linepoint> getSplinePoints(Spline& spline, float resolution);
float* evaluateSpline(Spline& spline, float resolution, float* tangents);
void getLineIntersection(Line& line_obj, int width, int height);
void fitbezierSpline(Spline& prevSpline, vector<Linepoint>& spline_x_y_points, int degree);
void fitSpline(vector<Spline>& spline_objects);
void getCumSum(float* in_arr, float* out_arr, int count);
void calculatenew_weights(float* weights, int numSamples, int* randIndex, int count); 
Spline fitsplinewithRansac(vector<Linepoint>& SamplePoints, int degree, int count);

#endif
