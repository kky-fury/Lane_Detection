#ifndef POLY_HPP
#define POLY_HPP

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include<vector>

using namespace std;

inline bool operator<(const Linepoint& lhs, const Linepoint& rhs)
{
	if(lhs.x != rhs.x)
		return lhs.x < rhs.x;
	else if (lhs.x == rhs.x)
		return lhs.y < rhs.y;

}

void getPolyFit(Line& line_objects, Mat& gray_IPM_image, vector<Linepoint>& x_y_points);
vector<double> polyfit(vector<Linepoint>& x_y_points, int degree);
void print_matrix(const boost::numeric::ublas::matrix<double> &m);
vector<double> getLinePoints(Line& line_obj);
void polyval(const std::vector<double>& oCoeff, Mat& gray_IPM_image, Line& line_obj) ;
void print_d_vector(const std::vector<double>& arr);
vector<Linepoint> getPoints(const vector<Linepoint>& x_y_points, Line& line_obj);
double getSlope(double a, double x, double b );


#endif
