#include"line.hpp"
#include"poly.hpp"

bool debug_poly = false;


void print_matrix(const boost::numeric::ublas::matrix<double> &m)
{

	for(unsigned i =0;i<m.size1();++i)
	{
		for(unsigned j =0;j<m.size2();++j)
		{
			cout<<m(i,j)<<"\t";
		}
		cout<<endl;
	}


	cout<<endl;


}

void print_d_vector(const std::vector<double>& arr)
{
	for(int i  = 0;i<arr.size();i++)
		cout<<arr[i]<<"\t";

	cout<<endl;


}

void getPolyFit(Line& line_objects, Mat& gray_IPM_image, vector<Linepoint>& x_y_points)
{
			
	vector<Linepoint> fitting_points = getPoints(x_y_points, line_objects);
	if(fitting_points.size() < 100)
		return;
	vector<double> coeff = polyfit(fitting_points, 2);	
	polyval(coeff, gray_IPM_image, line_objects);		


}

vector<Linepoint> getPoints(const vector<Linepoint>& x_y_points, Line& line_obj)
{
	int x_limit_max = max(line_obj.startpoint.x, line_obj.endpoint.x);
	int x_limit_min = min(line_obj.startpoint.x, line_obj.endpoint.x);

	vector<Linepoint> fitting_points;

	int size = (x_limit_max + 7) - (x_limit_min - 1) + 1;
	
	vector<int> search_points(size);
	int j;

	for(int k = x_limit_min - 1, j=0;k<=x_limit_max + 7;k++,j++)
	{	
		search_points[j] = k;
	}
	if(debug_poly)
	{
		cout<<"Size \t"<<size<<endl;
		print_int_vector(search_points);
	}
	vector<Linepoint>::const_iterator it;
	for(it = x_y_points.begin();it<x_y_points.end();it++)
	{
		vector<int>::iterator iter;
		iter = find(search_points.begin(), search_points.end(), it->x);
		if(iter != search_points.end())
		{
			fitting_points.push_back({it->x,it->y});
		}

	}

	if(debug_poly)
	{
		for(int i =0 ;i < fitting_points.size();i++)
		{
			cout<<"Coordinates \t"<<fitting_points[i].x<<"\t"<<fitting_points[i].y<<endl;	
		}
	}
	return fitting_points;
}


vector<double> polyfit(vector<Linepoint>& x_y_points, int degree)
{

	// Pick 10 Random Points//

	sort(x_y_points.begin(), x_y_points.end(), [](const Linepoint& lhs, const Linepoint& rhs) {return lhs.x < rhs.x;});
	using namespace boost::numeric::ublas;


	size_t count = x_y_points.size();
	degree++;

	

	matrix<double> oXmatrix(count, degree);
	matrix<double> oYmatrix(count, 1);

	for(size_t i = 0; i < count; i++)
	{
		oYmatrix(i, 0) = (double) x_y_points[i].y;

	}

	for(size_t nRow = 0; nRow<count;nRow++)
	{
		double nVal = 1.0;
		for(int nCol = 0;nCol < degree;nCol++)
		{
			oXmatrix(nRow, nCol) = nVal;
			nVal *= (double) x_y_points[nRow].x;

		}

	}
	if(debug_poly)
	{
		print_matrix(oXmatrix);
		print_matrix(oYmatrix);
	}

	matrix<double> oXtMatrix(trans(oXmatrix));
	if(debug_poly)
	{
		print_matrix(oXtMatrix);
	}
	matrix<double> oXtXMatrix(prec_prod(oXtMatrix, oXmatrix));
	if(debug_poly)
	{
		print_matrix(oXtXMatrix);
	}

	matrix<double> oXtYMatrix(prec_prod(oXtMatrix, oYmatrix));
	if(debug_poly)
	{
		print_matrix(oXtYMatrix);
	}
	permutation_matrix<int> pert(oXtXMatrix.size1());
	const std::size_t singular = lu_factorize(oXtXMatrix, pert);

	BOOST_ASSERT( singular == 0 );

	lu_substitute(oXtXMatrix, pert, oXtYMatrix);
	if(debug_poly)
	{
		print_matrix(oXtYMatrix);	
	}

	return std::vector<double>(oXtYMatrix.data().begin(), oXtYMatrix.data().end());

}

void polyval(const std::vector<double>& oCoeff, Mat& gray_IPM_image, Line& line_obj)
{

	vector<Linepoint> v_unique_points;
	double a,b,c, determinant;
	double x1,x2;
	a = oCoeff[2];
	b = oCoeff[1];
	
	for(int i = 399 ;i >= 0; i -= 1)
	{
		c = oCoeff[0] - i;
		determinant = b*b - 4*a*c;
		if(determinant > 0)
		{
			x1 =  (-b + sqrt(determinant)) / (2*a);
			x2 = (-b - sqrt(determinant)) / (2*a);
		}
		else if (determinant == 0)
		{
			x1 = (-b + sqrt(determinant)) / (2*a);
		}
	
		double diff = fabs(x1 -x2);
		if(diff > 8)
		{
				if(x1 > (line_obj.startpoint.x - 10) && x1 <  (line_obj.startpoint.x + 10))
				{
					v_unique_points.push_back({(int) round(x1), (int) i});
				}
				else
				{
				v_unique_points.push_back({(int) round(x2), (int) i});
				}
		}
		else
		{
				v_unique_points.push_back({(int) round(x1), (int) i});
				v_unique_points.push_back({(int) round(x2), (int) i});

		}
	}
	sort(v_unique_points.begin(), v_unique_points.end(), [](const Linepoint& lhs, const Linepoint& rhs){
			if(lhs.x != rhs.x)
			{
				return lhs.x  < rhs.x;
			}
			else
				return lhs.y < rhs.y;
			});
	//cv::line(gray_IPM_image, cv::Point(94, 60), cv::Point(98,16), (0,255,0),2);
	
	if(debug_poly)
	{
		for(int i =0;i<v_unique_points.size();i++)
		{
			cout<<"Points \t"<<v_unique_points[i].x<<"\t"<<v_unique_points[i].y<<endl;
		}		
	}

	for(int i = 1;i<=v_unique_points.size() - 1;i++)
	{
		Point pt1, pt2;
		pt1 = {(int) v_unique_points[i-1].x, (int) v_unique_points[i-1].y};
		pt2 = {(int) v_unique_points[i].x, (int) v_unique_points[i].y};
		if(debug_poly)
		{
			cout<<"Point 1 \t"<<pt1.x<<"\t"<<pt1.y<<endl;
			cout<<"Point 2 \t"<<pt2.x<<"\t"<<pt2.y<<endl;
		}
		cv::line(gray_IPM_image, pt1, pt2, (0,255, 0),2);
	}
}

vector<double> getLinePoints(Line& line_obj)
{

	Linepoint start, end;
	start = {line_obj.startpoint.x, line_obj.startpoint.y};
	end = {line_obj.endpoint.x, line_obj.endpoint.y};

	if(debug_poly)
	{
		cout<<"Startpoint \t"<<start.x<<"\t"<<start.y<<endl;
		cout<<"Endpoint \t"<<end.x<<"\t"<<end.y<<endl;
	
	}
	int deltay = end.y - start.y;
	int deltax = end.x - end.y;

	bool steep = false;
	if(abs(deltay) > abs(deltax))
	{
		steep = true;
		int t;

		t = start.x;
		start.x = start.y;
		start.y = t;

		t = end.x;
		end.x = end.y;
		end.y = t;

	}

	bool swap = false;

	if(start.x > end.x)
	{
		Linepoint t = start;
		start = end;
		end = t;
		swap = true;
		
	}

	deltay = end.y - start.y;
	deltax = end.x - start.x;

	int error = 0;
	int deltaerror = abs(deltay);

	int ystep = -1;
	
	if(deltay >=0)
	{	
		ystep = 1;
	}

	vector<Linepoint> pixels(end.x - start.x + 1);
	int i,j;
	j = start.y;

	int k, kupdate;
	
	if(!swap)
	{
		k =0;
		kupdate = 1;
	}
	else
	{
		k = pixels.size() -1;
		kupdate = -1;
			
	}

	for(i = start.x; i<end.x;i++, k+=kupdate)
	{
		if(steep)
		{
			pixels[k] = {j, i};

		}
		else
		{
			pixels[k] = {i, j};
		}

		error += deltaerror;

		if(2*error >= deltax)
		{
			j += ystep;
			error -= deltax;
		}

		
	}

	if(debug_poly)
	{
		for(int i =0;i<pixels.size() -1;i++)
		{
			cout<<"Points \t"<<pixels[i].x<<"\t"<<pixels[i].y<<endl;
		}
	}

	vector<double> x_values(pixels.size());

	for(int i = 0;i<pixels.size();i++)
	{
		x_values[i] = (double) pixels[i].x;
	}

	return x_values;
}
