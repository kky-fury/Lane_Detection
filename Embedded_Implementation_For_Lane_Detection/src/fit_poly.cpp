#include"line.hpp"
#include"poly.hpp"



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

void getPolyFit(vector<Line>& line_objects, Mat& gray_IPM_image)
{

	for(int i  =3;i<line_objects.size();i++)
	{

		cout<<"Size \t"<<line_objects.size()<<endl;
		//cout<<"Startpoint \t"<<line_objects[i].startpoint.x<<"\t"<<line_objects[i].startpoint.y<<endl;
		//cout<<"Endpoint \t"<<line_objects[i].endpoint.x<<"\t"<<line_objects[i].endpoint.y<<endl;
		vector<double> x_points = getLinePoints(line_objects[i]);
		vector<double> coeff = polyfit(line_objects[i].x_y_points, 2);	
		polyval(coeff, x_points, gray_IPM_image);		

	}


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

	//print_matrix(oXmatrix);
	//print_matrix(oYmatrix);

	matrix<double> oXtMatrix(trans(oXmatrix));
//	print_matrix(oXtMatrix);

	matrix<double> oXtXMatrix(prec_prod(oXtMatrix, oXmatrix));
//	print_matrix(oXtXMatrix);

	matrix<double> oXtYMatrix(prec_prod(oXtMatrix, oYmatrix));
//	print_matrix(oXtYMatrix);

	permutation_matrix<int> pert(oXtXMatrix.size1());
	const std::size_t singular = lu_factorize(oXtXMatrix, pert);

	BOOST_ASSERT( singular == 0 );

	lu_substitute(oXtXMatrix, pert, oXtYMatrix);
	//print_matrix(oXtYMatrix);	
	
	return std::vector<double>(oXtYMatrix.data().begin(), oXtYMatrix.data().end());

}

void polyval(const std::vector<double>& oCoeff, const std::vector<double>& oX, Mat& gray_IPM_image)
{
		size_t nCount =  oX.size();
		size_t nDegree = oCoeff.size();
		std::vector<double>	oY( nCount );

		/*
		for(int i  =0;i<oCoeff.size();i++)
		{
			cout<<"Coefficients value \t"<<oCoeff[i]<<endl;

		}
		*/

		for(size_t i = 0; i < nCount;i++)
		{
			double nY = 0;
			double nXT = 1;
			double nX = oX[i];
			for(size_t j = 0;j < nDegree;j++)
			{
				nY += oCoeff[j] * nXT;
				nXT *= nX;	
			}
			//cout<<"Value of nY \t"<<nY<<endl;
			oY[i] = nY;
		}

	/*
	for(int i = 0;i<nCount;i++)
	{
		cout<<"Value of X[i] \t"<<oX[i]<<"\t is"<<oY[i]<<endl;
	}
	*/
	
	double a,b,c, x1, x2, x3, x4;
	double determinant;

	a = oCoeff[2];
	b = oCoeff[1];
	c = oCoeff[0] - 1;
		
	/*value at top of the image*/

	determinant = b*b - 4*a*c;

	cout<<"value of determinant \t"<<determinant<<endl;
	if(determinant > 0)
	{
		x1 =  (-b + sqrt(determinant)) / (2*a);
		x2 = (-b - sqrt(determinant)) / (2*a);
	}
	if(determinant == 0)
	{
		x1 = (-b + sqrt(determinant)) / (2*a);
	}

	/*value at bottom of the image*/
	c = oCoeff[0] - 400;

	determinant = b*b - 4*a*c;
	cout<<"value of determinant \t"<<determinant<<endl;
	
	if(determinant > 0)
	{
		x3 =  (-b + sqrt(determinant)) / (2*a);
		x4 = (-b - sqrt(determinant)) / (2*a);
	}
	if(determinant == 0)
	{
		x3 = (-b + sqrt(determinant)) / (2*a);
	}

	//cout<<"X, Y Coordinates \t"<<x1<<"\t"<<"0"<<"\t"<<x2<<"\t"<<"0"<<"\t"<<x3<<"\t"<<"400"<<"\t"<<x4<<"\t"<<"400"<<endl;
	
	//print_d_vector(oX);
	//print_d_vector(oY);

	set<Linepoint> unique_points;
	
	for(int i = 0; i < nCount ;i++)
	{
		unique_points.insert({(int)oX[i], (int)oY[i]});
	}
	
	for(set<Linepoint>::iterator i = unique_points.begin();i != unique_points.end();i++)
	{
		//cout<<"Points \t"<<unique_points[i].x<<"\t"<<unique_points[i].y<<endl;
		Linepoint elem = *i;
		cout<<"Points \t"<<elem.x<<"\t"<<elem.y<<endl;

	
	}

	/*
	for(size_t i  = 1 ;i< nCount -1;i++)
	{
		Point pt1, pt2;
		pt1 = {(int) oX[i], (int) oY[i] };
		pt2 = {(int) oX[i + 1], (int) oY[i+1]};
	//	cout<<"Point 1 \t"<<pt1.x<<"\t"<<pt1.y<<endl;
	//	cout<<"Point 2 \t"<<pt2.x<<"\t"<<pt2.y<<endl;
		//if(pt1.x > 0 && pt1.y > 0 && pt2.x > 0 && pt2.y > 0)
		cv::line(gray_IPM_image, pt1, pt2, (0,255,0),2);
	}
	*/
}

vector<double> getLinePoints(Line& line_obj)
{

	Linepoint start, end;
	start = {line_obj.startpoint.x, line_obj.startpoint.y};
	end = {line_obj.endpoint.x, line_obj.endpoint.y};
	
	/*
	cout<<"Startpoint \t"<<start.x<<"\t"<<start.y<<endl;
	cout<<"Endpoint \t"<<end.x<<"\t"<<end.y<<endl;
	*/

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

	/*
	for(int i =0;i<pixels.size() -1;i++)
	{
		cout<<"Points \t"<<pixels[i].x<<"\t"<<pixels[i].y<<endl;
	}
	*/	

	vector<double> x_values(pixels.size());

	for(int i = 0;i<pixels.size();i++)
	{
		x_values[i] = (double) pixels[i].x;
	}

	return x_values;
}
