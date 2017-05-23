#include"line.hpp"

void print_int_vector(vector<int>& vec)
{
	vector<int>::const_iterator i;
	for(i = vec.begin();i<vec.end();i++)
	{
		cout<<*i<<"\t";
	}

	cout<<endl;

}

Linepoint Line:: getstartpoint()
{
	return this->startpoint;
	
};

Linepoint Line::getendpoint()
{

	return this->endpoint;

};

void Line::setPoints(line_coord* coordinates)
{
	this->startpoint = coordinates->startpoint;
	this->endpoint = coordinates->endpoint;


}
void getLineObjects(vector<Line>& line_objects, lin_votes* hough_lines, int* votes, int image_width, int image_height)
{
	
	int line_count  = hough_lines->countlines;
	for(int i =0;i<line_count;i++)
	{
		float theta_line = (hough_lines->lines + i)->y;
		float rho = (hough_lines-> lines + i)->x;
		
		line_coord* coordinates = getLineEndPoints(rho, theta_line, image_width, image_height);
		Line line_obj;
		line_obj.setPoints(coordinates);
		line_obj.votes = votes[i];
		line_objects[i] = line_obj;
		
	}

	//sort(line_objects.begin(), line_objects.end(), [] (const Line& lhs, const Line& rhs){return lhs.startpoint.x < rhs.startpoint.x;});
	//sort(line_objects.begin(), line_objects.end(), [] (const Line& lhs, const Line& rhs){return lhs.votes < rhs.votes;});

	sort(line_objects.begin(), line_objects.end(), [] (const Line& lhs, const Line& rhs){return std::max(lhs.startpoint.x, lhs.endpoint.x) < std::max(rhs.startpoint.x, rhs.endpoint.x);});

	for(int i = 0;i<line_objects.size();i++)
	{
		cout<<"X_coordinate \t"<<line_objects[i].startpoint.x<<"\t"<<line_objects[i].endpoint.x<<endl;
		cout<<"Current Line Votes \t"<<line_objects[i].votes<<endl;
	}
	
	checklanewidth(line_objects, line_count);
	
	for(int i = 0;i<line_objects.size();i++)
	{
		cout<<"X_coordinate \t"<<line_objects[i].startpoint.x<<"\t"<<line_objects[i].endpoint.x<<endl;
		cout<<"Current Line Votes \t"<<line_objects[i].votes<<endl;
	}

};

vector<Linepoint> initializePoints(vector<Line>& line_objects, unsigned int* clist, int count)
{
	vector<Linepoint> x_y_points(count);
	for(int i =0;i<count;i++)
	{
		unsigned int const q_value = clist[i];
		const int x = (q_value & 0xFFFF);
		const int y = (q_value >> 16 ) & 0xFFFF;
		x_y_points[i] = {x,y};
	}
	sort(x_y_points.begin(), x_y_points.end(), [] (const Linepoint& lhs, const Linepoint& rhs){return lhs.y < rhs.y; });	
	/*
	for(int i =0;i<count;i++)
	{
		cout<<"X Coordinate \t"<<x_y_points[i].x<<"\t"<<"Y Coordinate \t"<<x_y_points[i].y<<"\t"<<endl;

	}
	*/
	initializeLinePoints(x_y_points, line_objects);

	return x_y_points;

}

void initializeLinePoints(vector<Linepoint>& x_y_points, vector<Line>& line_objects)
{

	for(int i =0;i<line_objects.size();i++)
	{
		int x_limit_max = max(line_objects[i].startpoint.x, line_objects[i].endpoint.x);
		int x_limit_min = min(line_objects[i].startpoint.x, line_objects[i].endpoint.x);

		/*
		cout<<"X Limit Max"<<x_limit_max<<endl;
		cout<<"X Limit Min"<<x_limit_min<<endl;
		*/
		int size = x_limit_max - (x_limit_min -1) + 1;
		
		vector<int> search_points(size);
		int j;
		for(int k = x_limit_min - 1, j=0;k<=x_limit_max;k++,j++)
		{
				search_points[j] = k; 
		}
	
		//print_int_vector(search_points);	
		vector<Linepoint>::const_iterator it;
		for(it = x_y_points.begin();it<x_y_points.end();it++)
		{
			vector<int>::iterator iter;
			iter = find(search_points.begin(), search_points.end(), it->x);
			if(iter != search_points.end())
			{
				line_objects[i].x_y_points.push_back({it->x,it->y});				
			}
		}
		
	}



}


void checklanewidth(vector<Line>& line_objects, int line_count)
{
	int min_distance = 17;
	int max_distance_two_side_lanes = 38;
	int max_distance_two_edge_lanes = 70;

	vector<int> x_points(line_count); 
	vector<coor_vote> coor_vote_arr(line_count); 

	for(int i =0;i<line_count;i++)
	{
		int x_max = max(line_objects[i].startpoint.x, line_objects[i].endpoint.x);
		x_points[i] = x_max;
		coor_vote_arr[i] = {x_max, line_objects[i].votes};
		
	}

	sort(x_points.begin(), x_points.end());
	sort(coor_vote_arr.begin(), coor_vote_arr.end(), [](const coor_vote& lhs, const coor_vote& rhs){return lhs.coordinate < rhs.coordinate; });

	vector<int> diff_array(line_count);	
	adjacent_difference(x_points.begin(), x_points.end(), diff_array.begin());
	
	print_int_vector(diff_array);
	vector<int>::iterator i;
	vector<coor_vote>::iterator j;
	int curr_votes = coor_vote_arr[0].votes;

	for(i = diff_array.begin() + 1, j = coor_vote_arr.begin() + 1; i<diff_array.end(); i++,j++)
	{
		if(*i < min_distance || (*i  > max_distance_two_side_lanes && *i > max_distance_two_edge_lanes))
		{
			line_objects.erase(remove_if(line_objects.begin(), line_objects.end(), [&] (const Line& lhs){return (lhs.startpoint.x == j->coordinate || lhs.endpoint.x == j->coordinate) && (j->votes == lhs.votes) ;}),line_objects.end());
		}

	}




	/*
	for(int i =0;i<coor_vote_arr.size();i++)
	{
		cout<<"X Coordinate Point \t"<<coor_vote_arr[i].coordinate<<"\t"<<endl;
		cout<<"Votes \t"<<coor_vote_arr[i].votes<<endl;

	}
	*/
	//print_int_vector(x_points);

	/*
	if(line_count == 2)
	{
		vector<int> diff_array(line_count);
		adjacent_difference(x_points.begin(), x_points.end(), diff_array.begin());
		vector<int>::iterator i;
		vector<coor_vote>::iterator j;
		int curr_votes = coor_vote_arr[0].votes;
		for(i = diff_array.begin() + 1, j = coor_vote_arr.begin() + 1;i<diff_array.end();i++,j++)
		{
			if(*i < min_distance || j->votes < curr_votes || *i >=max_distance_two_edge_lanes)
			{
				if(j->votes <= curr_votes || *i >=max_distance_two_edge_lanes)
				{
					line_objects.erase(remove_if(line_objects.begin(), line_objects.end(), [&] (const Line& lhs){return (lhs.startpoint.x == j->coordinate || lhs.endpoint.x == j->coordinate) && (j->votes == lhs.votes) ;}),line_objects.end());
				}
			}
			if(j->votes > curr_votes)
			{
				curr_votes = j->votes;			
			}


		}


	}
	*
	*/
	/*
	else if(line_count == 3)
	{
		vector<int> diff_array(line_count);
		adjacent_difference(x_points.begin(), x_points.end(), diff_array.begin());
		//print_int_vector(diff_array);
		vector<int>::iterator i;
		vector<coor_vote>::iterator j;
		int curr_votes = coor_vote_arr[0].votes;
		for(i = diff_array.begin() + 1, j = coor_vote_arr.begin() + 1; i<diff_array.end();i++,j++)
		{
			if(*i < min_distance || j->votes < curr_votes || *i>=max_distance_two_edge_lanes)
			{
				if(j->votes <= curr_votes || *i>= max_distance_two_edge_lanes)
				{	
					line_objects.erase(remove_if(line_objects.begin(), line_objects.end(), [&] (const Line& lhs){return (lhs.startpoint.x == j->coordinate || lhs.endpoint.x == j->coordinate) && (j->votes == lhs.votes) ;}),line_objects.end());

				}
			}
			if(j->votes > curr_votes)
			{
				curr_votes = j->votes;
			}
	
		}

		if(line_objects.size() == 2)
		{
			if(line_objects[1].votes > line_objects[0].votes)
			{
				line_objects.erase(line_objects.begin());
			}
			else
			{
				line_objects.erase(line_objects.begin() + 1);
			}
		}
	}
	*/
	/*
	else if(line_count == 4)
	{
		vector<int> diff_array(line_count);
		adjacent_difference(x_points.begin(), x_points.end(), diff_array.begin());
		//print_int_vector(diff_array);
		vector<int>::iterator i;
		vector<coor_vote>::iterator j;
		int curr_votes = coor_vote_arr[0].votes;
		
		for(i = diff_array.begin() + 1, j = coor_vote_arr.begin() + 1; i<diff_array.end();i++,j++)
		{

			if(*i < min_distance || j->votes < curr_votes || *i >= max_distance_two_edge_lanes || *i >= max_distance_two_side_lanes)
			{
				if(j->votes <= curr_votes || *i>= max_distance_two_edge_lanes || *i >= max_distance_two_side_lanes)
				{
					
		*/
		/*
					cout<<"Current Votes \t"<<j->votes<<endl;
					cout<<"Max Votes \t"<<curr_votes<<endl;
					for(int i =0;i<line_objects.size();i++)
					{
						cout<<"X_coordinate \t"<<line_objects[i].startpoint.x<<"\t"<<line_objects[i].endpoint.x<<endl;	
					}
					cout<<"End of Loop"<<endl;
					*/
		/*
					line_objects.erase(remove_if(line_objects.begin(), line_objects.end(), [&] (const Line& lhs){return (lhs.startpoint.x == j->coordinate || lhs.endpoint.x == j->coordinate) && (j->votes == lhs.votes) ;}),line_objects.end());	
							
				}

			}	
			if(j->votes > curr_votes)
			{
				curr_votes = j->votes;
			}
			
			
		}
		if(line_objects.size() == 2)
		{
			if(line_objects[1].votes > line_objects[0].votes)
			{
				line_objects.erase(line_objects.begin());
			}
			else
			{
				line_objects.erase(line_objects.begin() + 1);
			}
		}
	}
*/
	/*
	else
	{
	*/	/*
		vector<int> diff_array(line_count);
		adjacent_difference(x_points.begin(), x_points.end(), diff_array.begin());
		//print_int_vector(diff_array);
		vector<int>::iterator i;
		vector<coor_vote>::iterator j;
		int curr_votes = coor_vote_arr[0].votes;
		*/
		/*
		for(int i =0;i<line_objects.size();i++)
		{
			cout<<"X_coordinate \t"<<line_objects[i].startpoint.x<<"\t"<<line_objects[i].endpoint.x<<endl;	
			cout<<line_objects[i].votes<<endl;
		}
		*/
		/*
		for(i = diff_array.begin() + 1, j = coor_vote_arr.begin() + 1; i<diff_array.end();i++,j++)
		{
			if(*i < min_distance || j->votes < curr_votes || *i >= max_distance_two_edge_lanes || *i >= max_distance_two_side_lanes)
			{	
				if(j->votes <= (curr_votes-3) || *i>= max_distance_two_edge_lanes || *i >= max_distance_two_side_lanes) 
				{
					line_objects.erase(remove_if(line_objects.begin(), line_objects.end(), [&] (const Line& lhs){return (lhs.startpoint.x == j->coordinate || lhs.endpoint.x == j->coordinate) && (j->votes == lhs.votes) ;}),line_objects.end());
				
				}
				*/
				/*
				for(int i =0;i<line_objects.size();i++)
				{
					
					cout<<"X_coordinate \t"<<line_objects[i].startpoint.x<<"\t"<<line_objects[i].endpoint.x<<endl;	
				}
				*/
			
		/*
			}
			if(j->votes > curr_votes)
				curr_votes = j->votes;
	
		
		}
		*/
		/*
		for(int i =0;i<line_objects.size();i++)
		{
			cout<<"X_coordinate \t"<<line_objects[i].startpoint.x<<"\t"<<line_objects[i].endpoint.x<<endl;	
			cout<<line_objects[i].votes<<endl;
		}
		*/
		/*
		if(line_objects.size() ==2 && fabs(max(line_objects[0].startpoint.x, line_objects[0].endpoint.x) - max(line_objects[1].startpoint.x, line_objects[1].endpoint.x)) < 5)
		{
			if(line_objects[1].votes > line_objects[0].votes)
			{
				line_objects.erase(line_objects.begin());
			}
			else
			{
				line_objects.erase(line_objects.begin() + 1);
			}

		}
		else if(line_objects.size() == 2 && fabs(max(line_objects[0].startpoint.x, line_objects[0].endpoint.x) - max(line_objects[1].startpoint.x, line_objects[1].endpoint.x)) > max_distance_two_side_lanes)
		{
			line_objects.erase(line_objects.begin() + 1);
		}
		else
		{
		*/
			/*
			for(int i = 0;i<line_objects.size();i++)
			{
				if(abs(line_objects[i].startpoint.x - line_objects[i].endpoint.x) > 20)
				{
				//	cout<<line_objects[i].startpoint.x<<"\t"<<line_objects[i].endpoint.x<<"\t"<<endl;
					line_objects.erase(remove_if(line_objects.begin(), line_objects.end(), [&] (const Line& lhs){return (lhs.startpoint.x == line_objects[i].startpoint.x) && (lhs.endpoint.x == line_objects[i].endpoint.x) ;}),line_objects.end());
				}

			}
			vector<int4> x_points_arr(line_objects.size());
			vector<int> x_points_arr_1(line_objects.size());

			for(int i =0;i<line_objects.size();i++)
			{
				x_points_arr[i] = {abs(line_objects[i].startpoint.x - line_objects[i].endpoint.x), line_objects[i].startpoint.x, line_objects[i].endpoint.x, line_objects[i].votes};
				x_points_arr_1[i] = max(line_objects[i].startpoint.x, line_objects[i].endpoint.x);

			}
		

			vector<int> diff_array_1(line_objects.size());
			adjacent_difference(x_points_arr_1.begin(), x_points_arr_1.end(), diff_array_1.begin());
			print_int_vector(diff_array_1);
			sort(x_points_arr.begin(), x_points_arr.end(), [](const int4& lhs, const int4& rhs){return (lhs.w  > rhs.w) ;});
			*/
			/*
			for(int i = 0;i<x_points_arr.size();i++)
			{
				cout<<"Values \t"<<x_points_arr[i].x<<"\t"<<x_points_arr[i].y<<"\t"<<x_points_arr[i].z<<"\t"<<x_points_arr[i].w<<endl;

			}
			*/
		//	int curr_votes_i = x_points_arr[i].w;
			/*
			for(int i = 0;i<x_points_arr.size();i++)
			{
				if(x_points_arr[i].x > 10 || x_points_arr[i].y < 35 || x_points_arr[i].z < 35 || x_points_arr[i].y > 140 || x_points_arr[i].z > 140)
				{
					line_objects.erase(remove_if(line_objects.begin(), line_objects.end(), [&] (const Line& lhs){return (lhs.startpoint.x == x_points_arr[i].y) && (lhs.endpoint.x == x_points_arr[i].z) ;}),line_objects.end());

				}
		
			}
			*/
	//	}
	
		/*
		for(int i =0;i<line_objects.size();i++)
		{
			cout<<"X_coordinate \t"<<line_objects[i].startpoint.x<<"\t"<<line_objects[i].endpoint.x<<endl;	
			//cout<<line_objects[i].votes<<endl;
		}
		*/
	//}

}


line_coord* getLineEndPoints(float rho, float theta_line, int image_width, int image_height)
{

	line_coord* line_end_points = (line_coord*)malloc(sizeof(line_coord));

	int xup, xdown, yleft, yright;
	if(cos(theta_line) == 0)
	{
		xup = image_width;
		xdown = image_width;
	}
	else
	{
		xup = (int)(rho/cos(theta_line));
		xdown = (int)((rho-image_height*sin(theta_line))/cos(theta_line));
	}

	if(sin(theta_line) == 0)
	{
		yleft = image_height;
		yright = image_height;
	}
	else
	{
		yleft = (int)(rho/sin(theta_line));
		yright = (int)((rho - image_width*cos(theta_line))/sin(theta_line));
	}
		
	Linepoint points[4];
	points[0] = {xup,0};
	points[1] = {xdown, image_height};
	points[2] = {0, yleft};
	points[3] = {image_width, yright};
	int i = 0;

	for(i = 0;i<4;i++)
	{
		if(isPointInside(points[i], image_width, image_height))
		{
			line_end_points->startpoint = points[i];
			break;
	
		}

	}

	for(int j = i + 1;j<4;j++)
	{
		if(isPointInside(points[j], image_width, image_height))
		{
			line_end_points->endpoint = points[j];
			break;
		}

	}

	//cout<<"Line Endpoints"<<"("<<line_end_points->startpoint.x<<","<<line_end_points->startpoint.y<<")"<<endl;
	//cout<<"Line Endpoints"<<"("<<line_end_points->endpoint.x<<","<<line_end_points->endpoint.y<<")"<<endl;

	return line_end_points;
	



};

bool isPointInside(Linepoint point, int image_width, int image_height)
{
	if(point.x >=0  && point.x <= image_width && point.y >=0 && point.y<= image_height)
		return true;
	else
		return false;
	
}

