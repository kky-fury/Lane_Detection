#include"read_file.hpp"

bool debug_file = false;

matrix_t readcalibfile(string str)
{
	ifstream infile(str);
	string line;

	matrix_t Tr_cam_to_road(3, row_t(4,0));
	
	while(getline(infile, line))
	{
		vector<string> strs;
		boost::split(strs, line, boost::is_any_of(":"));
		if(strs[0].compare("Tr_cam_to_road") == 0)
		{	
			vector<string> tokens;
			boost::split(tokens, strs[1], boost::is_any_of(" "));
			
			if(debug_file)
			{
				for(int i = 0;i<tokens.size();i++)
				{	
					cout<<tokens[i]<<endl;
				}

			}
			
			tokens.erase(tokens.begin());
	
			for(int i = 0; i < 3;i++)
			{
				for( int j = 0; j < 4;j++)
				{
					Tr_cam_to_road[i][j] = atof(tokens[i*4 + j].c_str());	
					if(debug_file)
						cout<<atof(tokens[i*3 + j].c_str())<<endl;
				}

			}
		//	print2dvector(Tr_cam_to_road);
		}
		if(debug_file)
		{
			for(int i  = 0; i<strs.size();i++)
				cout<<strs[i]<<endl;
		}
		
	}

	return Tr_cam_to_road;


}
