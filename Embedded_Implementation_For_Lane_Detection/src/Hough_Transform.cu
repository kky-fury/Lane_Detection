/*
*Adapted From "Fast Hough Trasform on GPU's"
*
*/
#include"hough.hpp"
#include"cuda_error_check.hpp"
bool debug_hough = false;
#define THREADS_X_HOUGH	32
#define THREADS_Y_HOUGH	4
#define PIXELS_PER_THREAD 16

__device__ static int g_counter;
__device__ static int g_counter_lines;
extern __shared__ int shmem[];


void print_array(float *arr, int size)
{
	for(int i =0;i<size;i++)
	{
		cout<<*(arr + i)<<"\t";
	}

	cout<<endl;

}

void print_image(unsigned char *image, int height, int width)
{


	for(int i =0;i<height;i++)
	{
		for(int j =0;j<width;j++)
		{
			cout<<(int)*(image + i*width + j)<<"\t";

		}
	
		cout<<endl;
	}

}


__global__ void getNonzeroEdgepoints(unsigned char const* const image, unsigned int* const list)
{

	
	__shared__ unsigned int s_queues[THREADS_Y_HOUGH][THREADS_X_HOUGH * PIXELS_PER_THREAD];
	__shared__ int s_qsize[THREADS_Y_HOUGH];
	__shared__ int s_globStart[THREADS_Y_HOUGH];

	const int x = blockIdx.x * blockDim.x * PIXELS_PER_THREAD + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(threadIdx.x == 0)
		s_qsize[threadIdx.y] = 0;
	__syncthreads();

	if(y < IMG_HEIGHT)
	{	
		const unsigned char* srcRow = image + y*IMG_WIDTH;
		for(int i = 0,xx = x; i<PIXELS_PER_THREAD && xx < IMG_WIDTH;++i,xx +=
				blockDim.x)
		{
			if(srcRow[xx])
			{
				const unsigned int val = (y<<16)|xx;
				const int qidx = atomicAdd(&s_qsize[threadIdx.y],1);
				s_queues[threadIdx.y][qidx] = val;

			}


		}

	}

	__syncthreads();

	if(threadIdx.x == 0 && threadIdx.y == 0 )
	{	
		int totalSize = 0;
		for(int i =0;i<blockDim.y;++i)
		{
				s_globStart[i] = totalSize;
				totalSize += s_qsize[i];	

		}
		
		const int global_Offset = atomicAdd(&g_counter, totalSize);
		for(int i  =0 ;i<blockDim.y;++i)
			s_globStart[i] += global_Offset;
	}

	__syncthreads();

	const int qsize = s_qsize[threadIdx.y];
	int gidx = s_globStart[threadIdx.y] +  threadIdx.x;
	for(int i = threadIdx.x; i<qsize; i+=blockDim.x, gidx +=blockDim.x)
	{
		list[gidx] = s_queues[threadIdx.y][i];

	}

}

__global__ void fillHoughSpace(unsigned int* const list, const int count, int* hough_space,const float irho, const float theta, const int numrho)
{

	int* smem = (int*)shmem;
	for(int i =threadIdx.x; i< numrho + 1;i+=blockDim.x)
		smem[i] = 0;
	__syncthreads();
	

	const int n = blockIdx.x;
	const float ang = n*theta;
	
	//printf("The angle value of n is %d \n", blockIdx.x);

	//printf("Angle Values : %f \n", ang);
	//printf("Inside Kernel");
	
	
	float sinVal;
	float cosVal;

	sincosf(ang, &sinVal, &cosVal);
	sinVal *= irho;
	cosVal *= irho;

	const int shift = (numrho -1)/2;

	for(int i  = threadIdx.x; i<count; i+= blockDim.x)
	{
		const unsigned int val = list[i];

		const int x = (val & 0xFFFF);
		const int y = (val >> 16) & 0xFFFF;
		int r = __float2int_rn(x*cosVal + y*sinVal);
		//printf("The value of x %d and the value of y %d : the value of r %d \n",x,y,r);
		r += shift;		
		atomicAdd(&smem[r+1],1);
	}
	
	__syncthreads();

	int* hrow = hough_space + (n+1)*(numrho + 2);
	for(int i = threadIdx.x ;i< numrho + 1; i+=blockDim.x)
	{	
		//printf("value of shared_memory at %d is %d \n",i,smem[i]);
		hrow[i] = smem[i];
	}
	

}


/*Non Maximum Suppression to get Valid Values*/
__global__ void getLines(const int * hough_space, float2* lines, int* votes, const int
		maxLines, const float rho, const float theta, const int threshold, const
		int numrho, const int rhspace)
{
	
	const int r = blockIdx.x*blockDim.x + threadIdx.x;
	const int n = blockIdx.y*blockDim.y + threadIdx.y;
	
	

	if(r >=numrho || n >=rhspace -2)
	{
		return;
	}

	const int curVotes = *(hough_space + (n+1)*(numrho + 2)+ (r+1));

	if(curVotes > *(hough_space + n*(numrho+2) + (r-1)) && 
			curVotes > *(hough_space + n*(numrho + 2) + r) && 
			curVotes > *(hough_space + n*(numrho + 2)+(r+1)) && 
			curVotes > *(hough_space + n*(numrho + 2) + (r+2)) && 
			curVotes > *(hough_space + n*(numrho+2) + (r+3)) && 
			curVotes > *(hough_space + (n+1)*(numrho +2)+ r-1) && 
			curVotes > *(hough_space + (n+1)*(numrho + 2) + r) && 
			curVotes > *(hough_space +(n+1)*(numrho +2) + (r+2)) && 
			curVotes > *(hough_space +(n+1)*(numrho +2) + (r+3)) && 
			curVotes > *(hough_space +(n+2)*(numrho +2) + (r-1)) && 
			curVotes > *(hough_space + (n+2)*(numrho +2) + r) && 
			curVotes > *(hough_space + (n+2)*(numrho +2) + (r+1)) && 
			curVotes > *(hough_space + (n+2)*(numrho +2) + (r+2)) && 
			curVotes > *(hough_space + (n+2)*(numrho +2) + (r+3)) && curVotes > threshold)
	{
		const float radius = (r - (numrho -1)*0.5f)*rho;
		const float angle = n*theta;

		const int index = atomicAdd(&g_counter_lines,1);
		if(index < maxLines)
		{
			//printf("index Value - %d \n", index);
			//printf("Current Votes - %d \n", curVotes);
			//printf("radius %f and angle %f \n", radius, angle);
			//*(lines + index) = make_float2(radius, angle);
			(lines +  index)->x = radius;
			(lines + index)->y = angle;
			//printf("value of radius - %f and value of angle - %f and curVotes - %d \n ", (lines +index)->x,(lines + index)->y, curVotes);
			*(votes + index) = curVotes;

		}
		


	}




}

lines_w_non_zero* houghTransform(unsigned char const* const edges,const int numangle, const int numrho,float thetaStep, float rStep)
{
	/*	if(debug_hough)
		{
			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaEventRecord(start,0);
		
		}
	*/
		/*Replace by maximum function using cuda*/
		const int threshold = 20;

		unsigned char* gimage;	
		unsigned int* glist; 

		void* counterPtr;
		cudaGetSymbolAddress(&counterPtr, g_counter);


		cudaMemset(counterPtr,0,sizeof(int));
		CudaCheckError();

		cudaFuncSetCacheConfig(getNonzeroEdgepoints, cudaFuncCachePreferShared);
			
		cudaMalloc((void**)&gimage, IMG_SIZE*sizeof(unsigned char));
		CudaCheckError();
	
		cudaMalloc((void**) &glist, IMG_SIZE*sizeof(unsigned int));
		CudaCheckError();
	
		/*Copy Image to GPU */	
	
		cudaMemcpy(gimage, edges, IMG_SIZE*sizeof(unsigned char),cudaMemcpyHostToDevice);
		CudaCheckError();
		
		dim3 dimBlock1(THREADS_X_HOUGH, THREADS_Y_HOUGH);
	
		//dim3 dimGrid1(1, 56);
		dim3 dimGrid1((IMG_WIDTH + THREADS_X_HOUGH*PIXELS_PER_THREAD
					-1)/(THREADS_X_HOUGH*PIXELS_PER_THREAD), (IMG_HEIGHT +
						THREADS_Y_HOUGH -1)/(THREADS_Y_HOUGH));
		
		getNonzeroEdgepoints<<<dimGrid1,dimBlock1>>>(gimage, glist);
		CudaCheckError();
		cudaDeviceSynchronize();

		int totalCount ;
		cudaMemcpy(&totalCount, counterPtr, sizeof(int),cudaMemcpyDeviceToHost);
		//cout<<"Total Count :"<<totalCount<<endl;

		unsigned int* clist = (unsigned int*)malloc(totalCount*sizeof(unsigned int));
		cudaMemcpy(clist, glist, totalCount*sizeof(unsigned int),cudaMemcpyDeviceToHost);
		CudaCheckError();
		
		if(debug_hough)
		{
			unsigned int* clist = (unsigned int*)malloc(totalCount*sizeof(unsigned int));
			cudaMemcpy(clist, glist, totalCount*sizeof(unsigned int),cudaMemcpyDeviceToHost);
			CudaCheckError();

			for(int i = 0; i< totalCount; i++)
			{	
				unsigned int const q_value = clist[i];
				cout<<"q_value : "<<q_value<<endl;
				const int x = (q_value & 0xFFFF);
				const int y = (q_value >> 16 ) & 0xFFFF;
				cout<<"coordinate ("<<x<<","<<y<<")"<<endl;
				cout<<"Value at coordinate :"<<(int)*(edges + y*IMG_WIDTH + x)<<endl;
			}

		
		}

		//Initialize hough_space
		int hough_size = (numangle + 2)*(numrho + 2);	
		int rhspace = numangle + 2;
		int colhspace = numrho + 2;
		
		//cout<<"rows : "<<rhspace<<endl;

		const dim3 block(1024);
		const dim3 grid(rhspace -2);

		//smemSize should be less than 49152 bytes

		size_t smemSize = (colhspace - 1)*sizeof(int);
		cout<<smemSize<<endl;

		thetaStep = thetaStep*(CV_PI/180);
	
		/*Allocate houghSpace on Gpu*/
		int *d_hough_space;

		cudaMalloc((void**)&d_hough_space,hough_size*sizeof(int));
		CudaCheckError();
	
		cudaMemset(d_hough_space, 0, hough_size*sizeof(int));
		CudaCheckError();
		
		fillHoughSpace<<<grid,block, smemSize>>>(glist, totalCount,d_hough_space, 1.0f/rStep, thetaStep, colhspace -2);
		CudaCheckError();

		cudaDeviceSynchronize();

	
		if(debug_hough)
		{
			int* hough_space = (int*)malloc(hough_size*sizeof(int));
			cudaMemcpy(hough_space, d_hough_space, hough_size*sizeof(int),cudaMemcpyDeviceToHost);
			CudaCheckError();
	
			for(int i =0;i<rhspace;i++)
			{	
				for(int j =0;j<colhspace;j++)
				{
					cout<<*(hough_space + i*colhspace +j)<<"\t";
	
				}
			
				cout<<endl;

			}
		}
	

		int maxLines = 75;
			
		float2* d_lines;
		int* d_votes;

		cudaMalloc((void**)&d_lines,maxLines*sizeof(float2));
		CudaCheckError();	

		cudaMalloc((void**)&d_votes, maxLines*sizeof(int));
		CudaCheckError();

		void *counterPtr_lines;			
		cudaGetSymbolAddress(&counterPtr_lines, g_counter_lines);
		
		cudaMemset(counterPtr_lines, 0, sizeof(int));
		CudaCheckError();

		const dim3 block_1(32,8);
		const int blocks_x = ((colhspace - 2 + block_1.x - 1)/(block_1.x));
		const int blocks_y = ((rhspace - 2 + block_1.y -1 )/(block_1.y));
		const dim3 grid_1(blocks_x, blocks_y);
			
		
		cudaFuncSetCacheConfig(getLines, cudaFuncCachePreferL1);
		getLines<<<grid_1, block_1>>>(d_hough_space, d_lines, d_votes, maxLines,rStep, thetaStep, threshold, colhspace -2, rhspace);
		CudaCheckError();	
		cudaDeviceSynchronize();

		int countlines;

		cudaMemcpy(&countlines, counterPtr_lines, sizeof(int),cudaMemcpyDeviceToHost);
		CudaCheckError();
		
		countlines = min(countlines, maxLines);
	
		float2* lines = (float2*)malloc(countlines*sizeof(float2)); 
		int* votes = (int*)malloc(countlines*sizeof(int));

		cudaMemcpy(lines, d_lines, countlines*sizeof(float2),cudaMemcpyDeviceToHost);
		CudaCheckError();	
		
		cudaMemcpy(votes, d_votes, countlines*sizeof(int),cudaMemcpyDeviceToHost);
		CudaCheckError();

		map<float, vector<floatwint>> theta_to_votes_map;	
		for(int i =0;i<countlines;i++)
		{
			floatwint obj = {(float)(lines + i)->x,  *(votes + i) };
			theta_to_votes_map[(lines + i)->y].push_back(obj);
		}
		
		for(auto it = theta_to_votes_map.begin() ;it!= theta_to_votes_map.end(); ++it)
		{
			sort(it->second.begin() , it->second.end(), [](const floatwint& lhs, const floatwint& rhs) {return lhs.y > rhs.y;});
			//cout<<"Vector Size \t"<<theta_to_votes_map[it->first].size()<<endl;
		}


	
		/*
		int nSelectedLines = theta_to_votes_map.size();
		float2* selLines = (float2*)malloc(nSelectedLines*sizeof(float2));
		int* selvotes = (int*)malloc(nSelectedLines*sizeof(selvotes));
		
		int index = 0;
		*/
		
		/*
		for(auto it = theta_to_votes_map.begin(); it!= theta_to_votes_map.end(); ++it)
		{
			(selLines + index)->x =  (it->second.begin())->x;
			(selLines + index)->y =  it->first;
			*(selvotes + index) = (it->second.begin())->y;
			index++;
		}
	*/
		auto it = theta_to_votes_map.begin();
		auto it_1 = next(it);
		int index = 0;
		/*count number of lines above 50 votes*/
		int count_lines = 0;
		int threshold_votes = 35;

		for(auto it2 = it->second.begin(); it2 != it->second.end(); ++it2)
		{
			if(it2->y > threshold_votes)
				count_lines++;
		}

		for(auto it2 = it_1->second.begin(); it2 != it_1->second.end();++it2)
		{
			if(it2->y > threshold_votes)
				count_lines++;

		}
		
		cout<<"Count of Lines \t"<<count_lines<<endl;

		int nSelectedLines;
		float2* selLines;
		int* selvotes;

		//int nSelectedLines = theta_to_votes_map[it->first].size() + theta_to_votes_map[it_1->first].size();
		if(count_lines > 0)
		{
			nSelectedLines  = count_lines;
			selLines = (float2*)malloc(nSelectedLines*sizeof(float2));
			selvotes = (int*)malloc(nSelectedLines*sizeof(selvotes));

			for(auto it2 = it->second.begin(); it2 != it->second.end(); ++it2)
			{
				if(it2->y > threshold_votes)
				{
					(selLines + index)->x =  (it2)->x;
					(selLines + index)->y = it->first;
					*(selvotes + index) = (it2)->y;
					index++;
				}
			}
		
			for(auto it2 = it_1->second.begin(); it2 != it_1->second.end();++it2)
			{
				if(it2->y > threshold_votes)
				{
					(selLines + index)->x =  (it2)->x;
					(selLines + index)->y = it_1->first;
					*(selvotes + index) = (it2)->y;
					index++;
				}
			}	
		}
		else
		{
			/*
			if(theta_to_votes_map[it->first].size() > 2)
			{
				nSelectedLines += 2;
			}
			*/
			//nSelectedLines  = theta_to_votes_map[it->first].size() + theta_to_votes_map[it_1->first].size();
		
			if(debug_hough)
			{
				cout<<"Key value \t"<<it->first<<" \t Vector Values \t"<<endl;
				cout<<"Size \t"<<theta_to_votes_map[it->first].size()<<endl;
			}

			nSelectedLines  = theta_to_votes_map[it->first].size();
			selLines = (float2*)malloc(nSelectedLines*sizeof(float2));
			selvotes = (int*)malloc(nSelectedLines*sizeof(selvotes));
		
			for(auto it2 = it->second.begin(); it2 != it->second.end(); ++it2)
			{
				(selLines + index)->x =  (it2)->x;
				(selLines + index)->y = it->first;
				*(selvotes + index) = (it2)->y;
				index++;
			}

		/*
			for(auto it2 = it_1->second.begin(); it2 != it_1->second.end();++it2)
			{
				(selLines + index)->x =  (it2)->x;
				(selLines + index)->y = it_1->first;
				*(selvotes + index) = (it2)->y;
				index++;
			}

			*/
		}
	
		if(debug_hough)
		{
			for(int i = 0 ; i < index;i++)
			{
				cout<<"Theta Value \t"<<(selLines + i)->y<<"\t"<<"Rho Value\t"<<(selLines + i)->x<<endl;
				cout<<"Votes \t"<<*(selvotes + i)<<endl;
			}
		}

		if(debug_hough)
		{
			for(auto it = theta_to_votes_map.begin() ;it!= theta_to_votes_map.end(); ++it)
			{
				cout<<"Key value \t"<<it->first<<" \t Vector Values \t";
	
				for(auto it2 = it->second.begin(); it2 != it->second.end(); ++it2)
				{
					cout<<it2->x<<"\t"<<it2->y<<"\t";
				}
	
				cout<<endl;
			}
		}
		
		if(debug_hough)
		{
			Mat gray_image = imread("/home/nvidia/Lane_Detection/Test_Images/IPM_test_image_4.png",0);
		
			for(int i =0;i<countlines;i++)
			{
				float theta_line = (lines + i)->y;
				float rho = (lines + i)->x;
				int curr_votes =  *(votes + i);

				cout<<"Rho - "<<rho<<" \t theta- "<<theta_line<<endl;
				cout<<"Corresponding Votes \t"<<curr_votes<<endl;

				cv::Point pt1, pt2;
	
				double a = cos(theta_line);
				double b = sin(theta_line);

				double x0 = a*rho;
				double y0 = b*rho;
	
				pt1.x = (int)(x0 + 400*(-b));
				pt1.y = (int)(y0 + 400*(a));
				pt2.x = (int)(x0 - 400*(-b));
				pt2.y = (int)(x0 - 400*(a));
				
				
				line(gray_image, pt1,pt2, (255,0,0),1);
			}
			imshow("Image", gray_image);
			waitKey(0);

		}

		lines_w_non_zero* values = (lines_w_non_zero*)malloc(sizeof(lines_w_non_zero));
		lin_votes* mem_hough_lines = (lin_votes*)malloc(sizeof(lin_votes));
		values->hough_lines = mem_hough_lines;
		//values->hough_lines->lines = lines;
		values->hough_lines->lines =  selLines;

		//values->hough_lines->countlines = countlines;
		values->hough_lines->countlines = nSelectedLines;

		values->clist = clist;
		values->count = totalCount;
		values->votes = selvotes;
		//values->votes = votes;
		/*
		lin_votes* hough_lines = (lin_votes*)malloc(sizeof(lin_votes));
		hough_lines->lines = lines;
		hough_lines->countlines = countlines;
		*/
		
	
	/*	
		if(debug_hough)
		{	
			cudaEventRecord(stop,0);
			cudaEventSynchronize(stop);

			float elapsed = 0;
			cudaEventElapsedTime(&elapsed, start, stop);

			cout<<"Elapsed Time"<<elapsed;
		}
	
	*/

		return values;

}


