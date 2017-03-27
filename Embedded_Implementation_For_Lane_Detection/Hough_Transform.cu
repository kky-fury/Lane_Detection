/*
*Test Code for hough transform using CUDA
*Adapted From "Fast Hough Trasform on GPU's"
*
*
*
*
*/
#include"hough.hpp"
bool debug = false;
#define THREADS_X 	32
#define THREADS_Y	4
#define PIXELS_PER_THREAD 16
#define BLOCKS_X 	(IMG_WIDTH  / (THREADS_X*PIXELS_PER_THREAD))
#define BLOCKS_Y 	(IMG_HEIGHT / THREADS_Y)
#define MAX_QUEUE_LENGTH (THREADS_X*THREADS_Y*PIXELS_PER_THREAD)


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


void print_houghspace(unsigned int* const array, int width)
{
		
		for(int i =0;i<HS_ANGLES;i++)
		{
			for(int j = 0;j<width;j++)
			{
				cout<<array[i*width + j]<<"\t";

			}
			cout<<endl;

		}

}

int getMaximum(unsigned int* const array, int width)
{

	int maximum = *(array + 0);

	for(int i =0;i<HS_ANGLES;i++)
	{
		for(int j =0 ;j<width;j++)
		{
			if(array[i*width + j] > maximum)
				maximum = array[i*width + j];
		

		}
			

	}
	return maximum;

}

/*__global__ void Hough(unsigned char const* const image, unsigned int const
		threshold, unsigned int* const houghspace_1, unsigned int* const houghspace_2)
{
	int const x = blockIdx.x*blockDim.x + threadIdx.x;
	int const y = blockIdx.y*blockDim.y + threadIdx.y;
	__shared__ float sh_m_array[THREADS_X*THREADS_Y];
	int const n = threadIdx.y*THREADS_X + threadIdx.x;

	//Debugging
	//printf("n value : %d \n", n);


	sh_m_array[n]  =  (n-((HS_ANGLES-1)/2.0f)) / (float)((HS_ANGLES-1)/2.0f);
	//printf("shared_array_value : %f \t at postion : %d with thread indexes x: \
	//		%d and \t y : %d \n",sh_m_array[n], n, threadIdx.x, threadIdx.y);
	__syncthreads();

	unsigned char pixel = image[y*IMG_WIDTH + x];
	if(pixel >= threshold)
	{
		for(int n = 0;n<HS_ANGLES;n++)
		{
			float const m = sh_m_array[n];
			int const b1 = x - (int)(y*m) + IMG_HEIGHT;
			int const b2 = y - (int)(x*m) + IMG_WIDTH;
		
			atomicAdd(&houghspace_1[n*HS_1_WIDTH+b1], 1);
			atomicAdd(&houghspace_2[n*HS_2_WIDTH+b2], 1);
		}
	}

	

}
*/

__device__ static int g_counter;
extern __shared__ int shmem[];

__global__ void getNonzeroEdgepoints(unsigned char const* const image, unsigned int* const list)
{

	
	__shared__ unsigned int s_queues[4][32 * PIXELS_PER_THREAD];
	__shared__ int s_qsize[4];
	__shared__ int s_globStart[4];

	const int x = blockIdx.x * blockDim.x * PIXELS_PER_THREAD + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(threadIdx.x == 0)
		s_qsize[threadIdx.y] = 0;
	__syncthreads();

	if(y < 224)
	{	
		const unsigned char* srcRow = image + y*IMG_WIDTH;
		for(int i = 0,xx = x; i<PIXELS_PER_THREAD && xx < 192;++i,xx +=
				blockDim.x)
		{
			if(srcRow[xx])
			{
				const unsigned int val = (y<<16)|xx;
				//Atomic
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

__global__ void getLines(unsigned int* const list, const int count, int*
		hough_space,const float irho, const float theta, const int numrho)
{

	int* smem = (int*)shmem;
	for(int i =threadIdx.x; i< numrho + 1;i+=blockDim.x)
		smem[i] = 0;
	__syncthreads();

	const int n = blockIdx.x;
	const float ang = n*theta;
	
	printf("Angle Values : %f \n", ang);
//	printf("Inside Kernel");
	
	/*
	float sinVal;
	float cosVal;

	sincosf(ang, &sinVal, &cosVal);
	sinVal *= irho;
	cosVal *= irho;

	const int shift = (numrho -1)/2;

	for(int i  = threadIdx.x; i<count; i+= blockDim.x)
	{
		const unsigned int val = list[i];
		const int x = (val & 0x0000FFFF);
		const int y = (val>>16) & 0x0000FFFF;


	}
	
	*/

}


/*__global__ void test_kernel(void)
{

	int x = threadIdx.x;
	printf("%d \n", x);


}

*/
void houghTransform(unsigned char const* const edges,const int numangle, const
		int numrho,float thetaStep, float
		rStep)
{
		unsigned char* gimage;	
		unsigned int* glist; 

		void* counterPtr;
		cudaGetSymbolAddress(&counterPtr, g_counter);
		cudaError_t c_err;


		c_err = cudaMemset(counterPtr,0,sizeof(int));
	
		if(c_err != cudaSuccess)
		{
			printf("%s in %s at line %d \n", cudaGetErrorString(c_err),
				__FILE__,__LINE__);
			exit(EXIT_FAILURE);
		}


		cudaFuncSetCacheConfig(getNonzeroEdgepoints, cudaFuncCachePreferShared);
			
		c_err = cudaMalloc((void**)&gimage, IMG_SIZE*sizeof(unsigned char));
		
		if(c_err != cudaSuccess)
		{
			printf("%s in %s at line %d \n", cudaGetErrorString(c_err),
				__FILE__,__LINE__);
			exit(EXIT_FAILURE);
		}
		
		c_err = cudaMalloc((void**) &glist, IMG_SIZE*sizeof(unsigned int));

		if(c_err != cudaSuccess)
		{
			printf("%s in %s at line %d \n", cudaGetErrorString(c_err),
				__FILE__,__LINE__);
			exit(EXIT_FAILURE);
		}
		
		/*Copy Image to GPU */	
	
		c_err = cudaMemcpy(gimage, edges, IMG_SIZE*sizeof(unsigned char),
			cudaMemcpyHostToDevice);

		if(c_err != cudaSuccess)
		{
			printf("%s in %s at line %d \n", cudaGetErrorString(c_err),
				__FILE__,__LINE__);
			exit(EXIT_FAILURE);
		}

		
		dim3 dimBlock1(THREADS_X, THREADS_Y);
		//dim3 dimGrid1(BLOCKS_X, BLOCKS_Y);
		dim3 dimGrid1(1, 56);
		getNonzeroEdgepoints<<<dimGrid1,dimBlock1>>>(gimage, glist);

		c_err = cudaGetLastError();
		if(c_err != cudaSuccess)
		{
			printf("Error: %s\n", cudaGetErrorString(c_err));
		
		}
		
		cudaDeviceSynchronize();

		int totalCount ;
		cudaMemcpy(&totalCount, counterPtr, sizeof(int),
				cudaMemcpyDeviceToHost);
		cout<<"Total Count :"<<totalCount<<endl;

		if(debug)
		{
			unsigned int* clist = (unsigned int*)malloc(totalCount*sizeof(unsigned int));
			c_err = cudaMemcpy(clist, glist, totalCount*sizeof(unsigned int),cudaMemcpyDeviceToHost);
			if(c_err != cudaSuccess)
			{
					printf("%s in %s at line %d \n", cudaGetErrorString(c_err),
				__FILE__,__LINE__);
					exit(EXIT_FAILURE);
			}
			for(int i = 0; i< totalCount; i++)
			{	
				unsigned int const q_value = clist[i];
				cout<<"q_value : "<<q_value<<endl;
				unsigned int const x = (q_value & 0x0000FFFF);
				unsigned int const y = (q_value >> 16) & 0x0000FFFF;
				cout<<"coordinate ("<<x<<","<<y<<")"<<endl;
				cout<<"Value at coordinate :"<<(int)*(edges + y*IMG_WIDTH + x)<<endl;
			}

		
		}

		//Initialize hough_space
		int hough_size = (numangle + 2)*(numrho + 2);	
		int rhspace = numangle + 2;
		int colhspace = numrho + 2;
		int* hough_space = (int*)calloc(hough_size, sizeof(int));
		
		const dim3 block(1024);
		const dim3 grid(rhspace -2);

		//smemSize should be less than 49152 bytes

		size_t smemSize = (colhspace - 1)*sizeof(int);
		cout<<smemSize<<endl;

		thetaStep = thetaStep*(CV_PI/180);
	
		/*Allocate houghSpace on Gpu*/
		int *d_hough_space;

		c_err = cudaMalloc((void**)&d_hough_space,hough_size*sizeof(int));
		if(c_err != cudaSuccess)
		{
			printf("%s in %s at line %d \n", cudaGetErrorString(c_err),
				__FILE__,__LINE__);
			exit(EXIT_FAILURE);
		}
		

		c_err = cudaMemset(d_hough_space, 0, hough_size*sizeof(int));
		
		if(c_err != cudaSuccess)
		{
			printf("%s in %s at line %d \n", cudaGetErrorString(c_err),
				__FILE__,__LINE__);
			exit(EXIT_FAILURE);
		}

		//cudaFuncSetCacheConfig(getLines,cudaFuncCachePreferShared);
		
		getLines<<<grid,block,smemSize>>>(glist, totalCount,d_hough_space, 1.0f/
				rStep, thetaStep, colhspace -2);
		
		c_err = cudaGetLastError();	
		if(c_err != cudaSuccess)
		{
			printf("Error: %s\n", cudaGetErrorString(c_err));
		
		}	

		cudaDeviceSynchronize();

		


}












int main(int argc, char* argv[])
{

	Mat src_host = imread("/home/nvidia/Binary_test_image_for_cuda_ht.png",
			CV_8UC1);
	cout<<"cols"<<src_host.cols<<endl;
	cout<<"rows"<<src_host.rows<<endl;

	//cout<<src_host<<endl;
	//cout<<src_host.at<unsigned int>(48,34)<<endl;
	int count = 0;
	//cout<<src_host<<endl;
		
	count = countNonZero(src_host);
	cout<<count<<endl;

	Size size = src_host.size();
	int width = size.width;
	int height = size.height;

	if(debug)
	{
		imshow("Result",src_host);
		waitKey(0);
		Size size = src_host.size();
		cout<<size<<endl;
		int width = size.width;
		int height = size.height;	
		cout<<width<<endl;
		cout<<height<<endl;	
	}

	/*Convert array to uchar* (0-255)*/	
	unsigned char *edge_image = src_host.data;
	if(debug)
	{
		print_image(edge_image, height,width);	
	
	}
	//unsigned char* rowptr = edge_image + 2*IMG_WIDTH;
	//cout<<(int)*rowptr<<endl;

	/*unsigned int* houghspace_gpu_1 = (unsigned int*)malloc(HS_1_SIZE*sizeof(unsigned int));
	unsigned int* houghspace_gpu_2 = (unsigned int*)malloc(HS_2_SIZE*sizeof(unsigned int));
	
	unsigned int const threshold = 50;

	houghTransform(edge_image, threshold, houghspace_gpu_1, houghspace_gpu_2);	
	*/
		
	float rMin = 0;
	float rMax = (IMG_WIDTH + IMG_HEIGHT)*2 + 1;
	float rStep = 1.0;

	float thetaMin = 0;
	float thetaMax = 180;
	float thetaStep = 1;
	
	const int numangle = std::round((thetaMax - thetaMin)/thetaStep);
	const int numrho = std::round(rMax/rStep);

	if(1)
	{
		cout<<numangle<<endl;
		cout<<numrho<<endl;
	}

	float* r_values = new float[numrho];
	float* th_vaues = new float[numangle];
	
	int ri, thetai;
	float r, theta;

	for(r = rMin + rStep/2, ri=0;ri<numrho;ri++,r +=rStep)
	{
		r_values[ri] = r;

	}

	for(theta = thetaMin, thetai =0;thetai<numangle;thetai++,theta
			+=thetaStep)
	{
		th_vaues[thetai] =theta;

	}

	if(debug)
	{
		print_array(r_values, numrho);
		print_array(th_vaues, numangle);
	}
	
	//int count = countNonZero(src_host);
	//cout<<count<<endl;	
	
	houghTransform(edge_image, numangle, numrho,thetaStep, rStep);
	
	

	







}
