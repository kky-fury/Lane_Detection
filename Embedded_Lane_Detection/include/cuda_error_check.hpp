#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string>
#include <cuda.h>

#ifndef CUDA_ERROR_CHECK_HPP
#define CUDA_ERROR_CHECK_HPP

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall(cudaError err, const char *file, const int line)
{	
	if (cudaSuccess != err)
	{
		fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n",file, line, cudaGetErrorString(err));
		exit(-1);
	}
	return;
}

inline void __cudaCheckError(const char *file, const int line)
{
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n",file, line, cudaGetErrorString(err));
		exit(-1);
	}

	err = cudaDeviceSynchronize();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",file, line, cudaGetErrorString(err));
		exit(-1);
	}

	return;
}


#endif
