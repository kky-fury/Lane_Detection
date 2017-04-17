#include"matrix_mul.hpp"

void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {
	
		// Create a pseudo-random number generator
		curandGenerator_t prng;
		curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
		// Set the seed for the random number generator using the system // clock
		curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());
		// Fill the array with random numbers on the device
		curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);

}


void gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int k, const int n)
{
	int lda=m,ldb=k,ldc=m;
	const float alf = 1;
	const float bet = 0;
	const float *alpha = &alf;
	const float *beta = &bet;

	cublasHandle_t handle;
	cublasCreate(&handle);

	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B,ldb, beta, C, ldc);

	cublasDestroy(handle);



}


void print_matrix(const thrust::device_vector<float> &A, int nr_rows_A, int nr_cols_A) 
{

for(int i = 0; i < nr_rows_A; ++i)
{
	for(int j = 0; j < nr_cols_A; ++j)
	{
		std::cout << A[j * nr_rows_A + i] << " ";
			
	}
	std::cout << std::endl;
}
    
std::cout << std::endl;

}




























int main()
{
	int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;
	
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	nr_rows_A =3;
	nr_cols_A = 3;
	nr_rows_B = 3;
	nr_cols_B = 80000;

	nr_rows_C = 3;
	nr_cols_C = 80000;

	thrust::device_vector<float> d_A(nr_rows_A * nr_cols_A), d_B(nr_rows_B *nr_cols_B), d_C(nr_rows_C * nr_cols_C);


	GPU_fill_rand(thrust::raw_pointer_cast(&d_A[0]), nr_rows_A, nr_cols_A);
	GPU_fill_rand(thrust::raw_pointer_cast(&d_B[0]), nr_rows_B, nr_cols_B);

	/*std::cout << "A =" << std::endl;
	print_matrix(d_A, nr_rows_A, nr_cols_A);
	std::cout << "B =" << std::endl;
	print_matrix(d_B, nr_rows_B, nr_cols_B);
	*/
	
	cudaEventRecord(start, 0);
	gpu_blas_mmul(thrust::raw_pointer_cast(&d_A[0]),thrust::raw_pointer_cast(&d_B[0]),thrust::raw_pointer_cast(&d_C[0]), nr_rows_A, nr_cols_A, nr_cols_B);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&time, start, stop);
	printf ("Time for the kernel: %f ms\n", time);

	std::cout << "C =" << std::endl;
	//print_matrix(d_C, nr_rows_C, nr_cols_C);

	return 0;




}
