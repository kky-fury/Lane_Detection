#ifndef MATRIX_MUL_HPP
#define MATRIX_MUL_HPP

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cublas_v2.h>
#include <curand.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>



void GPU_fill_rand(float*, int, int);
void gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int k, const int n);
void print_matrix(const thrust::device_vector<float> &A, int nr_rows_A, int nr_cols_A) ;




























#endif

