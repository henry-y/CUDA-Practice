#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <device_launch_parameters.h>

/*
    Naive GEMM Kernel
    each thread computes one element of the result matrix C
    load one row of A and one column of B into shared memory

    A: [n, k]
    B: [k, m]
    C: [n, m]
*/
__global__ void NaiveGEMMKernel(float *A, float *B, float *C, int n, int m, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / m;
    int col = idx % m;

    float c = 0;
    for (int i = 0; i < k; i++) {
        c += A[row * k + i] * B[i * m + col];
    }
    C[row * m + col] = c;
}

/*
    Tiling GEMM Kernel
    each thread block computes one tile of the result matrix C
    each thread computes one element of the result matrix C
    load one row of A and one column of B into shared memory
*/
__global__ void TilingGEMMKernel(float *A, float *B, float *C, int n, int m, int k) {

}

int main() {

    int n, m, k;

    n = 10000;
    m = 10000;
    k = 10000;

    float *h_A = (float *)malloc(n * k * sizeof(float));
    float *h_B = (float *)malloc(k * m * sizeof(float));
    float *h_C = (float *)malloc(n * m * sizeof(float));    

    for (int i = 0; i < n * m; i++) {
        h_A[i] = rand() / RAND_MAX;
    }

    for (int i = 0; i < m * k; i++) {
        h_B[i] = rand() / RAND_MAX;
    }

    for (int i = 0; i < n * k; i++) {
        h_C[i] = 0;
    }

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, n * k * sizeof(float));
    cudaMalloc(&d_B, k * m * sizeof(float));
    cudaMalloc(&d_C, n * m * sizeof(float));

    cudaMemcpy(d_A, h_A, n * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, k * m * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(1024);
    dim3 blocksPerGrid(108);
    // Warmup
    NaiveGEMMKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n, m, k);
    cudaDeviceSynchronize();

    // 计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);

    NaiveGEMMKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n, m, k);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Naive GEMM Kernel Time: %f ms\n", elapsedTime);

    cudaMemcpy(h_C, d_C, n * m * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}