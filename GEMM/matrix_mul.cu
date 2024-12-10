// #define USE_CUBLAS

#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>
#ifdef USE_CUBLAS
#include <cublas_v2.h>
#endif
#include <device_launch_parameters.h>
#include <cmath>
#include <nvtx3/nvToolsExt.h>
using namespace std;

const int TILE_WIDTH = 16;	// 定义块block大小

/////////
// Matrix multiplication with shared memory (CUDA Kernel) on the device: C = A * B
/////////
const int BLOCK_SIZE = TILE_WIDTH;
__global__ void MatrixMulSharedMemKernel(float *A,
    float *B, float *C, int wA,
    int wB) {



}


//! For square matrices only
__global__ void MatrixMulKernel(float* d_M, float* d_N, float* d_P, int m, int n, int k)
{
  // 计算全局索引
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // 严格的边界检查
  if (row < m && col < n) {
    float pValue = 0.0f;
    
    // 计算一个元素
    for (int i = 0; i < k; i++) {
      pValue += d_M[row * k + i] * d_N[i * n + col];
    }
    
    // 存储结果
    d_P[row * n + col] = pValue;
  }
}

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set matrix multiply on CPU
//! C = A * B
//! @param C          reference data, computed but preallocated
//! @param A          matrix A as provided to device
//! @param B          matrix B as provided to device
//! @param hA         height of matrix A
//! @param wA         width of matrix A
//! @param wB         width of matrix B
////////////////////////////////////////////////////////////////////////////////
void
matrixMulCPU(float *C, const float *A, const float *B, unsigned int hA, unsigned int wA, unsigned int wB)
{
    for (unsigned int i = 0; i < hA; ++i)
        for (unsigned int j = 0; j < wB; ++j)
        {
            double sum = 0;

            for (unsigned int k = 0; k < wA; ++k)
            {
                double a = A[i * wA + k];
                double b = B[k * wB + j];
                sum += a * b;
            }

            C[i * wB + j] = (float)sum;
        }
}

void printDiff(float *data1, float *data2, int width, int height, int iListLength, float fListTol)
{
    printf("Listing first %d Differences > %.6f...\n", iListLength, fListTol);
    int i,j,k;
    int error_count=0;

    for (j = 0; j < height; j++)
    {
        for (i = 0; i < width; i++)
        {
            k = j * width + i;
            float fDiff = fabs(data1[k] - data2[k]);

            if (fDiff > fListTol)
            {
                if (error_count < iListLength)
                {
                    printf("    Loc(%d,%d)\tCPU=%.5f\tGPU=%.5f\tDiff=%.6f\n", i, j, data1[k], data2[k], fDiff);
                }

                error_count++;
            }
        }
    }

    printf(" \n  Total Errors = %d\n", error_count);
}

void getArg(int argc, char* argv[], int &size, int &check)
{
  if (argc != 3)
  {
    cerr << "Usage: " << argv[0] << " <check_enable> <size>\n";
    cerr << "\tcheck_enable: 1 to enable result checking\n";
    cerr << "\tsize: size of the matrix\n";
    exit(1);
  }

  int val1, val2;
  try
  {
    val1 = stoi(argv[1]);
    val2 = stoi(argv[2]);
  }
  catch (const invalid_argument& e)
  {
    cerr << "ERROR: parameters should be integer\n";
    exit(1);
  }

  check = val1;
  size = val2;
}



int main(int argc, char* argv[])
{
  int size, check;
  getArg(argc, argv, size, check);

  int m = size, n = size, k = size;
  printf("size: %d, check: %d\n", size, check);
  
  // 声明存放在GPU上的数组
  float *h_M, *h_N, *d_M, *d_N;
  float *h_P, *d_P;
  
  size_t sizeM = m * k * sizeof(float);
  size_t sizeN = k * n * sizeof(float);
  size_t sizeP = m * n * sizeof(float);

  // Allocate host memory
  h_M = (float*) malloc(sizeM);
  h_N = (float*) malloc(sizeN);
  h_P = (float*) malloc(sizeP);
  float *reference = (float *)malloc(sizeP);

  // Allocate device memory
  cudaMalloc(&d_M, sizeM);
  cudaMalloc(&d_N, sizeN);
  cudaMalloc(&d_P, sizeP);

  // Init data 
  for(int i = 0; i < m * n; ++i)
  {
    if(i % 2 == 0)
      h_M[i] = 1.0;
    else
      h_M[i] = 0.5;
  }

  for(int i = 0;i < n * k; ++i)
  {
    if(i % 2 == 0)
      h_N[i] = 0.5;
    else
      h_N[i] = 1.0;
  }


  // Launch kernel 定义grid&block
  dim3 grid(
    (n + TILE_WIDTH - 1) / TILE_WIDTH,  // 列方向
    (m + TILE_WIDTH - 1) / TILE_WIDTH   // 行方向
  );
  dim3 block(TILE_WIDTH, TILE_WIDTH);

  // Copy data from CPU to GPU
  cudaMemcpy(d_M, h_M, sizeM, cudaMemcpyHostToDevice);
  cudaMemcpy(d_N, h_N, sizeN, cudaMemcpyHostToDevice);

  // 添加 warmup
  nvtxRangePushA("Warmup Start");
  printf("Warming up...\n");
  for (int i = 0; i < 3; i++) {
      MatrixMulKernel<<<grid, block>>>(d_M, d_N, d_P, m, n, k);
  }
  cudaDeviceSynchronize();
  nvtxRangePop();

// Timing records 
  nvtxRangePushA("Kernel Execution Start");
  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);

  int nIter = 5;
#ifdef USE_CUBLAS
  cublasHandle_t handle;
  cublasCreate(&handle);
#endif
  const float alpha = 1.0f;
  const float beta  = 0.0f;
  for (int j = 0; j < nIter; j++) {
    //matrixMulCPU(reference, h_M, h_N, m, k, n);
    MatrixMulKernel<<<grid, block>>>(d_M, d_N, d_P, m, n, k);
    //MatrixMulSharedMemKernel<<<grid, block>>>(d_M, d_N, d_P, m, n);
    //cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_N, n, d_M, k, &beta, d_P, n);
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  nvtxRangePop();
  float msecPerMatrixMul;
  cudaEventElapsedTime(&msecPerMatrixMul, start, stop);
  msecPerMatrixMul /= nIter;
  printf("Kernel Elapsed Time: %.3f ms\n", msecPerMatrixMul);

  // Compute and print the performance
  double flopsPerMatrixMul = 2.0 * (double)m * (double)n * (double)k;
  double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
  printf("Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n",
		  gigaFlops,
		  msecPerMatrixMul,
		  flopsPerMatrixMul);

  // Copy data from GPU to CPU 
  cudaMemcpy(h_P, d_P, sizeP, cudaMemcpyDeviceToHost);

  // compute reference solution
  if (check == 1)
  {
    printf("Computing result using host CPU...");
    matrixMulCPU(reference, h_M, h_N, m, k, n);
    printf("done.\n");
    printDiff(reference, h_P, n, m, 100, 1.0e-5f);
  }

  free(h_P);
  free(h_M);
  free(h_N);
  cudaFree(d_P);
  cudaFree(d_M);
  cudaFree(d_N);
#ifdef USE_CUBLAS
  cublasDestroy(handle);
#endif

  
  // 添加错误检查
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
      printf("CUDA Error: %s\n", cudaGetErrorString(err));
      return -1;
  }

  return 0;
}

