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

__global__ void MatrixMulSharedMemKernel_v1(float *A,
    float *B, float *C, int wA,
    int wB) {

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int Crow = bx * TILE_WIDTH + ty;
    int Ccol = by * TILE_WIDTH + tx;
    // 写入(Crow, Ccol)

    // 每次读取一个block的A和B
    // (Arow, Acol)
    int AleftRowPoint = bx * TILE_WIDTH;
    int AleftColPoint = 0;
    int AEndColPoint = wA;
    
    int BleftRowPoint = 0;
    int BleftColPoint = by * TILE_WIDTH;

    float cval = 0.0f;

    for(; AleftColPoint < AEndColPoint; AleftColPoint += TILE_WIDTH, BleftRowPoint += TILE_WIDTH) {
        __shared__ float As[TILE_WIDTH+1][TILE_WIDTH+1];
        __shared__ float Bs[TILE_WIDTH+1][TILE_WIDTH+1];
        if(AleftRowPoint + ty < wA && AleftColPoint + tx < wA) {
            As[ty][tx] = A[(AleftRowPoint + ty) * wA + AleftColPoint + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

       if(BleftRowPoint + ty < wB && BleftColPoint + tx < wB) {
            Bs[ty][tx] = B[(BleftRowPoint + ty) * wB + BleftColPoint + tx];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for(int k = 0; k < TILE_WIDTH; k++) {
            cval = fma(As[ty][k], Bs[k][tx], cval);
        }

        __syncthreads();
    }

    if(Crow < wA && Ccol < wB) {
       C[Crow * wB + Ccol] = cval;
    }
}



// // each thread compute 16 elements
// __global__ void MatrixMulSharedMemKernel_v2(float *A,
//     float *B, float *C, int wA,
//     int wB) {
    
//     int bx = blockIdx.x;
//     int by = blockIdx.y;
//     int tx = threadIdx.x;
//     int ty = threadIdx.y;
    
//     int Crow = bx * blockDim.x + tx;
//     int Ccol = by * blockDim.y + ty;
//     // 写入(Crow, Ccol)

//     // 每次读取一个block的A和B
//     // (Arow, Acol)
//     int AleftRowPoint = bx * TILE_WIDTH;
//     int AleftColPoint = 0;

//     int BleftRowPoint = 0;
//     int BleftColPoint = by * TILE_WIDTH;

//     float cval[4] = {0.0f, 0.0f, 0.0f, 0.0f};

//     for(int tile = 0; tile < TILE_WIDTH; tile++) {
//       __shared__ float As[4][TILE_WIDTH][TILE_WIDTH];
//       __shared__ float Bs[4][TILE_WIDTH][TILE_WIDTH];

//       int t_idx_x = threadIdx.x * 4 + tile;
//       int t_idx_y = threadIdx.y * 4 + tile;

//       As[t_idx_x][t_idx_y] = A[AleftRowPoint * wA + t_idx_x * wA + t_idx_y];
//       Bs[0][t_idx_x][t_idx_y] = B[BleftRowPoint * wB + t_idx_x * wB + t_idx_y];
//       Bs[1][t_idx_x][t_idx_y] = B[BleftRowPoint * wB + t_idx_x * wB + t_idx_y + 1];
//       Bs[2][t_idx_x][t_idx_y] = B[BleftRowPoint * wB + t_idx_x * wB + t_idx_y + 2];
//       Bs[3][t_idx_x][t_idx_y] = B[BleftRowPoint * wB + t_idx_x * wB + t_idx_y + 3];

//       __syncthreads();

//       #pragma unroll
//       for(int k = 0; k < TILE_WIDTH; k++) {
//         cval[0] += As[t_idx_x][k] * Bs[0][k][t_idx_y];
//         cval[1] += As[t_idx_x][k] * Bs[1][k][t_idx_y];
//         cval[2] += As[t_idx_x][k] * Bs[2][k][t_idx_y];
//         cval[3] += As[t_idx_x][k] * Bs[3][k][t_idx_y];
//       }
//     }

//     if(Crow < wA && Ccol < wB) {
//         C[Crow * wB + Ccol] = cval[0];
//         C[Crow * wB + Ccol + 1] = cval[1];
//         C[Crow * wB + Ccol + 2] = cval[2];
//         C[Crow * wB + Ccol + 3] = cval[3];
//     }
// }

//! For square matrices only
__global__ void MatrixMulKernel(float* d_M, float* d_N, float* d_P, int width)
{
  // 计算行索引
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  
  // 计算列索引
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  
  // 确保线程在矩阵范围内
  if ((row < width) && (col < width)) {
    float pValue = 0.0;
    
    // 计算矩阵乘法的一个元素
    for (int k = 0; k < width; k++) {
      pValue += d_M[row * width + k] * d_N[k * width + col];
    }
    
    // 将计算结果存储到输出矩阵中
    d_P[row * width + col] = pValue;
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
  
  // 声明存放在GPU上的数组
  float *h_M, *h_N, *d_M, *d_N;
  float *h_P, *d_P;
  
  size_t sizeM = m * k * sizeof(float);
  size_t sizeN = k * n * sizeof(float);
  size_t sizeP = m * n * sizeof(float);


  // Launch kernel 定义grid&block
  dim3 grid((int)ceil(k*1.0 / TILE_WIDTH), (int)ceil(m*1.0/ TILE_WIDTH));
  dim3 block(TILE_WIDTH, TILE_WIDTH);
  
  printf("Grid: (%d, %d)\n", grid.x, grid.y);
  printf("Block: (%d, %d)\n", block.x, block.y);

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

  // Copy data from CPU to GPU
  cudaMemcpy(d_M, h_M, sizeM, cudaMemcpyHostToDevice);
  cudaMemcpy(d_N, h_N, sizeN, cudaMemcpyHostToDevice);

  #define MatrixMulSharedMemKernel MatrixMulSharedMemKernel_v1

  // 添加 warmup
  {
    nvtxRangePushA("Warmup Start");
    printf("Warming up...\n");
    for (int i = 0; i < 3; i++) {
        MatrixMulSharedMemKernel<<<grid, block>>>(d_M, d_N, d_P, m, n);
    }
    cudaDeviceSynchronize();
    nvtxRangePop();
  }

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
        // MatrixMulKernel<<<grid, block>>>(d_M, d_N, d_P, m);
        MatrixMulSharedMemKernel<<<grid, block>>>(d_M, d_N, d_P, m, n);
        // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_N, n, d_M, k, &beta, d_P, n);
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

  return 0;
}

