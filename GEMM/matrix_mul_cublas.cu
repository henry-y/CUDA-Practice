#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <nvtx3/nvToolsExt.h>
using namespace std;

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

void matrixMulCPU(float *C, const float *A, const float *B, unsigned int hA, unsigned int wA, unsigned int wB)
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
    int error_count=0;

    for (int j = 0; j < height; j++)
    {
        for (int i = 0; i < width; i++)
        {
            int k = j * width + i;
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

int main(int argc, char* argv[])
{
    int size, check;
    getArg(argc, argv, size, check);

    int m = size, n = size, k = size;
    
    float *h_A, *h_B, *d_A, *d_B;
    float *h_C, *d_C;
    
    size_t sizeA = m * k * sizeof(float);
    size_t sizeB = k * n * sizeof(float);
    size_t sizeC = m * n * sizeof(float);

    // 分配主机内存
    h_A = (float*) malloc(sizeA);
    h_B = (float*) malloc(sizeB);
    h_C = (float*) malloc(sizeC);
    float *reference = (float *)malloc(sizeC);

    // 分配设备内存
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    // 初始化数据
    for(int i = 0; i < m * k; ++i)
    {
        h_A[i] = (i % 2 == 0) ? 1.0f : 0.5f;
    }

    for(int i = 0; i < k * n; ++i)
    {
        h_B[i] = (i % 2 == 0) ? 0.5f : 1.0f;
    }

    // 数据拷贝到GPU
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    // 创建CUBLAS句柄
    cublasHandle_t handle;
    cublasCreate(&handle);

    // CUBLAS参数
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // 添加 warmup
    nvtxRangePushA("Warmup Start");
    printf("Warming up...\n");
    for (int i = 0; i < 3; i++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B, n, d_A, k, &beta, d_C, n);
    }
    cudaDeviceSynchronize();
    nvtxRangePop();

    // Timing records 
    nvtxRangePushA("Kernel Execution Start");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    int nIter = 5;

    // 执行矩阵乘法
    for (int j = 0; j < nIter; j++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B, n, d_A, k, &beta, d_C, n);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    nvtxRangePop();
    float msecPerMatrixMul;
    cudaEventElapsedTime(&msecPerMatrixMul, start, stop);
    msecPerMatrixMul /= nIter;
    printf("[cuBLAS] Kernel Elapsed Time: %.3f ms\n", msecPerMatrixMul);

    // 计算和打印性能指标
    double flopsPerMatrixMul = 2.0 * (double)m * (double)n * (double)k;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf("[cuBLAS] Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n",
           gigaFlops,
           msecPerMatrixMul,
           flopsPerMatrixMul);

    // 拷贝结果回主机
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // 验证结果
    if (check == 1)
    {
        printf("Computing result using host CPU...");
        matrixMulCPU(reference, h_A, h_B, m, k, n);
        printf("done.\n");
        printDiff(reference, h_C, n, m, 100, 1.0e-5f);
    }

    // 清理资源
    free(h_C);
    free(h_A);
    free(h_B);
    free(reference);
    cudaFree(d_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
} 