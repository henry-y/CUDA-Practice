nvcc -arch=compute_80 -lineinfo -std=c++20 ./matrix_mul_opt.cu -o build/matrix_mul_opt -lcudart -I/usr/local/cuda-12.2/include
./build/matrix_mul_opt 1 1024