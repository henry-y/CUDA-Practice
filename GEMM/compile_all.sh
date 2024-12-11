mkdir -p build
module load cuda/12.2
nvcc -arch=compute_80 -lineinfo ./matrix_mul.cu -o build/matrix_mul -lcudart -I/usr/local/cuda-12.2/include
nvcc -arch=compute_80 -lineinfo ./matrix_mul_opt.cu -o build/matrix_mul_opt -lcudart -I/usr/local/cuda-12.2/include
nvcc -arch=compute_80 -lineinfo ./matrix_mul_cublas.cu -lcublas -o build/matrix_mul_cublas -lcudart -I/usr/local/cuda-12.2/include

echo "Running basic version:"
./build/matrix_mul 1 1024
echo -e "\nRunning optimized version:"
./build/matrix_mul_opt 1 1024
echo -e "\nRunning cuBLAS version:"
./build/matrix_mul_cublas 1 1024

echo "Running basic version:"
./build/matrix_mul 1 1000
echo -e "\nRunning optimized version:"
./build/matrix_mul_opt 1 1000
echo -e "\nRunning cuBLAS version:"
./build/matrix_mul_cublas 1 1000
