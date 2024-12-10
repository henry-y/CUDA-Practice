# module load cuda/12.2
# nvcc -arch=compute_80 ./matrix_mul_opt.cu -o build/matrix_mul_opt
# nsys profile ./build/matrix_mul_opt 1 1024

module load cuda/12.2
nvcc -arch=compute_80 ./matrix_mul_opt.cu -o build/matrix_mul_opt
nsys profile ./build/matrix_mul_opt 1 1024