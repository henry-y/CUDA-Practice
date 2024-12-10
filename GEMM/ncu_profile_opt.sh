#!/bin/bash

module load cuda/12.2
nvcc -o build/matrix_mul_opt matrix_mul_opt.cu
ncu -f -o profile_opt --set full ./build/matrix_mul_opt 1 1024

