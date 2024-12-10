#!/bin/bash

module load cuda/12.2
ncu -f -o profile_opt --set full ./build/matrix_mul_opt 1 1024
