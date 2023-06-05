#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <vector>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <math.h>
#include <sstream>
#include <stdint.h>
#include <string>
#include <vector>

const int WARP_SIZE = 32;
const int WARPS_PER_BLOCK = 8;
const int BLOCK = WARPS_PER_BLOCK * WARP_SIZE; 

__global__ void spmm_row_wise(
    float *output,
    float *input, 
    int *row_ptr, 
    int *col_ptr, 
    int *degrees, 
    int num_nodes, 
    int dimension,
    int dimWorker,
    int num_warps
);

void run_spmm_row_wise(
    float *output,
    float *input, 
    int *row_ptr, 
    int *col_ptr, 
    int *degrees, 
    int num_nodes, 
    int dimension,
    int dimWorker,
    int num_warps,
    int reapeats
);

