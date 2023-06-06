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

using namespace std;

/* Total number of nodes */
int NODE_NUM = 0;
/* Total number of nodes in CSR */
int NODE_ACT_NUM = 0;
/* Total number of non-zeros */
int FEATURE_TOTAL = 0;

/* Row and column ptr */
int *row_ptr;
int *col_ptr;

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

int main(int argc, char *argv[]) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); 

    if (argc < 5) {
        cout << "Usage: ./application <input_file> <dimension> <num_warps> <repeat>" << endl;
        exit(-1);
    }

    ifstream matrix_file(argv[1]);
    int dimension = atoi(argv[2]);
    int num_warps = atoi(argv[3]);
    int repeats = atoi(argv[4]);


    /* Read the input file */
    string line;
    string cell;

    /* Count the toal number of nodes and non-zeros */
    getline(matrix_file, line); 
    stringstream lineStream(line);
    while(std::getline(lineStream,cell, ',')) {
        NODE_ACT_NUM++;
        FEATURE_TOTAL = stoi(cell);
    }
    NODE_NUM = NODE_ACT_NUM - 1;
    
    cout << "Total number of rows: " << NODE_NUM << " and non-zeros: " << FEATURE_TOTAL << endl;

    row_ptr   = (int *) malloc(NODE_ACT_NUM * sizeof(int));
    col_ptr = (int *) malloc(FEATURE_TOTAL * sizeof(int));
    
    /* Populate row and col ptrs*/
    matrix_file.seekg(ios_base::beg);
    {
        getline(matrix_file, line);
        int i = 0;
        stringstream lineStream(line);
        string cell;
    
        while(std::getline(lineStream,cell, ',')) {
            row_ptr[i] = stoi(cell);
            //cout << cell << endl;
            i++;
        }
        i = 0;
    }
    {
        getline(matrix_file, line);
        int i = 0;
        stringstream lineStream(line);
        string cell;
    
        while(std::getline(lineStream,cell, ',')) {
            col_ptr[i] = stoi(cell);
            //cout << cell << endl;
            i++;
        }
        i = 0;
    }
    /* This part of code remains the same for any kernel */
    /* Host side memory allocations */
    float *h_input    = (float *) malloc(NODE_ACT_NUM * dimension * sizeof(float));
    float *h_output   = (float *) malloc(NODE_ACT_NUM * dimension * sizeof(float)); 
    int   *h_degrees  = (int *) malloc(NODE_ACT_NUM * sizeof(int));
    
    /* Filling the input with dummy data */
    for (int i = 0; i < NODE_ACT_NUM * dimension; i++) {
        h_input[i]  = 1.0f;
        h_output[i] = 0.0f;
    }
    /* Calculating degree of each node */
    for (int i = 0; i < NODE_NUM; i++) {
        h_degrees[i] = row_ptr[i + 1] - row_ptr[i];
    }
   
    /* Device allocation */
    float *d_input, *d_output;
    int *d_row_ptr, *d_col_ptr, *d_degrees;
    
    cudaMalloc((void**) &d_input, NODE_ACT_NUM * dimension * sizeof(float));
    cudaMemcpy(d_input, h_input, NODE_ACT_NUM * dimension * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaMalloc((void**) &d_row_ptr, (NODE_ACT_NUM) * sizeof(int));
    cudaMemcpy(d_row_ptr, row_ptr, (NODE_ACT_NUM) * sizeof(int), cudaMemcpyHostToDevice);
   
    cudaMalloc((void**) &d_col_ptr, (FEATURE_TOTAL) * sizeof(int));
    cudaMemcpy(d_col_ptr, col_ptr, FEATURE_TOTAL * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**) &d_output, NODE_ACT_NUM * dimension * sizeof(float));
    cudaMemset(&d_output, 0, NODE_ACT_NUM * dimension * sizeof(float)); 
    
    cudaMalloc((void**) &d_degrees, (NODE_ACT_NUM) * sizeof(int));
    cudaMemcpy(d_degrees, h_degrees, NODE_ACT_NUM * sizeof(int), cudaMemcpyHostToDevice);


    run_spmm_row_wise(d_output, d_input, d_row_ptr, d_col_ptr, d_degrees, NODE_NUM, 
    dimension, WARP_SIZE, num_warps, repeats);

    cudaMemcpy(h_output, d_output, NODE_ACT_NUM * dimension * sizeof(float), cudaMemcpyDeviceToHost);
    
    /* Verify the output */
    for (int i = 0; i < NODE_NUM; i++) {
        for (int j = 0; j < dimension; j++) {
            cout <<  (float)h_output[i * dimension + j] << "-";
        }
        std::cout << endl;
    }
    return 0;  
}

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
    int repeats
) {
    int grid = num_warps;
    for (int i = 0; i < repeats; i++) {
            spmm_row_wise<<<grid, BLOCK>>>(
                (float *) output, (float *) input, 
                (int *) row_ptr, (int *) col_ptr, (int *) degrees, 
                num_nodes, dimension, dimWorker, grid);
            cudaDeviceSynchronize();
    }
}

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
) {
    int tid =  blockIdx.x * blockDim.x + threadIdx.x;  // global thread-id
    int warp_id = tid / WARP_SIZE;                      // global warp-id
    int lane_id = threadIdx.x % WARP_SIZE;              // warp thread-id -- laneid

    if (warp_id < num_warps) {
        /* Get the bounds */
        float step = (float)num_nodes / num_warps;
        int row_start = step * warp_id;
        int row_end = step * (warp_id + 1);
        
        int num_features = 0;
        int features_start = 0;
        float src_norm = 0;
        float degree_norm_inv = 0;
        int index = 0;
        float output_temp = 0;
        
        if (lane_id < dimension) {
            for (int i = row_start; i < row_end; i++) {
                output_temp = 0;
                num_features = row_ptr[i + 1] - row_ptr[i];
                features_start = row_ptr[i]; 
                src_norm = degrees[i];  
        
                #pragma unroll
                for (int j = 0; j < num_features; j++) {
                    index = col_ptr[features_start];
                    degree_norm_inv = __fmaf_rn(src_norm, degrees[index], 0);
                    output_temp += __fmaf_rn(degree_norm_inv, input[index * dimension + lane_id], 0);
                    features_start++;
                }
                output[i * dimension + lane_id] = output_temp;
            }    
        }
    }
}