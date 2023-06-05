#include "kernels.h"

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