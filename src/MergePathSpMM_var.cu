#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>

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

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    }
}

__device__ inline 
void atomicAdd_F(float* address, float value)
{
  float old = value;  
  while ((old = atomicExch(address, atomicExch(address, 0.0f)+old))!=0.0f);
}


struct CoordinateT {
    int x;
    int y;
};
CoordinateT MergePathSearch( int diagonal, volatile int* RP, int* NZ_INDICES, int num_rows, int nnz);
std::vector<int *> generate_mp_sched(int num_threads);

__global__ void spmm_merge_path(
    float *output,
    float *input, 
    int *row_pointers, 
    int *column_index, 
    int *degrees, 
    int *feature_start,
    int *feature_end,
    int *feature_start_num,
    int *feature_end_num,    
    int *start_row,
    int *end_row,
    int num_nodes, 
    int dim,
    int dimWorker,
    int warpPerBlock,
    int num_warps,
    int sched_to_process
);

__global__ void spmm_merge_path_64(
    float *output,
    float *input, 
    int *row_pointers, 
    int *column_index, 
    int *degrees, 
    int *feature_start,
    int *feature_end,
    int *feature_start_num,
    int *feature_end_num,    
    int *start_row,
    int *end_row,
    int num_nodes, 
    int dim,
    int dimWorker,
    int warpPerBlock,
    int num_warps,
    int factor
);

    
int main(int argc, char *argv[]) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); 

    if (argc < 6) {
        cout << "Usage: ./application <input_file> <dimension> <cost> <threads_per_warp> <repeat>" << endl;
        exit(-1);
    }

    ifstream matrix_file(argv[1]);
    int dimension = atoi(argv[2]);
    int cost = atoi(argv[3]);
    int threads_per_warp = (atoi(argv[4]));
    int repeats = atoi(argv[5]);
    int num_threads = 0; 

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
    num_threads = (NODE_ACT_NUM + FEATURE_TOTAL - 1) / cost;
    if (num_threads < 1024) num_threads = 1024;

    cout << "Total number of rows: " << NODE_NUM << " and non-zeros: " << FEATURE_TOTAL << endl;
    
    row_ptr = (int *) malloc(NODE_ACT_NUM * sizeof(int));
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

    /* Weight Matrix */
    float *h_input  = (float *) malloc(NODE_ACT_NUM * dimension * sizeof(float));
    float *h_output = (float *) malloc(NODE_ACT_NUM * dimension * sizeof(float)); 
    int *h_degrees  = (int *) malloc(NODE_ACT_NUM * sizeof(int));
   
    for (int i = 0; i < NODE_ACT_NUM * dimension; i++) {
        h_input[i] = 1.0f;
    }
    for (int i = 0; i < NODE_NUM; i++) {
        h_degrees[i] = row_ptr[i + 1] - row_ptr[i];
    }
   
    /* Device allocation */
    float *d_input, *d_output;
    int *d_row_pointer, *d_col_index, *d_degrees;
    int *d_feature_start, *d_feature_start_num, *d_feature_end, *d_feature_end_num;
    int *d_row_start, *d_row_end;
    auto mp_sched = generate_mp_sched(num_threads);

    cudaMalloc((void**) &d_input, NODE_ACT_NUM * dimension * sizeof(float));
    cudaMemcpy(d_input, h_input, NODE_ACT_NUM * dimension * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaMalloc((void**) &d_row_pointer, (NODE_ACT_NUM) * sizeof(int));
    cudaMemcpy(d_row_pointer, row_ptr, (NODE_ACT_NUM) * sizeof(int), cudaMemcpyHostToDevice);
   
    cudaMalloc((void**) &d_col_index, (FEATURE_TOTAL) * sizeof(int));
    cudaMemcpy(d_col_index, col_ptr, FEATURE_TOTAL * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**) &d_output, NODE_ACT_NUM * dimension * sizeof(float));
    cudaMemset(&d_output, 0, NODE_ACT_NUM * dimension* sizeof(float)); 
    
    cudaMalloc((void**) &d_degrees, (NODE_ACT_NUM) * sizeof(int));
    cudaMemcpy(d_degrees, h_degrees, NODE_ACT_NUM * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**) &d_feature_start, num_threads * sizeof(int));
    cudaMemcpy(d_feature_start, mp_sched[0], num_threads * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**) &d_feature_start_num, num_threads * sizeof(int));
    cudaMemcpy(d_feature_start_num, mp_sched[1], num_threads * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**) &d_feature_end, num_threads * sizeof(int));
    cudaMemcpy(d_feature_end, mp_sched[2], num_threads * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**) &d_feature_end_num, num_threads * sizeof(int));
    cudaMemcpy(d_feature_end_num, mp_sched[3], num_threads * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**) &d_row_start, num_threads * sizeof(int));
    cudaMemcpy(d_row_start, mp_sched[4], num_threads * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**) &d_row_end, num_threads * sizeof(int));
    cudaMemcpy(d_row_end, mp_sched[5], num_threads * sizeof(int), cudaMemcpyHostToDevice);


    if (dimension <= WARP_SIZE) {
        if (threads_per_warp > WARP_SIZE / dimension) {
            cout << "Cannot process the given number of threads withing a warp." << endl;
            cout << "Max threads per warp can be " << WARP_SIZE / dimension << endl;
            exit(-1);
        }

        const int grid = (num_threads * WARP_SIZE + BLOCK - 1) / (BLOCK); 
        for (int i = 0; i < repeats; i++) {
            spmm_merge_path<<<grid, BLOCK>>>(
                (float *) d_output, (float *) d_input, 
                (int *) d_row_pointer, (int *) d_col_index, (int *) d_degrees, 
                (int *) d_feature_start,
                (int *) d_feature_end,
                (int *) d_feature_start_num,
                (int *) d_feature_end_num,
                (int *) d_row_start,
                (int *) d_row_end,
                NODE_NUM, dimension, WARP_SIZE, WARPS_PER_BLOCK, num_threads, 32 / dimension);
            cudaDeviceSynchronize();
        }
    }
    else {

        int factor = ceil(dimension / WARP_SIZE);
        int num_threads_gpu = num_threads * factor; 
        int grid = (num_threads_gpu * WARP_SIZE + BLOCK  - 1) / (BLOCK);
        
        for (int i = 0; i < repeats; i++) {
            spmm_merge_path_64<<<grid, BLOCK>>>(
                (float *) d_output, (float *) d_input, 
                (int *) d_row_pointer, (int *) d_col_index, (int *) d_degrees, 
                (int *) d_feature_start,
                (int *) d_feature_end,
                (int *) d_feature_start_num,
                (int *) d_feature_end_num,
                (int *) d_row_start,
                (int *) d_row_end,
                NODE_NUM, dimension, WARP_SIZE, WARPS_PER_BLOCK, num_threads_gpu, factor);
            cudaDeviceSynchronize();
        }
    }

    
    cudaMemcpy(h_output, d_output, NODE_ACT_NUM * dimension * sizeof(float), cudaMemcpyDeviceToHost);

    // for (int i = 0; i < NODE_NUM; i++) {
    //     for (int j = 0; j < dimension; j++) {
    //         cout <<  float(h_output[i * dimension + j]) << "-";
    //     }
    //     std::cout << endl;
    // }

    return 0;
}


CoordinateT MergePathSearch( int diagonal, volatile int* RP, int* NZ_INDICES, int num_rows, int nnz)
{
    
    int x_min = max(diagonal - nnz, 0);
    int x_max = min(diagonal, num_rows);

    while (x_min < x_max) {
        // so this is div by 2
        int pivot = (x_min + x_max) >> 1;
        if (RP[pivot] <= NZ_INDICES[diagonal - pivot - 1]) {
            x_min = pivot + 1;
        } 
        else {
            x_max = pivot;
        }
    }
    return CoordinateT{min(x_min, num_rows), diagonal - x_min};
}


std::vector<int *> generate_mp_sched(int num_threads) {

    int *feature_start_all = new int[num_threads];
    int *feature_end_all   = new int[num_threads];
    int *feature_start_num = new int[num_threads];
    int *feature_end_num   = new int[num_threads];
    
    int *start_row_all     = new int[num_threads];
    int *end_row_all       = new int[num_threads];
    int *NZ_INDICES        = new int[FEATURE_TOTAL];
    
    for (int i = 0; i < FEATURE_TOTAL; i++) {
        NZ_INDICES[i] = i;
    }

    for (int i = 0; i < num_threads; i++) {
        feature_start_all[i] = 0;
        feature_end_all[i]   = 0;
        feature_start_num[i] = 0;
        feature_end_num[i]   = 0;
        start_row_all[i]     = 0;
        end_row_all[i]       = 0;
    }

    for (int i = 0; i < num_threads; i++) {
        int core_id = i;

        int num_merge_items = NODE_ACT_NUM + FEATURE_TOTAL; 
        int items_per_thread = (num_merge_items + num_threads - 1) / num_threads;

        int diagonal = min(items_per_thread * core_id, num_merge_items);
        int diagonal_end = min(diagonal + items_per_thread, num_merge_items);
                                                                
        CoordinateT thread_coord = MergePathSearch(diagonal, row_ptr, NZ_INDICES, NODE_ACT_NUM, FEATURE_TOTAL);
        CoordinateT thread_coord_end = MergePathSearch(diagonal_end, row_ptr, NZ_INDICES, NODE_ACT_NUM, FEATURE_TOTAL);
    
        int start = thread_coord.x - 1;
        int end = thread_coord_end.x - 1;
        if (start < 0) start = 0;

        int num_features = 0;

        int feature_start = thread_coord.y;
        if (row_ptr[start] == feature_start) {
            feature_start = 0;
        }
        if (core_id == 0) {
            feature_start = 0;
        }

        int feature_end = thread_coord_end.y;
        if (row_ptr[end] == feature_end) {
            feature_end = 0;
        }

        if (feature_start != 0) {
            if (start == end && feature_end != 0) {
                num_features = feature_end - feature_start;
                feature_end = 0;
            }
            else {
                num_features = row_ptr[start + 1] - feature_start;
            }
            
        }
        int num_features_end = 0;
        if (feature_end != 0) num_features_end = feature_end - row_ptr[end];

        feature_start_all[core_id] = feature_start;
        feature_end_all[core_id]   = feature_end; 
        feature_start_num[core_id] = num_features;
        feature_end_num[core_id]   = num_features_end;   
        start_row_all[core_id]     = start;     
        end_row_all[core_id]       = end;       

    }

    return {feature_start_all, feature_start_num, 
            feature_end_all,
            feature_end_num,
            start_row_all, end_row_all};
    
}


__global__ void spmm_merge_path(
    float *output,
    float *input, 
    int *row_pointers, 
    int *column_index, 
    int *degrees, 
    int *feature_start,
    int *feature_end,
    int *feature_start_num,
    int *feature_end_num,    
    int *start_row,
    int *end_row,
    int num_nodes, 
    int dim,
    int dimWorker,
    int warpPerBlock,
    int num_warps,
    int nz_to_process
) {

    int tid =  blockIdx.x * blockDim.x + threadIdx.x;  // global thread-id
    int warpId = tid / WARP_SIZE;                             // global warp-id
    int laneid = threadIdx.x % WARP_SIZE;                     // warp thread-id -- laneid
    
    if (warpId < num_warps) {
        int nz_start = laneid / dim; // for all the rows start from this nz
        laneid = laneid % dim; // correct lane id
        
        int start = start_row[warpId];
        int end = end_row[warpId];
        int fstart = feature_start[warpId];
        int fstart_num = feature_start_num[warpId];
        int fend = feature_end[warpId];
        int fend_num = feature_end_num[warpId];

        float partial_results_start = 0;
        float  partial_results_end = 0;
        float output_temp = 0; 
        float degree_norm_inv = 0;
        float src_norm = 0;
        int index = 0;
        int num_features = 0;
        int features_start = 0;

        if (fstart != 0) {
            src_norm = degrees[start];  
            
            for (int j = nz_start; j < fstart_num; j += nz_to_process) {
                index = column_index[fstart]; fstart += nz_to_process;
                degree_norm_inv =  __fmaf_rn(src_norm, degrees[index], 0);
                partial_results_start += __fmaf_rn(degree_norm_inv, input[index * dim + laneid], 0); 
                            
            }
            atomicAdd_F(&output[start * dim + laneid], partial_results_start);         
            start = start + 1;

        }

        for (int i = start; i < end; i++) {
            src_norm = degrees[i];
            output_temp = 0.0f;

            num_features = row_pointers[i + 1] - row_pointers[i];
            features_start = row_pointers[i]; 
    
            #pragma unroll
            for (int j = nz_start; j < num_features; j += nz_to_process) {
                index = column_index[features_start];
                degree_norm_inv =  __fmaf_rn(src_norm, degrees[index], 0);
                output_temp += __fmaf_rn(degree_norm_inv, input[index * dim + laneid], 0);
                features_start += nz_to_process;
            }
            atomicAdd_F(&output[i * dim + laneid], output_temp); //everything is atomic 
       
        }             

        if (fend != 0) {
            src_norm = 1;  
         
            #pragma unroll
            for (int j = nz_start; j < fend_num; j += nz_to_process) {
                index = column_index[fend]; fend += nz_to_process;
                degree_norm_inv =  __fmaf_rn(src_norm, degrees[index], 0);
                partial_results_end += __fmaf_rn(degree_norm_inv, input[index * dim + laneid], 0); 
            } 
            atomicAdd_F(&output[end * dim + laneid], partial_results_end);
        }
        return;
    }
}


__global__ void spmm_merge_path_64(
    float *output,
    float *input, 
    int *row_pointers, 
    int *column_index, 
    int *degrees, 
    int *feature_start,
    int *feature_end,
    int *feature_start_num,
    int *feature_end_num,    
    int *start_row,
    int *end_row,
    int num_nodes, 
    int dim,
    int dimWorker,
    int warpPerBlock,
    int num_warps,
    int factor
) {

    int tid =  blockIdx.x * blockDim.x + threadIdx.x;  // global thread-id
    int warpId = tid / WARP_SIZE;                             // global warp-id
    int laneid = threadIdx.x % WARP_SIZE;                     // warp thread-id -- laneid
    
    if (warpId < num_warps) {
        
        laneid += (warpId % factor) * 32;
        if (laneid > dim)  return;
        warpId = warpId / factor;
       
        int start = start_row[warpId];
        int end = end_row[warpId];
        int fstart = feature_start[warpId];
        int fstart_num = feature_start_num[warpId];
        int fend = feature_end[warpId];
        int fend_num = feature_end_num[warpId];

        float partial_results_start = 0;
        float  partial_results_end = 0;
        float output_temp = 0; 
        float degree_norm_inv = 0;
        float src_norm = 0;
        int index = 0;
        int num_features = 0;
        int features_start = 0;

        if (fstart != 0) {
            src_norm = degrees[start];  
            
            for (int j = 0; j < fstart_num; j++) {
                index = column_index[fstart++];
                degree_norm_inv =  __fmaf_rn(src_norm, degrees[index], 0);
                partial_results_start +=  __fmaf_rn(degree_norm_inv, input[index * dim + laneid], 0); 
                            
            }
            atomicAdd(&output[start * dim + laneid], partial_results_start);         
            start = start + 1;

        }

        for (int i = start; i < end; i++) {
            src_norm = degrees[i];
            output_temp = 0.0f;

            num_features = row_pointers[i + 1] - row_pointers[i];
            features_start = row_pointers[i]; 
    
            #pragma unroll
            for (int j = 0; j < num_features; j++) {
                index = column_index[features_start];
                degree_norm_inv =  __fmaf_rn(src_norm, degrees[index], 0);
                output_temp += __fmaf_rn(degree_norm_inv, input[index * dim + laneid], 0);
                features_start++;
            }

            output[i * dim + laneid] = output_temp;
        }             

        if (fend != 0) {
            src_norm = 1;  
         
            #pragma unroll
            for (int j = 0; j < fend_num; j++) {
                index = column_index[fend++];
                degree_norm_inv =  __fmaf_rn(src_norm, degrees[index], 0);
                partial_results_end +=  __fmaf_rn(degree_norm_inv, input[index * dim + laneid], 0); 
            } 
            atomicAdd(&output[end * dim + laneid], partial_results_end);
        }
        return;
    }
}
