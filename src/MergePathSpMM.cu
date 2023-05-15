
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

#define ROW_PTR 0
#define COL_IDX 1
#define WARP_SIZE 32

int dim = 4;
int NODE_NUM = 0;
int NODE_ACT_NUM = 0;
int FEATURE_TOTAL = 0;
int *feature_indices;
int *feature_indices_2;

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
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
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
                                                                
        CoordinateT thread_coord = MergePathSearch(diagonal, feature_indices, NZ_INDICES, NODE_ACT_NUM, FEATURE_TOTAL);
        CoordinateT thread_coord_end = MergePathSearch(diagonal_end, feature_indices, NZ_INDICES, NODE_ACT_NUM, FEATURE_TOTAL);
    
        int start = thread_coord.x - 1;
        int end = thread_coord_end.x - 1;
        if (start < 0) start = 0;

        int num_features = 0;

        int feature_start = thread_coord.y;
        if (feature_indices[start] == feature_start) {
            feature_start = 0;
        }
        if (core_id == 0) {
            feature_start = 0;
        }

        int feature_end = thread_coord_end.y;
        if (feature_indices[end] == feature_end) {
            feature_end = 0;
        }

        if (feature_start != 0) {
            if (start == end && feature_end != 0) {
                num_features = feature_end - feature_start;
                feature_end = 0;
            }
            else {
                num_features = feature_indices[start + 1] - feature_start;
            }
            
        }
        int num_features_end = 0;
        if (feature_end != 0) num_features_end = feature_end - feature_indices[end];

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


__global__ void spmm_forward_cuda_kernel_mp(
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
) {

    int tid =  blockIdx.x * blockDim.x + threadIdx.x;  // global thread-id
    int warpId = tid / WARP_SIZE;                             // global warp-id
    int laneid = threadIdx.x % WARP_SIZE;                     // warp thread-id -- laneid

    if (warpId < num_warps) {
        int sched_id = laneid / dim;
        int lane_id_8 = laneid % dim;
        if (sched_id >= sched_to_process) { 
            return;
        }
        
        int start = start_row[sched_to_process * warpId + sched_id];
        int end = end_row[sched_to_process * warpId + sched_id];
        int fstart = feature_start[sched_to_process * warpId + sched_id];
        int fstart_num = feature_start_num[sched_to_process * warpId + sched_id];
        int fend = feature_end[sched_to_process * warpId + sched_id];
        int fend_num = feature_end_num[sched_to_process * warpId + sched_id];

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
                degree_norm_inv = __fmaf_rn(src_norm, degrees[index], 0);
                partial_results_start += __fmaf_rn(degree_norm_inv, input[index * dim + lane_id_8], 0); 
                            
            }
                       
            atomicAdd_F((float*) &output[start * dim + lane_id_8], partial_results_start);
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
                degree_norm_inv = __fmaf_rn(src_norm, degrees[index], 0);
                output_temp += __fmaf_rn(degree_norm_inv, input[index * dim + lane_id_8], 0);
                features_start++;
            }

            output[i * dim + lane_id_8] = output_temp;
        }             

        if (fend != 0) {
            src_norm = 1;  
         
            #pragma unroll
            for (int j = 0; j < fend_num; j++) {
                index = column_index[fend++];
                degree_norm_inv = __fmaf_rn(src_norm, degrees[index], 0);
                partial_results_end += __fmaf_rn(degree_norm_inv, input[index * dim + lane_id_8], 0); 
            } 
        
            atomicAdd_F((float*) &output[end * dim + lane_id_8], partial_results_end);
        }
        return;
    }
}


__global__ void spmm_forward_cuda_kernel_mp_64(
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
    int num_warps_2
) {

    int tid =  blockIdx.x * blockDim.x + threadIdx.x;  // global thread-id
    int warpId = tid / WARP_SIZE;                             // global warp-id
    int laneid = threadIdx.x % WARP_SIZE;                     // warp thread-id -- laneid

    if (warpId < num_warps_2) {
        if (warpId % 2 == 1) laneid +=  32;
        if (laneid > dim)  return;
        warpId = warpId / 2;
       
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
                degree_norm_inv = __fmaf_rn(src_norm, degrees[index], 0);
                partial_results_start += __fmaf_rn(degree_norm_inv, input[index * dim + laneid], 0); 
                            
            }
                      
            atomicAdd_F((float*) &output[start * dim + laneid], partial_results_start);
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
                degree_norm_inv = __fmaf_rn(src_norm, degrees[index], 0);
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
                degree_norm_inv = __fmaf_rn(src_norm, degrees[index], 0);
                partial_results_end += __fmaf_rn(degree_norm_inv, input[index * dim + laneid], 0); 
            } 
        
            atomicAdd_F((float*) &output[end * dim + laneid], partial_results_end);
        }
        return;
    }
}

int main(int argc, char *argv[]) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); 

    if (argc < 2) {
        cout << "Please enter cost as well" << endl;
        exit(-1);
    }
    ifstream feature_indices_file(argv[1]);

    string line;
    string cell;

    getline(feature_indices_file, line); 
    stringstream lineStream(line);
    while(std::getline(lineStream,cell, ',')) {
        NODE_ACT_NUM++;
        FEATURE_TOTAL = stoi(cell);
    }
    NODE_NUM = NODE_ACT_NUM - 1;
    cout << FEATURE_TOTAL << endl;

    feature_indices   = (int *) malloc(NODE_ACT_NUM * sizeof(int));
    feature_indices_2 = (int *) malloc(FEATURE_TOTAL * sizeof(int));
    
    feature_indices_file.seekg(ios_base::beg);
    {
        getline(feature_indices_file, line);
        int i = 0;
        stringstream lineStream(line);
        string cell;
    
        while(std::getline(lineStream,cell, ','))
        {
            feature_indices[i] = stoi(cell);
            //cout << cell << endl;
            i++;
        }
        i = 0;
    }
    {
        getline(feature_indices_file, line);
        int i = 0;
        stringstream lineStream(line);
        string cell;
    
        while(std::getline(lineStream,cell, ','))
        {
            feature_indices_2[i] = stoi(cell);
            //cout << cell << endl;
            i++;
        }
        i = 0;
    }


    int cost = (atoi(argv[2]));
    dim = (atoi(argv[3]));
    int threads_per_warp = (atoi(argv[4]));


    int num_threads = (NODE_ACT_NUM + FEATURE_TOTAL - 1) / cost;
    if (num_threads < 1024) num_threads = 1024;
    int num_nodes = NODE_NUM;

    /* Weight Matrix */
    float *h_input  = (float *) malloc(NODE_ACT_NUM * dim * sizeof(float));
    float *h_output = (float *) malloc(NODE_ACT_NUM * dim * sizeof(float)); 
    int *h_degrees  = (int *) malloc(NODE_ACT_NUM * sizeof(int));
   
    for (int i = 0; i < NODE_ACT_NUM * dim; i++) {
        h_input[i] = 1.0f;
    }
    for (int i = 0; i < NODE_NUM; i++) {
        h_degrees[i] = feature_indices[i + 1] - feature_indices[i];
    }
   
    /* Device allocation */
    float *d_input, *d_output;
    int *d_row_pointer, *d_col_index, *d_degrees;
    int *d_feature_start, *d_feature_start_num, *d_feature_end, *d_feature_end_num;
    int *d_row_start, *d_row_end;
    auto mp_sched = generate_mp_sched(num_threads);

    cudaMalloc((void**) &d_input, NODE_ACT_NUM * dim * sizeof(float));
    cudaMemcpy(d_input, h_input, NODE_ACT_NUM * dim * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaMalloc((void**) &d_row_pointer, (NODE_ACT_NUM) * sizeof(int));
    cudaMemcpy(d_row_pointer, feature_indices, (NODE_ACT_NUM) * sizeof(int), cudaMemcpyHostToDevice);
   
    cudaMalloc((void**) &d_col_index, (FEATURE_TOTAL) * sizeof(int));
    cudaMemcpy(d_col_index, feature_indices_2, FEATURE_TOTAL * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**) &d_output, NODE_ACT_NUM * dim * sizeof(float));
    cudaMemset(&d_output, 0, NODE_ACT_NUM * dim * sizeof(float)); 
    
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

    /* Kernel Params */
    const int warpPerBlock = 8;
    const int block = warpPerBlock * WARP_SIZE; 

    int repeats = 200;
    if (dim <= 32) {
        int sched_to_process = threads_per_warp;
        if (threads_per_warp > 32 / dim) {
            cout << "Cannot process the given number of threads withing a warp." << endl;
            cout << "Max threads per warp can be " << 32 / dim << endl;
            exit(-1);
        }
        const int grid = num_threads / sched_to_process; 
        for (int i = 0; i < repeats; i++) {
            spmm_forward_cuda_kernel_mp<<<grid, block>>>(
                (float *) d_output, (float *) d_input, 
                (int *) d_row_pointer, (int *) d_col_index, (int *) d_degrees, 
                (int *) d_feature_start,
                (int *) d_feature_end,
                (int *) d_feature_start_num,
                (int *) d_feature_end_num,
                (int *) d_row_start,
                (int *) d_row_end,
                num_nodes, dim, 32, 8, grid, sched_to_process);
            cudaDeviceSynchronize();
        }
    }
    else {
        
        const int grid = num_threads; 
        
        for (int i = 0; i < repeats; i++) {
            spmm_forward_cuda_kernel_mp_64<<<grid * 2, block>>>(
                (float *) d_output, (float *) d_input, 
                (int *) d_row_pointer, (int *) d_col_index, (int *) d_degrees, 
                (int *) d_feature_start,
                (int *) d_feature_end,
                (int *) d_feature_start_num,
                (int *) d_feature_end_num,
                (int *) d_row_start,
                (int *) d_row_end,
                num_nodes, dim, 32, 8, grid, grid * 2);
        cudaDeviceSynchronize();
    }
    }

    
    cudaMemcpy(h_output, d_output, NODE_ACT_NUM * dim * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < NODE_NUM; i++) {
        for (int j = 0; j < dim; j++) {
        //     // if (feature_indices[i+1] - feature_indices[i] != (int)h_output[i * dim]) {
        //     //     cout << feature_indices[i+1] - feature_indices[i] <<"," << (int)h_output[i * dim]  << endl;
        //     // }
        //     printf("%d, %d",, (int)h_output[i * dim]);
            //cout <<  feature_indices[i+1] - feature_indices[i] << "," << (int)h_output[i * dim + j] << "-";
        }
        
        //std::cout << endl;
    }
    return 0;
   
}

