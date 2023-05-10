
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

int dim = 16;
int NODE_NUM = 0;
int NODE_ACT_NUM = 0;
int FEATURE_TOTAL = 0;

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

std::vector<int *> build_part(
    int partSize, 
    int *indptr,
    int num_nodes
  ) {

   
    int degree, thisNumParts, numParts = 0;
	for(int i = 0; i < num_nodes; i++)
	{
        degree = indptr[i + 1] - indptr[i];
        if(degree % partSize == 0)
		    thisNumParts = degree / partSize;
        else
			thisNumParts = degree / partSize + 1;
        numParts += thisNumParts;
	}
    int *partPtr = new int[numParts + 1];
    int *part2Node = new int[numParts];
    int *degrees = new int[num_nodes];

    int part_counter = 0;
	for(int i = 0; i < num_nodes; i++)
	{
        int degree = indptr[i + 1] - indptr[i];
        degrees[i] = degree;

        if(degree % partSize == 0)
			thisNumParts = degree / partSize ;
        else
			thisNumParts = degree / partSize + 1;

        for (int pid = 0; pid < thisNumParts; pid++){
            int partBeg = indptr[i] + pid * partSize;
            int partEnd = partBeg + partSize < indptr[i  + 1]? partBeg + partSize: indptr[i + 1];
            partPtr[part_counter] = partBeg;
            part2Node[part_counter++] = i;
            if (i == num_nodes - 1 &&  partEnd == indptr[i + 1])
                partPtr[part_counter] = partEnd;
        }
	}
    return {partPtr, part2Node, degrees};
}

int calculate_part_num(
    int partSize, 
    int *indptr,
    int num_nodes
  ) {
    int degree, thisNumParts, numParts = 0;
	for(int i = 0; i < num_nodes; i++)
	{
        degree = indptr[i + 1] - indptr[i];
        if(degree % partSize == 0)
		    thisNumParts = degree / partSize;
        else
			thisNumParts = degree / partSize + 1;
        numParts += thisNumParts;
	}
    
    return numParts;
  }


__global__ void spmm_forward_cuda_kernel_64(
    float *output,
    float *input, 
    int *row_pointers, 
    int *column_index, 
    int *degrees, 
    int *part_pointers, 
    int *part2Node, 
    const int num_nodes, 
    const int dim, 
    const int num_parts,
    const int partSize, 
    const int dimWorker,
    const int warpPerBlock 
) 
{ 
    int tid =  blockIdx.x * blockDim.x + threadIdx.x;  
    int warpId = tid / WARP_SIZE;                            
    int block_warpId = threadIdx.x / WARP_SIZE;               
    int laneid = threadIdx.x % WARP_SIZE;                    
    
    extern __shared__ int part_meta[];                                      
    int *partial_ids = part_meta;                                           
    float *partial_results = (float*)&part_meta[partSize*warpPerBlock];   

    if (warpId < num_parts){
          
        int srcId = part2Node[warpId];             
        int partBeg = part_pointers[warpId];      
        int partEnd = part_pointers[warpId + 1];    
        float src_norm = degrees[srcId];           

        // Cache the part neighbors by all threads from a warp.
        const int pindex_base = block_warpId * partSize;
        
        #pragma unroll
        for (int nidx = partBeg + laneid; nidx < partEnd; nidx += WARP_SIZE){
            partial_ids[pindex_base + nidx - partBeg] = column_index[nidx];
        }
        
        __syncwarp();

        const int presult_base = block_warpId * dim;
        for (int nIdx = 0; nIdx < partEnd - partBeg; nIdx++)
        {
            int nid = partial_ids[pindex_base + nIdx]; 
            float degree_norm_inv = __fmaf_rn(src_norm, degrees[nid], 0);

            if (nIdx == 0)
                if (laneid < dimWorker)
                #pragma unroll
                for (int d = laneid; d < dim; d += dimWorker){
                    partial_results[presult_base + d] = 0.0f;
                }
            
            if (laneid < dimWorker)
            #pragma unroll
            for (int d = laneid; d < dim; d += dimWorker){
                partial_results[presult_base + d] += __fmaf_rn(degree_norm_inv, input[nid * dim + d], 0);
             
            }
        }

        if (laneid < dimWorker)
        #pragma unroll
        for (int d = laneid; d < dim; d += dimWorker){
            atomicAdd_F((float*)&output[srcId * dim + d], partial_results[presult_base + d]);
        }
    }

}

__global__ void spmm_forward_cuda_kernel(
    float *output,
    float *input, 
    int *row_pointers, 
    int *column_index, 
    int *degrees, 
    int *part_pointers, 
    int *part2Node, 
    const int num_nodes, 
    const int dim, 
    const int num_parts,
    const int partSize, 
    const int dimWorker,
    const int warpPerBlock,
    const int updated_dim,
    const int ngs_to_process
) 
{
    int tid =  blockIdx.x * blockDim.x + threadIdx.x;        
    int warpId = tid / WARP_SIZE;                            
    int block_warpId = threadIdx.x / WARP_SIZE;              
    int laneid = threadIdx.x % WARP_SIZE;                    
    extern __shared__ int part_meta[];                                     
    int *partial_ids = part_meta;                                        
    if (warpId < num_parts && laneid < updated_dim) 
    { 

        float partial_results = 0.0f;
    
        int ng_id = laneid / dim;
        int laneid_8 = laneid % dim;
       
        int local_ng_id = ngs_to_process * warpId;

        int srcId = part2Node[local_ng_id + ng_id]; 
        int partBeg = part_pointers[local_ng_id + ng_id];        
        int partEnd = part_pointers[local_ng_id + ng_id + 1];    
        float src_norm = degrees[srcId];            
        
        const int pindex_base    = (ngs_to_process * block_warpId + ng_id) * partSize;
  
        #pragma unroll
        for (int nidx = partBeg + laneid_8; nidx < partEnd; nidx += WARP_SIZE)
        {
            partial_ids[pindex_base + nidx - partBeg] = column_index[nidx];
        }
        
        __syncwarp(); 
        
        for (int nIdx = 0; nIdx < partEnd - partBeg; nIdx++) {
            int nid = partial_ids[pindex_base + nIdx];
            float degree_norm_inv = __fmaf_rn(src_norm, degrees[nid], 0); 
            partial_results += __fmaf_rn(degree_norm_inv, input[nid * dim + laneid_8], 0);     
        }
        
        atomicAdd_F((float*)&output[srcId * dim + laneid_8], partial_results);
        
    }
    
}
int main(int argc, char *argv[]) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); 

    if (argc < 3) {
        cout << "Please enter part num and dim as well" << endl;
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

    int *feature_indices   = (int *) malloc(NODE_ACT_NUM * sizeof(int));
    int *feature_indices_2 = (int *) malloc(FEATURE_TOTAL * sizeof(int));
    
    feature_indices_file.seekg(ios_base::beg);
    {
        getline(feature_indices_file, line);
        int i = 0;
        stringstream lineStream(line);
        string cell;
    
        while(std::getline(lineStream,cell, ','))
        {
            feature_indices[i] = stoi(cell);
            cout << cell << endl;
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
            cout << cell << endl;
            i++;
        }
        i = 0;
    }

    int part_size = atoi(argv[2]);
    int num_parts  = calculate_part_num(part_size, feature_indices, NODE_NUM);
    auto part_info = build_part(part_size, feature_indices, NODE_NUM);
    int num_nodes = NODE_NUM;


    /* Weight Matrix */
    float *h_input  = (float *) malloc(num_nodes * dim * sizeof(float));
    float *h_output = (float *) malloc(num_nodes * dim * sizeof(float)); 
    int *h_degrees  = (int *) malloc(num_nodes * sizeof(int));
   
    for (int i = 0; i < num_nodes * dim; i++) {
        h_input[i] = 1.0f;
    }
    for (int i = 0; i < num_nodes; i++) {
        h_degrees[i] = feature_indices[i + 1] - feature_indices[i];
    }
    
 
    /* Device allocation */
    float *d_input, *d_output;
    int *d_row_pointer, *d_col_index, *d_part_ptr, *d_part_to_node, *d_degrees;
   
    cudaMalloc((void**) &d_input, NODE_ACT_NUM * dim * sizeof(float));
    cudaMemcpy(d_input, h_input, NODE_ACT_NUM * dim * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**) &d_row_pointer, (NODE_ACT_NUM) * sizeof(int));
    cudaMemcpy(d_row_pointer, feature_indices, (NODE_ACT_NUM) * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**) &d_col_index, (FEATURE_TOTAL) * sizeof(int));
    cudaMemcpy(d_col_index, feature_indices_2, FEATURE_TOTAL * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**) &d_output, NODE_ACT_NUM * dim * sizeof(float));
    cudaMemset(&d_output, 0, NODE_ACT_NUM * dim * sizeof(float)); 

    cudaMalloc((void**) &d_part_ptr, (num_parts) * sizeof(int));
    cudaMemcpy(d_part_ptr, part_info[0], (num_parts) * sizeof(int), cudaMemcpyHostToDevice); 

    cudaMalloc((void**) &d_part_to_node, (num_parts) * sizeof(int));
    cudaMemcpy(d_part_to_node, part_info[1], (num_parts) * sizeof(int), cudaMemcpyHostToDevice); 

    cudaMalloc((void**) &d_degrees, (NODE_ACT_NUM) * sizeof(int));
    cudaMemcpy(d_degrees, h_degrees, NODE_ACT_NUM * sizeof(int), cudaMemcpyHostToDevice);
   
    /* Kernel Params */
    const int warpPerBlock = 8;
    const int block = warpPerBlock * WARP_SIZE; 
  
    int repeats = 200;
    if (dim > 32) {
        const int grid = (num_parts * WARP_SIZE + block  - 1) / (block);      
        int shared_memory = part_size*warpPerBlock*sizeof(int)+warpPerBlock*dim*sizeof(float);
  
        for (int k = 0; k < repeats; k++) {
            spmm_forward_cuda_kernel_64<<<grid, block, shared_memory>>>(
                (float *) d_output, (float *) d_input, 
                (int *) d_row_pointer, (int *) d_col_index, (int *) d_degrees, 
                (int *) d_part_ptr, (int *) d_part_to_node,
                num_nodes, dim, num_parts, part_size, 32, 8);
            cudaDeviceSynchronize();
        }
    }
    else {
        const int ngs_to_process = 32 / dim;
        const int updated_dim = ngs_to_process * dim;
        int shared_memory = ngs_to_process*part_size*warpPerBlock*sizeof(int);
        const int grid = (num_parts * WARP_SIZE + block  - 1) / (block * ngs_to_process); 
   
        for (int k = 0; k < repeats; k++) {
            spmm_forward_cuda_kernel<<<grid, block, shared_memory>>>(
                (float *) d_output, (float *) d_input, 
                (int *) d_row_pointer, (int *) d_col_index, (int *) d_degrees, 
                (int *) d_part_ptr, (int *) d_part_to_node,
                num_nodes, dim, num_parts, part_size, 32, 8, updated_dim, ngs_to_process);
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
         //printf("%f,", h_output[i * dim + j]);
        }
        
    //std::cout << endl;
    }
    return 0;
   
}
