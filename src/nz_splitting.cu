
#include <cuda.h>
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

std::vector<int *> build_part(
    int partSize, 
    int *indptr,
    int num_nodes);

int calculate_part_num(
    int partSize, 
    int *indptr,
    int num_nodes);

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
    const int warpPerBlock);

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
    const int ngs_to_process); 


int main(int argc, char *argv[]) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); 

    if (argc < 5) {
        cout << "Usage: ./application <input_file> <dimension> <part_size> <repeat>" << endl;
        exit(-1);
    }

    ifstream matrix_file(argv[1]);
    int dimension = atoi(argv[2]);
    int part_size = atoi(argv[3]);
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
    
    /* Calcuate part size and part pointer */
    int num_parts  = calculate_part_num(part_size, row_ptr, NODE_NUM);
    auto part_info = build_part(part_size, row_ptr, NODE_NUM);


    /* Device allocation */
    float *d_input, *d_output;
    int *d_row_ptr, *d_col_ptr, *d_degrees;
    int  *d_part_ptr, *d_part_to_node;

    cudaMalloc((void**) &d_input, NODE_ACT_NUM * dimension * sizeof(float));
    cudaMemcpy(d_input, h_input, NODE_ACT_NUM * dimension * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaMalloc((void**) &d_row_ptr, (NODE_ACT_NUM) * sizeof(int));
    cudaMemcpy(d_row_ptr, row_ptr, (NODE_ACT_NUM) * sizeof(int), cudaMemcpyHostToDevice);
   
    cudaMalloc((void**) &d_col_ptr, (FEATURE_TOTAL) * sizeof(int));
    cudaMemcpy(d_col_ptr, col_ptr, FEATURE_TOTAL * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**) &d_output, NODE_ACT_NUM * dimension * sizeof(float));
    cudaMemset(d_output, 0, NODE_ACT_NUM * dimension * sizeof(float)); 
    
    cudaMalloc((void**) &d_degrees, (NODE_ACT_NUM) * sizeof(int));
    cudaMemcpy(d_degrees, h_degrees, NODE_ACT_NUM * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**) &d_part_ptr, (num_parts) * sizeof(int));
    cudaMemcpy(d_part_ptr, part_info[0], (num_parts) * sizeof(int), cudaMemcpyHostToDevice); 

    cudaMalloc((void**) &d_part_to_node, (num_parts) * sizeof(int));
    cudaMemcpy(d_part_to_node, part_info[1], (num_parts) * sizeof(int), cudaMemcpyHostToDevice); 

    if (dimension > 32) {
        /* Calculate the grid */
        const int grid = (num_parts * WARP_SIZE + BLOCK  - 1) / (BLOCK);      
        int shared_memory = (part_size * WARPS_PER_BLOCK * sizeof(int)) + (WARPS_PER_BLOCK * dimension * sizeof(float));
  
        for (int k = 0; k < repeats; k++) {
            spmm_forward_cuda_kernel_64<<<grid, BLOCK, shared_memory>>>(
                (float *) d_output, (float *) d_input, 
                (int *) d_row_ptr, (int *) d_col_ptr, (int *) d_degrees, 
                (int *) d_part_ptr, (int *) d_part_to_node,
                NODE_NUM, dimension, num_parts, part_size, WARP_SIZE, WARPS_PER_BLOCK);
            cudaDeviceSynchronize();
        }
    }
    else {
        const int ngs_to_process = WARP_SIZE / dimension;
        const int updated_dimension = ngs_to_process * dimension;
        int shared_memory = ngs_to_process * part_size * WARPS_PER_BLOCK * sizeof(int);
        const int grid = (num_parts * WARP_SIZE + BLOCK  - 1) / (BLOCK * ngs_to_process); 
        
        for (int k = 0; k < repeats; k++) {
            spmm_forward_cuda_kernel<<<grid, BLOCK, shared_memory>>>(
                (float *) d_output, (float *) d_input, 
                (int *) d_row_ptr, (int *) d_col_ptr, (int *) d_degrees, 
                (int *) d_part_ptr, (int *) d_part_to_node,
                NODE_NUM, dimension, num_parts, part_size, WARP_SIZE, WARPS_PER_BLOCK, updated_dimension, ngs_to_process);
            cudaDeviceSynchronize();
        }
    }

    
    cudaMemcpy(h_output, d_output, NODE_ACT_NUM * dimension * sizeof(float), cudaMemcpyDeviceToHost);

    //for (int i = 0; i < NODE_NUM; i++) {
    //    for (int j = 0; j < dimension; j++) {
    //        cout <<  h_output[i * dimension + j] << "-";
    //    }
    //    std::cout << endl;
    //}

    return 0;
}


std::vector<int *> build_part(
    int partSize, 
    int *indptr,
    int num_nodes
  )
   {
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
    const int warpPerBlock)
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
                for (int d = laneid; d < dim; d += dimWorker) {
                    partial_results[presult_base + d] = 0.0f;
                }
            
            if (laneid < dimWorker)
            #pragma unroll
            for (int d = laneid; d < dim; d += dimWorker) {
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
    const int ngs_to_process) 
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



