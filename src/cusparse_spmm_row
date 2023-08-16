#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMM
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

using namespace std;

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}


int main(int argc, char** argv)
{
    ifstream matrix_file(argv[1]);
    int DIM = atoi(argv[2]);
    int repeat = atoi(argv[3]);
    
    int NODE_ACT_NUM = 0;
    int NODE_NUM  = 0;
    int FEATURE_TOTAL = 0;

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
    
    int *row_ptr = (int *) malloc(NODE_ACT_NUM * sizeof(int));
    int *col_ptr = (int *) malloc(FEATURE_TOTAL * sizeof(int));
    
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
    
    int   A_num_rows = NODE_NUM;
    int   A_num_cols = NODE_NUM;
    int   A_nnz = FEATURE_TOTAL;
    int   B_num_rows = NODE_NUM;
    int   B_num_cols = DIM;

    int   A_num_off = NODE_ACT_NUM;
    int   ldb             = NODE_NUM;
    int   ldc             = NODE_NUM;
    int   B_size          = ldb * DIM;
    int   C_size          = ldc * DIM;


    cout << "A size: (" << A_num_rows << ", " << A_num_cols << ")" << endl;
    cout << "B size: (" << B_num_rows << ", " << B_num_cols << ")" << endl;
    cout << "C size: (" << ldc << ", " << B_num_cols << ")" << endl;

    cout << "A #non-zeros: " << A_nnz << endl << endl;

    int   *hA_csrOffsets;
    int   *hA_columns;

    float *hB;
    float *hC;

    float alpha           = 1.0f;
    float beta            = 0.0f;

   

    posix_memalign((void**) &hC, 64, (C_size) * sizeof(float)); 
    //--------------------------------------------------------------------------
    // Read A from file
    float *hA_values = new float[FEATURE_TOTAL];
    
    for (int i = 0; i < FEATURE_TOTAL; i++) hA_values[i] = 1;

    //--------------------------------------------------------------------------
    // Read B from file
    posix_memalign((void**) &hB, DIM, (B_num_rows * B_num_cols) * sizeof(float)); 
    for (int i = 0; i < B_num_rows * B_num_cols; i++) hB[i] = 1;

    cout << "Start multiplication..." << endl << endl;
    

    //--------------------------------------------------------------------------
    // Device memory management
    int   *dA_csrOffsets, *dA_columns;
    float *dA_values, *dB, *dC;
    CHECK_CUDA( cudaMalloc((void**) &dA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_nnz * sizeof(int))    )
    CHECK_CUDA( cudaMalloc((void**) &dA_values,  A_nnz * sizeof(float))  )
    CHECK_CUDA( cudaMalloc((void**) &dB,         B_size * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dC,         C_size * sizeof(float)) )

    CHECK_CUDA( cudaMemcpy(dA_csrOffsets, row_ptr,
                           (A_num_rows + 1) * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_columns, col_ptr, A_nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB, B_size * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC, hC, C_size * sizeof(float),
                           cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, DIM, dB,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) )
    // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, DIM, dC,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )


    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;
    using std::chrono::nanoseconds;

    float time = 0;
    for (int k=0; k < repeat; k++){
        auto t1 = high_resolution_clock::now();

        // execute SpMM
        CHECK_CUSPARSE( cusparseSpMM(handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                    CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) )

        auto t2 = high_resolution_clock::now();
        auto ms_int = duration_cast<nanoseconds>(t2 - t1);
        time = time + ms_int.count();
    }
    std::cout << "Time elapsed (us): " << time/ (float) (200*1000) << "\n";


    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA( cudaMemcpy(hC, dC, C_size * sizeof(float),
                           cudaMemcpyDeviceToHost) )
    // for (int i = 0; i < A_num_rows; i++) {
    //     for (int j = 0; j < B_num_cols; j++) {
    //         cout << hC[i + j * ldc] << ",";
    //     }
    //     cout << endl;
    // }
                           // int correct = 1;
    // for (int i = 0; i < A_num_rows; i++) {
    //     for (int j = 0; j < B_num_cols; j++) {
    //         if (hC[i + j * ldc] != hC_result[i + j * ldc]) {
    //             correct = 0; // direct floating point comparison is not reliable
    //             break;
    //         }
    //     }
    // }
    // if (correct)
    //     printf("spmm_csr_example test PASSED\n");
    // else
    //     printf("spmm_csr_example test FAILED: wrong result\n");
    //--------------------------------------------------------------------------
}
