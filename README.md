# MergePath-SpMM
Merg-path based Parallel Sparse Matrix-Matrix Algorithm for Irregular Sparse Matrices

# Requirements
CuSPARSE 10 or higer.

# Compilation
```
cd src
nvcc MergePathSpmm.cu -o merge-path
```

# Datasets
The datasets are present in the input folder.
To run a sepecific dataset update MergePathSpmm.cu:22
```
#include "input/<dataset_name>.h"
```

# Execution
Use the following command to execute MergePath-SpMM.
```
nvprof ./merge-path <COST>
```

# Measurements
The kernel time will be is given by 'spmm_forward_cuda_kernel_mp' entry.


