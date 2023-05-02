# MergePath-SpMM

Parallel Sparse Matrix-Matrix Algorithm for Graph Neural Network Acceleration

# Citation

Mohsin Shan, Deniz Gurevin, Jared Nye, Caiwen Ding, Omer Khan, "MergePath-SpMM: Parallel Sparse Matrix-Matrix Algorithm for Graph Neural Network Acceleration", IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS 2023), Raleigh, NC, April 23-25, 2023.

# Requirements
CuSPARSE 10 or higer.

# Compilation
```
cd src
nvcc MergePathSpmm.cu -O3 -o merge-path 
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


