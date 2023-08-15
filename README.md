# MergePath-SpMM

Parallel Sparse Matrix-Matrix Algorithm for Graph Neural Network Acceleration

# Citation

Mohsin Shan, Deniz Gurevin, Jared Nye, Caiwen Ding, Omer Khan, "MergePath-SpMM: Parallel Sparse Matrix-Matrix Algorithm for Graph Neural Network Acceleration", IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS 2023), Raleigh, NC, April 23-25, 2023.

# Requirements
CuSPARSE 10 or higer.

# Datasets
Use the following commands to download and extract datasets
```
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1SF8rfz66qJ-b0MT5v5qWuxNJpydb3_hj' -O datasets.tar
tar -xvf datasets.tar
```

# Compilation
Use the following command to build all the kernels.
```
make all
```
To build a specific kernel (row_wise, nz_splitting, mergepath) use following command.
```
make <kernel_name>
```

# Datasets
The script scripts/fetch_dataset.py can be used to download any dataset from the pytorch geometric datasets. It will also convert the data into appropriate format used by the kernels.

# Execution
Use the following command to execute the kernels.
```
nvprof ./<kernel_name> <params>
```



