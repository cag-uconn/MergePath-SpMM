# MergePath-SpMM

Parallel Sparse Matrix-Matrix Algorithm for Graph Neural Network Acceleration

# Citation
@INPROCEEDINGS{10158225,
  author={Shan, Mohsin and Gurevin, Deniz and Nye, Jared and Ding, Caiwen and Khan, Omer},
  booktitle={2023 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)}, 
  title={MergePath-SpMM: Parallel Sparse Matrix-Matrix Algorithm for Graph Neural Network Acceleration}, 
  year={2023},
  volume={},
  number={},
  pages={145-156},
  doi={10.1109/ISPASS57527.2023.00023}}

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
# Additional Results
![Additional Results](additional_results.png)
The figure expands on the experimental data presented in the paper. It shows speedups of cuSparse using the XW matrix in both column and row major formats, as well as MergePath-SpMM and GNNAdvisor-opt over GNNAdvisor at default dimension size of 16. In cuSPARSE, the dense matrix can be stored in memory using either row-major or column-major arrangement. The paper only shows results using the column-major format. This format demonstrates better performance for structured graphs, as the column-wise parallelization strategy can be applied, offering substantial spatial locality. Columns are typically larger than rows in size due to the number of nodes, while the size of rows depends on dimensions. Furthermore, the structured and balanced nature of the A matrix prevents load imbalance in column-wise strategy. However, this format doesn't perform effectively with power law graphs, which experience load imbalance due to the unstructured nature of the graph. For power law graphs, previous studies have shown that employing a row-wise parallelization strategy yields better results, which is also shown here. On average, cuSPARSE with the row-major format outperforms GNNAdvisor as majority of the graphs evaluated are power-law graphs.
