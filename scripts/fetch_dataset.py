# https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html

from torch_geometric.datasets import *
from torch_geometric import utils as scipy_utils


if __name__ == "__main__":

    graph = Planetoid(root = "./", name = "Cora")
    scipy_coo = scipy_utils.to_scipy_sparse_matrix(graph.data.edge_index)
    scipy_csr = scipy_coo.tocsr()
    
    with open("graph.csr", "w") as csr_file:
        for elem in scipy_csr.indptr:
            csr_file.write(str(elem) + ",")
    
        csr_file.write("\n")

        for elem in scipy_csr.indices:
            csr_file.write(str(elem)+ ",")