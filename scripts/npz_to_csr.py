# Check this website to get list of available datasets
# https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html

import numpy as np
import argparse
from scipy.sparse import coo_matrix

parser = argparse.ArgumentParser()
parser.add_argument("--graph", type=str, help="name of graph to convert to npz format")
parser.add_argument("--output", type=str, help="name of the output file")
args = parser.parse_args()

graph_obj = np.load(args.graph)
src_li = graph_obj['src_li']
dst_li = graph_obj['dst_li']
data = np.ones(len(src_li))

coo_mat = coo_matrix((data, (src_li, dst_li)))
csr_mat = coo_mat.tocsr()

with open(args.output, "w") as csr_file:
    for elem in csr_mat.indptr:
        csr_file.write(str(elem) + ",")

    csr_file.write("\n")

    for elem in csr_mat.indices:
        csr_file.write(str(elem)+ ",")
