"""
@ Description: helper functions
"""

import os
from copy import deepcopy
import pandas as pd
import torch
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
import dgl
import scipy.sparse as sp
import numpy as np
from biopandas.pdb import PandasPdb
from typing import List, Union
from sklearn.metrics.pairwise import euclidean_distances
from torch.utils.data import Dataset


def run_dssp(pdb_file: str) -> pd.DataFrame:
    """Run biopython DSSP for SS8(3), RASA Angle(Phi, Phi)"""
    pdb_name = pdb_file.split('/')[-1].split('.')[0]
    p = PDBParser()

    structure = p.get_structure(pdb_name, pdb_file)
    model = structure[0]
    dssp = DSSP(model, pdb_file)
    key_list = list(dssp.keys())

    ss8_list = []
    rasa_list = []
    phi_list = []
    psi_list = []

    for key in key_list:
        ss8, rasa, phi, psi = dssp[key][2:6]
        ss8_list.append(ss8)
        rasa_list.append(rasa)
        phi_list.append(phi)
        psi_list.append(psi)

    feature_df = pd.DataFrame(list(zip(ss8_list, rasa_list, phi_list, psi_list)),
                              columns=['ss8', 'rasa', 'phi', 'psi'])

    return feature_df


def laplacian_positional_encoding(g: dgl.DGLGraph, pos_enc_dim: int) -> torch.Tensor:
    """
        Graph positional encoding v/ Laplacian eigenvectors
        :return torch.Tensor (L, pos_enc_dim)
    """

    # Laplacian
    A = g.adjacency_matrix()
    s = torch.sparse_coo_tensor(indices=A.coalesce().indices(),
                                values=A.coalesce().values(),
                                size=A.coalesce().size())
    A = s.to_dense()
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.A)
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    laplacian_feature = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float().reshape(-1, pos_enc_dim)
    return laplacian_feature


def ss3_one_hot(df: pd.DataFrame) -> torch.Tensor:
    """treat ss8 to ss3 get one hot encoding, return size L * 3"""
    tokens_dict = {'H': 0, 'B': 2, 'E': 2, 'G': 0, 'I': 0, 'T': 1, 'S': 1, '-': 1}
    one_hot_array = np.zeros([df.shape[0], 3])
    ss8_list = df.ss8.to_list()

    for idx, item in enumerate(ss8_list):
        if item not in tokens_dict:
            raise KeyError(f'This {item} is not secondary structure type.')
        col_idx = tokens_dict[item]
        one_hot_array[idx, col_idx] = 1

    return torch.from_numpy(one_hot_array).reshape(-1, 3)


def pdb2fasta(pdb_file: str) -> str:
    """extract sequence from pdb file"""

    amino = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N',
             'ASP': 'D', 'CYS': 'C', 'GLU': 'E',
             'GLN': 'Q', 'GLY': 'G', 'HIS': 'H',
             'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
             'MET': 'M', 'PHE': 'F', 'PRO': 'P',
             'SER': 'S', 'THR': 'T', 'TRP': 'W',
             'TYR': 'Y', 'VAL': 'V'}

    if not os.path.isfile(pdb_file):
        raise FileExistsError(f'PDB File does not exist {pdb_file}')

    with open(pdb_file, 'r') as file:
        content = file.readlines()

    seq = []
    prev_mark = -1

    for line in content:
        if line[:4] == 'ATOM':
            pos_mark = line[22: 26].strip()
            if pos_mark != prev_mark:
                seq.append(amino[line[17:20]])
            prev_mark = pos_mark

    return "".join(seq)


def read_fasta(fasta_file: str):
    """
    Read fasta file, return sequence id, length and content
    Support Fasta format example:
    >Target_id|length
    CCCCCCCCCCCCCCCCC
    """
    with open(fasta_file) as f:
        content = f.readlines()
        f.close()
    seq = ''
    if len(content) == 1:
        seq += content[0].strip()
        return seq
    else:
        target_id, length = content[0].split('|')
        target_id = target_id.strip().strip('>')
        length = int(length.strip())
        seq += content[1].strip()
    return target_id, length, seq


def sequence_one_hot(fasta_file: str) -> torch.Tensor:
    """Sequence one hot encoding, return size L * 21"""
    tokens_dict_regular_order = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
                                 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
                                 'M': 10, 'N': 11, 'P': 12, 'Q': 13,
                                 'R': 14, 'S': 15, 'T': 16, 'V': 17,
                                 'W': 18, 'Y': 19, 'X': 20}

    seq = fasta_file
    length = len(seq)

    one_hot_array = np.zeros([length, 21])

    for idx, item in enumerate(seq.upper()):
        if item not in tokens_dict_regular_order.keys():
            item = 'X'
        col_idx = tokens_dict_regular_order[item]
        one_hot_array[idx, col_idx] = 1

    return torch.from_numpy(one_hot_array).reshape(-1, 21)


def edge_sin_pos(g: dgl.DGLGraph) -> torch.Tensor:
    """Edge wise encoding"""
    return torch.sin((g.edges()[0] - g.edges()[1]).float()).reshape(-1, 1)


def update_node_feature(graph: dgl.DGLGraph, new_node_features: List) -> None:
    """Node feature update helper"""
    for node_feature in new_node_features:
        if not graph.ndata:
            graph.ndata['f'] = node_feature
        else:
            graph.ndata['f'] = torch.cat((graph.ndata['f'], node_feature), dim=1)


def update_edge_feature(graph: dgl.DGLGraph, new_edge_features: List) -> None:
    """Edge feature update helper"""
    for edge_feature in new_edge_features:
        if not graph.edata:
            graph.edata['f'] = edge_feature
        else:
            graph.edata['f'] = torch.cat((graph.edata['f'], edge_feature), dim=1)
    return None


def remove_n(lst: List, pattern='\n') -> List:
    return [i.strip(pattern) for i in lst]


def txt_to_list(txt_file: str, pattern='\n') -> List:
    """read txt file to list"""
    with open(txt_file, 'r') as f:
        tmp_list = f.readlines()
    tmp_list = remove_n(tmp_list, pattern=pattern)
    return tmp_list


def list_to_txt(lst: List, txt_file: str) -> None:
    """read out list to txt file"""
    with open(txt_file, 'w') as f:
        for i in lst:
            f.writelines(i + '\n')
    return None


def pdb2graph_new_chain_info(pdb_file: str, knn=10):
    """
    Build KNN graph for a protein, return graph and src vertex
    and end dst vertex for graph, without self-loop, PEEE chain ID
    """
    atom_df = PandasPdb().read_pdb(pdb_file).df['ATOM']
    atom_df_full = deepcopy(atom_df)  # return all atom df for distance calculation
    atom_df = atom_df[atom_df.loc[:, 'atom_name'] == 'CA']
    node_coords = torch.tensor(atom_df[['x_coord', 'y_coord', 'z_coord']].values, dtype=torch.float32)
    protein_graph = dgl.knn_graph(node_coords, knn)
    protein_graph = protein_graph.remove_self_loop()  # remove self loop
    srcs = protein_graph.edges()[0]
    dsts = protein_graph.edges()[1]

    edges = list(zip(srcs, dsts))

    # CA-CA distance
    atom_df_ca = atom_df[atom_df.loc[:, 'atom_name'] == 'CA']
    chain_id_list = atom_df_ca.loc[:, 'chain_id'].tolist()
    chain_id_dict = dict(zip([i for i in range(len(chain_id_list))], chain_id_list))  # {idx: chain_id}

    uniform_chain_feature = []
    for i in edges:
        u, v = i
        u = u.item()
        v = v.item()
        if (chain_id_dict[u] == chain_id_list[v]) and (abs(u - v) == 1):
            uniform_chain_feature.append(1)
        else:
            uniform_chain_feature.append(0)
    return atom_df_full, protein_graph, edges, torch.tensor(uniform_chain_feature).reshape(-1, 1)


def distance_helper(pdb_file: str, pdb_name: str,
                    output_folder: str, atom_type='CB',
                    save_flag=True) -> Union[tuple, np.ndarray]:
    """Calculate CA-CA, or CB-CB or N-O distance for a pdb file"""
    ppdb = PandasPdb().read_pdb(pdb_file)
    test_df = ppdb.df['ATOM']

    if atom_type == 'CB':
        # GLY does not have CB, use CA to instead of.
        filtered_df = test_df[((test_df.loc[:, 'residue_name'] == 'GLY') & (test_df.loc[:, 'atom_name'] == 'CA')) \
                              | (test_df.loc[:, 'atom_name'] == 'CB')]
    elif atom_type == 'CA':
        filtered_df = test_df[test_df.loc[:, 'atom_name'] == 'CA']
    elif atom_type == 'NO':
        filtered_df = test_df[(test_df.loc[:, 'atom_name'] == 'N') | (test_df.loc[:, 'atom_name'] == 'O')]
    else:
        raise ValueError('Atom type should be CA, CB or NO.')

    if atom_type != 'NO':
        coord = filtered_df.loc[:, ['x_coord', 'y_coord', 'z_coord']].values.tolist()
        real_dist = euclidean_distances(coord)
    else:
        coord_N = filtered_df[filtered_df.loc[:, 'atom_name'] == 'N'].loc[:,
                  ['x_coord', 'y_coord', 'z_coord']].values.tolist()
        coord_O = filtered_df[filtered_df.loc[:, 'atom_name'] == 'O'].loc[:,
                  ['x_coord', 'y_coord', 'z_coord']].values.tolist()
        real_dist = euclidean_distances(coord_N, coord_O)  # up-triangle N-O, low-triangle O-N

    real_dist = np.round(real_dist, 3)

    if save_flag:
        np.save(file=os.path.join(output_folder, pdb_name + f'_{atom_type}.npy'), arr=real_dist)
        return real_dist.shape
    else:
        return real_dist


class TestData(Dataset):
    """Data loader"""
    def __init__(self, dgl_folder: str):
        self.data_list = os.listdir(dgl_folder)
        self.data_list = [os.path.join(dgl_folder, i) for i in self.data_list]
        self.data = []
        self._prepare()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _prepare(self):
        for i in range(len(self.data_list)):
            g, tmp = dgl.data.utils.load_graphs(self.data_list[i])
            self.data.append(g[0])


def collate(samples) -> dgl.DGLGraph:
    """Customer collate function"""
    batched_graph = dgl.batch(samples)
    return batched_graph
