"""
@ Description: Data Generator
"""


import os
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
from tqdm import tqdm
from joblib import Parallel, delayed

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler

import dgl
import torch



# from src.clean_pipe import clean_pipe
from src.utils import pdb2fasta, run_dssp, laplacian_positional_encoding, \
    ss3_one_hot, sequence_one_hot, pdb2graph_new_chain_info, update_node_feature, update_edge_feature


def filter_atoms(test_df: pd.DataFrame, atom_type: str) -> pd.DataFrame:
    if atom_type == 'CB':
        return test_df[((test_df.loc[:, 'residue_name'] == 'GLY') & (test_df.loc[:, 'atom_name'] == 'CA')) \
                            | (test_df.loc[:, 'atom_name'] == 'CB')]
    elif atom_type == 'CA':
        return test_df[test_df.loc[:, 'atom_name'] == 'CA']
    elif atom_type == 'NO':
        return test_df[(test_df.loc[:, 'atom_name'] == 'N') | (test_df.loc[:, 'atom_name'] == 'O')]
    else:
        raise ValueError('Atom type should be CA, CB or NO.')


def distance_helper_v2(pdb_file: str) -> list(np.ndarray, np.ndarray, np.ndarray):
    """
    Calculate CA-CA, or CB-CB or N-O distance for a pdb file
    return: distance matrix [CA, CB, NO]
    """
    ppdb = PandasPdb().read_pdb(pdb_file)
    test_df = ppdb.df['ATOM']

    distance_matrix_list = []
    
    for atom_type in ['CA', 'CB', 'NO']:
        filtered_df = filter_atoms(test_df, atom_type)

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

        distance_matrix_list.append(real_dist)
    
    return distance_matrix_list


def build_protein_graph(pdb_file: str,
                        model_name: str,
                        out: str,
                        dist_matirx: list(np.array, np.array, np.array)):
    """Build KNN graph and assign node and edge features. node feature: N * 35, Edge feature: E * 6"""

    print(f'Processing {model_name}')
    scaler = MinMaxScaler()

    # 1. extract sequence from pdb
    sequence = pdb2fasta(pdb_file)

    # 3. build graph and extract edges and PEEE
    _, g, edges, peee = pdb2graph_new_chain_info(pdb_file, knn=10)

    # 4. node features
    # 4.1. SS8, rasa, phi, psi
    feature_df = run_dssp(pdb_file)
    ss3_feature = ss3_one_hot(feature_df)
    phi_feature = torch.tensor(scaler.fit_transform(torch.tensor(feature_df['phi'].values).reshape(-1, 1)))
    psi_feature = torch.tensor(scaler.fit_transform(torch.tensor(feature_df['psi'].values).reshape(-1, 1)))
    rasa_feature = torch.tensor(feature_df['rasa']).reshape(-1, 1)

    # 4.2. sequence one hot as node
    one_hot_feature = sequence_one_hot(sequence)

    # 4.3. laplacian positional encoding
    lap_enc_feature = laplacian_positional_encoding(g, pos_enc_dim=8)

    # 5. edge features
    # 5.1. edge sine position encoding
    edge_sin_pos = torch.sin((g.edges()[0] - g.edges()[1]).float()).reshape(-1, 1)

    # 5.2. CA-CA, CB-CB, N-O distance
    # load distance map
    CACA = dist_matirx[0]
    CBCB = dist_matirx[1]
    NO = dist_matirx[2]
    assert CACA.shape == CBCB.shape == NO.shape, f'{model_name} distance map shape not match: {CACA.shape}, {CBCB.shape}, {NO.shape}'

    caca_feature, cbcb_feature, no_feature = [], [], []

    for i in edges:
        caca_feature.append(CACA[i])
        cbcb_feature.append(CBCB[i])
        no_feature.append(NO[i])

    contact_feature = torch.tensor([1 if cb_distance < 8.0 else 0 for cb_distance in cbcb_feature]).reshape(-1, 1)
    caca_feature = torch.tensor(scaler.fit_transform(torch.tensor(caca_feature).reshape(-1, 1)))
    cbcb_feature = torch.tensor(scaler.fit_transform(torch.tensor(cbcb_feature).reshape(-1, 1)))
    no_feature = torch.tensor(scaler.fit_transform(torch.tensor(no_feature).reshape(-1, 1)))

    # 6. add feature to graph
    update_node_feature(g, [ss3_feature, rasa_feature,
                            phi_feature, psi_feature,
                            lap_enc_feature, one_hot_feature])

    update_edge_feature(g, [edge_sin_pos, caca_feature,
                            cbcb_feature, no_feature,
                            contact_feature, peee])

    dgl.save_graphs(filename=os.path.join(out, f'{model_name}.dgl'), g_list=g)
    return None
    

def wrapper(pdb_file: str, save_folder: str):
    pdb_id = pdb_file.split('.')[0].split('/')[-1].strip()
    
    dist_matirx_list = distance_helper_v2(pdb_file)
    
    build_protein_graph(pdb_file=pdb_file,
                        model_name=pdb_id,
                        out=save_folder,
                        dist_matirx=dist_matirx_list)
    return None
        
        
if __name__ == '__name__':
    parser = ArgumentParser(description='Evaluate protein complex structures')
    parser.add_argument('--input_pdb_folder', '-p', type=str, help='pdb file path', required=True)
    parser.add_argument('--dgl_save_folder', '-d', type=str, help='output folder', required=True)
    parser.add_argument('--cores', '-c', type=int, help='multi-cores', required=False, default=10)
    args = parser.parse_args()

    input_pdb_folder = args.input_pdb_folder
    dgl_save_folder = args.dgl_save_folder
    cores = args.cores
    
    if not os.path.isdir(input_pdb_folder):
        raise FileNotFoundError(f'Please check input pdb folder {input_pdb_folder}')
    input_pdb_folder = os.path.abspath(input_pdb_folder) # get absolute path
    
    pdbs = os.listdir(input_pdb_folder)
    pdbs = [os.path.join(input_pdb_folder, pdb) for pdb in pdbs if pdb.endswith('.pdb')]        
    if len(pdbs) == 0:
        raise ValueError(f'The pdb folder is empty.')
    
    if not os.path.isdir(dgl_save_folder):
        print(f'Creating output folder')
        os.makedirs(dgl_save_folder)
    
    dgl_save_folder = os.path.abspath(dgl_save_folder) # get absolute path
    
    Parallel(n_jobs=cores)(delayed(wrapper)(pdb_file, dgl_save_folder) for pdb_file in tqdm(pdbs))
    
    print('All done.')