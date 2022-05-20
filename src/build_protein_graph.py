"""
@ Description: Protein to DGL graph with node and edge features
"""

import os
import dgl
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.utils import pdb2fasta, run_dssp, laplacian_positional_encoding, \
    ss3_one_hot, sequence_one_hot, pdb2graph_new_chain_info, update_node_feature, update_edge_feature


def build_protein_graph(pdb_file: str,
                        model_name: str,
                        out: str,
                        dist_path: str) -> None:
    """Build KNN graph and assign node and edge features. node feature: N * 35, Edge feature: E * 6"""
    if not os.path.isdir(dist_path):
        raise FileNotFoundError(f'Please check distance folder {dist_path}')

    print(f'Processing {model_name}')
    scaler = MinMaxScaler()

    # 1. extract sequence from pdb
    sequence = pdb2fasta(pdb_file)

    # 3. build graph and extract edges and PEEE
    atom_df_full, g, edges, peee = pdb2graph_new_chain_info(pdb_file, knn=10)

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
    CACA = np.load(os.path.join(dist_path, model_name + '_CA.npy'))
    CBCB = np.load(os.path.join(dist_path, model_name + '_CB.npy'))
    NO = np.load(os.path.join(dist_path, model_name + '_NO.npy'))

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
    print(f'{model_name} SUCCESS')
    return None
