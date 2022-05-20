"""
@ Description: End to End Evaluate protein complex
"""

import os
from argparse import ArgumentParser
from shutil import rmtree
from pathlib import Path
import logging

import torch
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from scipy.special import softmax
from torch.utils.data import DataLoader

# customer modules
from src.clean_pipe import clean_pipe
from src.utils import list_to_txt, distance_helper, TestData, collate
from src.build_protein_graph import build_protein_graph
from src.run_DPROQ_li_multitask_v2_gate import DPROQ

parser = ArgumentParser(description='Evaluate protein complex structures')
parser.add_argument('--complex_folder', '-c', type=str, required=True)
parser.add_argument('--work_dir', '-w', type=str, help='working director to save temporary files', required=True)
parser.add_argument('--result_folder', '-r', type=str, help='The ranking result', required=True)
parser.add_argument('--threads', '-t', type=int, help='number of threads for graph generation and dataloader',
                    default=10, required=False)
parser.add_argument('--delete_tmp', '-s', type=bool, help='Save working director or not', default=False, required=False)
args = parser.parse_args()

complex_folder = args.complex_folder
work_dir = args.work_dir
result_folder = args.result_folder
threads = args.threads
delete_tmp = args.delete_tmp

if not os.path.isdir(complex_folder):
    raise FileNotFoundError(f'Please check complex folder {complex_folder}')
else:
    complex_folder = os.path.abspath(complex_folder)

if len(os.listdir(complex_folder)) == 0:
    raise ValueError(f'The complex folder is empty.')

if not os.path.isdir(work_dir):
    logging.info(f'Creating work folder')
    os.makedirs(work_dir)

if not os.path.isdir(result_folder):
    logging.info(f'Creating result folder')
    os.makedirs(result_folder)

dist_folder = os.path.join(work_dir, 'DIST')
tidy_folder = os.path.join(work_dir, 'TIDY')
os.makedirs(dist_folder, exist_ok=True)
os.makedirs(tidy_folder, exist_ok=True)

# clean input pdb -> reresiude ->  reatom, keep line starts with ATOM, TER, and END
if not os.path.isfile(os.path.join(work_dir, 'tidy.flag')):
    logging.info('Staring clean pdb file')
    for item in os.listdir(complex_folder):
        raw_pdb = os.path.join(complex_folder, item)
        clean_pipe(raw_pdb, tidy_folder)
    f = open(os.path.join(work_dir, 'tidy.flag'), "x")
else:
    logging.info('Clean pdb files generated.')


def check_distance(pdb_file: str, decoy_name: str, output_folder: str) -> None:
    """Check CA-CA, CB-CB, N-O distance maps are same shape."""
    caca_dist_shape = distance_helper(pdb_file, decoy_name, output_folder, atom_type='CA')
    cbcb_dist_shape = distance_helper(pdb_file, decoy_name, output_folder, atom_type='CB')
    no_dist_shape = distance_helper(pdb_file, decoy_name, output_folder, atom_type='NO')
    if caca_dist_shape != cbcb_dist_shape != no_dist_shape:
        ERROR_LIST.append(pdb_file)
        logging.warning('Requires CA-CA, CB-CB, N-O with same shape.')
        logging.info(f'{decoy_name} CA shape: {caca_dist_shape}, CB shape: {cbcb_dist_shape}, NO shape: {no_dist_shape}')

    return None


ERROR_LIST = []  # to record error targets
if not os.path.isfile(os.path.join(work_dir, 'dist.flag')):
    logging.info('Generating DIST file')
    for i in os.listdir(tidy_folder):
        decoy_name = i[:-9]  # remove _tidy.pdb
        tidy_pdb = os.path.join(tidy_folder, i)
        check_distance(pdb_file=tidy_pdb, decoy_name=decoy_name, output_folder=dist_folder)
    f = open(os.path.join(work_dir, 'dist.flag'), "x")
else:
    logging.info('DIST files generated')

tidy_pdb_list = os.listdir(tidy_folder)
tidy_pdb_list = [os.path.join(tidy_folder, i) for i in tidy_pdb_list]

if len(ERROR_LIST) != 0:
    logging.info(f'ERROR LIST contains {len(ERROR_LIST)} items:')
    for i in ERROR_LIST:
        logging.info(i)
    list_to_txt(ERROR_LIST, os.path.join(result_folder, 'error.list'))
    for i in ERROR_LIST:
        tidy_pdb_list.remove(i)

# generating graph
dgl_folder = os.path.join(work_dir, 'DGL')
os.makedirs(dgl_folder, exist_ok=True)


def wrapper(data_entry: str):
    build_protein_graph(pdb_file=data_entry,
                        model_name=data_entry.split('/')[-1][:-9],  # remove _tidy.pdb
                        out=dgl_folder,
                        dist_path=dist_folder)


if not os.path.isfile(os.path.join(work_dir, 'dgl.flag')):
    logging.info('Generating DGL files')
    Parallel(n_jobs=10)(delayed(wrapper)(i) for i in tidy_pdb_list)
    f = open(os.path.join(work_dir, 'dgl.flag'), "x")
else:
    logging.info('DGL files generated')

# build data loader
eval_set = TestData(dgl_folder)
eval_loader = DataLoader(eval_set.data,
                         batch_size=8,
                         num_workers=4,
                         pin_memory=True,
                         collate_fn=collate,
                         shuffle=False)

# load pre_train model
device = torch.device('cuda')  # set cuda device
current_path = Path().absolute()
ckpt_file = f'{current_path}/ckpt/pre_train_seed_222.ckpt'
model = DPROQ.load_from_checkpoint(ckpt_file)
logging.info(f'Loading {ckpt_file}')
model = model.to(device)
model.eval()  # turn on model eval mode

pred_dockq = []
pred_class = []
# expected_value_list = []

for idx, batch_graphs in enumerate(eval_loader):
    batch_x = batch_graphs.ndata['f'].to(torch.float)
    batch_e = batch_graphs.edata['f'].to(torch.float)
    batch_graphs = batch_graphs.to(device)
    batch_x = batch_x.to(device)
    batch_e = batch_e.to(device)
    batch_scores, batch_class = model.forward(batch_graphs, batch_x, batch_e)
    pred_dockq.extend(batch_scores.cpu().data.numpy().tolist())
    pred_class.extend(batch_class.cpu().data.numpy().tolist())

# for i in pred_class:
#     ep_score = np.sum(softmax(i) * np.array([0, 1, 2, 3])) / 3
#     expected_value_list.append(ep_score.item())

pred_dockq = [i[0] for i in pred_dockq]

model_list = os.listdir(dgl_folder)
model_list = [i.split('.')[0] for i in model_list]
df = pd.DataFrame(list(zip(model_list, pred_dockq)), columns=['MODEL', 'PRED_DOCKQ'])
df.sort_values(by='PRED_DOCKQ', ascending=False, inplace=True)
df.loc[:, 'PRED_DOCKQ'] = df.loc[:, 'PRED_DOCKQ'].round(5)
df.to_csv(os.path.join(result_folder, 'Ranking.csv'), index=False)
logging.info(f"Result is {os.path.join(result_folder, 'DOCKQ_ranking.csv')}")

# df_2 = pd.DataFrame(list(zip(model_list, expected_value_list)), columns=['MODEL', 'Escore'])
# df_2.sort_values(by='Escore', ascending=False, inplace=True)
# df_2.loc[:, 'Escore'] = df_2.loc[:, 'Escore'].round(5)
# df_2.to_csv(os.path.join(result_folder, 'Escore_ranking.csv'), index=False, sep=' ', header=False)
# logging.info(f"Result2 is {os.path.join(result_folder, 'Escore_ranking.csv')}")

if delete_tmp:
    logging.info('DELETING ALL TEMPORARY FILES, ONLY KEEP FINAL RESULTS')
    rmtree(work_dir)
    logging.info('DELETED ALL TEMPORARY FILE, ONLY KEEP FINAL RESULTS')
