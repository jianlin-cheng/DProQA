"""
@ Description: Clean raw pdb
@ Description: Shawn Chen
@ Date: 2022-03-28
"""

import os
from shutil import copy
from subprocess import call
from typing import List
from biopandas.pdb import PandasPdb
from pathlib import Path
from utils import txt_to_list, list_to_txt

# get father path
father_path = Path(__file__).resolve().parents[1]

tool_path = f'{father_path}/src/tool/'
chain_tool = f'{tool_path}/pdb_selchain.py'
keep_tool = f'{tool_path}/keep_start_with_atom.py'
atom_tool = f'{tool_path}/pdb_reatom.py'
residue_tool = f'{tool_path}/pdb_reres.py'
merge_tool = f'{tool_path}/pdb_merge.py'
tidy_tool = f'{tool_path}/pdb_tidy.py'


def clean_pipe(pdb_file: str, output_folder: str, save_flag=False) -> None:
    """
    Reformat pdb by:
    1. set each chain's residue number starts from 1
    2. set whole pdb file's atom number starts from 1
    3. add TER
    """

    # add postfix for file if it does not end with .pdb
    ENDSPDB = True
    if pdb_file.endswith('.pdb'):
        model_name = pdb_file.split('/')[-1].split('.')[0]
    else:
        ENDSPDB = False
        model_name = pdb_file.split('/')[-1]
        new_pdb_file = pdb_file + '.pdb'
        copy(pdb_file, new_pdb_file)
        pdb_file = new_pdb_file
    
    # get chain list
    ppdb = PandasPdb().read_pdb(pdb_file)
    sequence = ppdb.amino3to1()
    chain_list_tmp = sequence.loc[:, 'chain_id'].tolist()
    chain_list = []
    [chain_list.append(x) for x in chain_list_tmp if x not in chain_list]

    # 1. Keep ATOM
    tmp_file_1 = os.path.join(output_folder, model_name + '_tmp_v1.pdb')
    cmd = f'python {keep_tool} {pdb_file} {tmp_file_1}'
    call([cmd], shell=True)

    # 2. Split chain
    tmp_pdb_list = []
    for idx, item in enumerate(chain_list):
        tmp_pdb_file = os.path.join(output_folder, model_name + '_' + chain_list[idx] + '.pdb')
        cmd = f'python {chain_tool} -{chain_list[idx]} {tmp_file_1} > {tmp_pdb_file}'
        call([cmd], shell=True)
        tmp_pdb_list.append(tmp_pdb_file)

    # 3. Re-residue
    re_residue_list = []
    for i in tmp_pdb_list:
        chain_id = i[-5]
        tmp_pdb_file = os.path.join(output_folder, model_name + '_' + chain_id + '_reresidue.pdb')
        cmd = f'python {residue_tool} -1 {i} > {tmp_pdb_file}'
        call([cmd], shell=True)
        re_residue_list.append(tmp_pdb_file)

    # 4. Combine
    merge_list = ' '.join(i for i in re_residue_list)
    merged_dile = os.path.join(output_folder, model_name + '_merged.pdb')
    cmd = f"python {merge_tool} {merge_list} > {merged_dile}"
    call([cmd], shell=True)

    # 5. Re-atom
    re_atom_file = os.path.join(output_folder, model_name + '_reatom.pdb')
    cmd = f"python {atom_tool} -1 {merged_dile} > {re_atom_file}"
    call([cmd], shell=True)

    # 6. tidy
    tidy_file = os.path.join(output_folder, model_name + '_tidy.pdb')
    cmd = f"python {tidy_tool} {re_atom_file} > {tidy_file}"
    call([cmd], shell=True)

    # remove tmp files
    if not save_flag:
        print('Remove tmp files')
        os.remove(tmp_file_1)
        for i in tmp_pdb_list:
            os.remove(i)
        for i in re_residue_list:
            os.remove(i)
        os.remove(merged_dile)
        os.remove(re_atom_file)
    
    if not ENDSPDB:
        if os.path.isfile(new_pdb_file):
            os.remove(new_pdb_file)
