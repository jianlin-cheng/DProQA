# Description

DProQ: A Gated-Graph Transformer for Predicting the Quality of Protein Complex Structure

# Installation

1. Set up Conda environment locally a new python environment
   
   ```bash
   conda env create --name DProQ -f environment.yml
   ```

2. Activate conda environment
   ```bash
   conda acitvate DPRoQ
   ```
   
# Running DProQ for inference
   
For inference, the GPU is required. We provide few protein complexes in ./example/raw_pdb folder.
Otherwise, you could do inference for your own protein complex structure.
All protein file should be ended by .pdb.

## Usage
   ```bash
   python ./evaluate_complex.py
   -c --complex_folder     Raw protien complex complex_folder
   -w --work_dir           Working directory to save all intermedia files and folders, it will created if it is not exits
   -r --result_folder      Result folder to save two ranking results, it will created if it is not exits
   -r --threads            Number of threads for parallel feature generation and dataloader, default=10
   -s --save_tmp           Set True to save work_dir and intermedia files, Set False to delete work_dir. default=False
   ```
## Run EMA pipeline
   
GPU REQUIRED!!!
   ```bash
   python ./evaluate_complex.py -c ./example/raw_pdb -w ./example/work/ -r ./example/result
   ```

4. The ranking results. ranking_res.csv is stored in result folder.
