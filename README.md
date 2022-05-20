# DProQ: A Gated-Graph Transformer for Predicting the Quality of Protein Complex Structure

DProQ, is a Gated-Graph Transformer model for end-to-end protein complex structure's quality evaluation.
DProQ achieves significant speed-ups and better quality compared to current baseline method. If you have any 
questions or suggestions, please contact us by xcbh6@umsystem.edu. We are happy help!



# Citation
If you think our work is helpful, please cite this work.


# Installation
1. Download this repository
   ```bash
   git clone https://github.com/BioinfoMachineLearning/DProQ.git
   ```

2. Set up Conda environment locally a new python environment
   
   ```bash
   conda env create --name DProQ -f environment.yml
   ```

3. Activate conda environment
   ```bash
   conda acitvate DPRoQ
   ```

# Usage
   ```bash
   python evaluate_complex.py
   -c --complex_folder     Raw protien complex complex_folder
   -w --work_dir           Working directory to save all intermedia files and folders, it will created if it is not exits
   -r --result_folder      Result folder to save two ranking results, it will created if it is not exits
   -r --threads            Number of threads for parallel feature generation and dataloader, default=10
   -s --save_tmp           Set True to save work_dir and intermedia files, Set False to delete work_dir. default=False
   ```

# Using DProQ for protein complex structure quality assessment.
DProQ requires GPU. We provide few protein complexes in example/raw_pdb folder. Otherwise, you could do inference for your own protein complex structure.
All protein file should be ended by .pdb.
   
GPU REQUIRED!!!
   ```bash
   python ./evaluate_complex.py -c ./example/raw_pdb -w ./example/work/ -r ./example/result
   ```

4. The ranking results. ranking_res.csv is stored in result folder. It is sorted.