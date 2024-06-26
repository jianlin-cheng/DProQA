# DProQA: A Gated-Graph Transformer for Protein Complex Structure Assessment

## ISMB2023

Our work has been accepted by ISMB 2023, [here](https://academic.oup.com/bioinformatics/article/39/Supplement_1/i308/7210460) is the latest manuscript.

## CASP15 result

In terms of TMscore ranking loss, DProQA (team: MULTICOM_egnn) ranked 9th among all 26 methods and **3rd** place among all single-model methods.
![tm_loss.png](./images/CASP15.png)

------------------------
DProQA is a Gated-Graph Transformer model for  end-to-end protein complex structure's quality evaluation. DProQA achieves significant speed-ups and better quality compared to the current baseline method. If you have any questions or suggestions, please contact us at  <xcbh6@umsystem.edu>. We are happy to help!

![pipeline.png](./images/pipeline.png)

![gated_graph_transformer.png](./images/GGT_V4.png)

# Citation

If you think our work is helpful, please cite our work by:

```
@article{10.1093/bioinformatics/btad203,
    author = {Chen, Xiao and Morehead, Alex and Liu, Jian and Cheng, Jianlin},
    title = "{A gated graph transformer for protein complex structure quality assessment and its performance in CASP15}",
    journal = {Bioinformatics},
    volume = {39},
    number = {Supplement_1},
    pages = {i308-i317},
    year = {2023},
    month = {06},
    issn = {1367-4811},
    doi = {10.1093/bioinformatics/btad203},
    url = {https://doi.org/10.1093/bioinformatics/btad203},
    eprint = {https://academic.oup.com/bioinformatics/article-pdf/39/Supplement\_1/i308/50741583/btad203.pdf},
}
```
# Installation

1. Download this repository
   
   ```bash
   git clone https://github.com/jianlin-cheng/DProQA.git
   ```

2. Set up conda environment locally
   
   ```bash
   cd DProQA
   conda env create --name DProQA -f environment.yml
   ```

3. Activate conda environment
   
   ```bash
   conda activate DProQA
   ```
   
# Dataset
## Benchmark sets

We provide our benchmark tests HAF2 and DBM55-AF2 for download by:

```bash
wget https://zenodo.org/record/6569837/files/DproQ_benchmark.tgz
```

Each dataset contains:

1. `decoy` folder: decoys files
2. `native` folder: native structure files
3. `label_info.csv`: DockQA scores and CAPRI class label

## Build own training data
Due to the larger data size, we are not able to provide direct download. But we provided our training, validation targets information in `Data/data_list` folder. You could to use any protein complex structure prediction tool to generate predicted structures (pdb format) first, then use our data generation script to generate protein graph (.dgl) for training. 

```bash
customer_data_folder
├── pdbcode1.pdb
├── pdbcode2.pdb
├── pdbcode3.pdb
├── pdbcode4.pdb
└── pdbcode5.pdb
```

```bash
python ./src/data_generator.py
--input_pdb_folder -p input pdbs folder
--dgl_save_folder -d dgl files save folder
--cores  -c number of cores for parallel processing

# example code
python ./src/data_generator.py -p /example_pdbs_folder/ -d /dgl_save_folder -c 10
```

# Usage

Here is the inference.py script parameters' introduction.

```bash
python inference.py
-c --complex_folder     Raw protein complex complex_folder
-w --work_dir           Working directory to save all intermedia files and folders, it will be created if it is not exit
-r --result_folder      Result folder to save two ranking results, it will be created if it is not exit
-t --threads            Number of threads for parallel feature generation and dataloader, default=10
-s --delete_tmp         Set False to save work_dir and intermedia files, otherwise set True, default=False
```

# Use provided model weights to predict protein complex structures' quality

**DProQA requires GPU**. We provide a few protein complexes in `example` folder for testing. The evaluation result Ranking.csv is stored in result_folder.

```bash
python ./inference.py -c ./examples/6AL0/ -w ./examples/work/ -r ./examples/result
```

You can build your own dataset for evaluation, the data folder should look like this:

```bash
customer_data_folder
├── decoy_1.pdb
├── decoy_2.pdb
├── decoy_3.pdb
├── decoy_4.pdb
└── decoy_5.pdb
```
