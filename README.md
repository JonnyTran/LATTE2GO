# LATTE2GO: Layer-stacking Attention for protein function prediction on Gene Ontology

In this repository, you will be able to reproduce the results of the paper 
"Protein function prediction by incorporating knowledge graph representation of heterogeneous interactions and Gene Ontology."
In addition, you will be able to run the ablation experiments to study the effect of various heterogeneous dataset inputs 
and parameters on the LATTE2GO model performance.

If there is any problem with the code, please open an issue or contact me at `nhat.tran@mavs.uta.edu`.

# Prerequisites
To run the experiments, you need at least 5.5GB of disk space, 50GB of RAM memory, and at least 10GB of GPU RAM.

Please install package requirements, download the dataset, and run the bash commands provided by following 
instructions.

## Install dependencies
Please
```bash
conda install --file requirements.txt
```
or alternatively:
```bash
pip install -r requirements.txt
```

## Download the dataset
The MULTISPECIES and HUMAN_MOUSE dataset for AFP is downloadable from AWS S3.

Run the following commands to download necessary files to `data/` directory:
```bash
python download_data.py
```

# Run experiments

## AFP experiments
Run the following commands to train and evaluate the model on the DeepGraphGO multi-species AFP dataset:

<details><summary>Parameters for `experiments/run.py`</summary> 

```yml
dataset:
  values: [ "MULTISPECIES", "HUMAN_MOUSE" ]
pred_ntypes:
  values: [ 'molecular_function', 'biological_process', 'cellular_component', 'molecular_function biological_process cellular_component' ]
method:
  values: [ "LATTE2GO-1", "LATTE-1", "LATTE2GO-2", "HGT", "DeepGraphGO", "MLP", "DeepGOZero", "RGCN" ]
inductive:
  values: [ false ]
seed:
  values: [ 1 ]
```
</details>

```bash
python experiments/run.py --method LATTE2GO-2 --dataset MULTISPECIES --pred_ntypes molecular_function --seed 1
```

## LATTE2GO ablation experiments
To run the ablation experiments with various combination of the heterogeneous RNA-protein interactions dataset or 
LATTE2GO hyperparameters, modify the `experiments/configs/latte2go.yaml` file with these parameters.

<details><summary>Parameters for `experiments/configs/latte2go.yaml`</summary> 

```yml
ntype_subset:
  values:
    - 'Protein MessengerRNA MicroRNA LncRNA biological_process cellular_component molecular_function'
    - 'Protein MessengerRNA MicroRNA LncRNA'
    - 'Protein MessengerRNA MicroRNA'
    - 'Protein MessengerRNA'
    - 'Protein'
    - ''
go_etypes:
  values:
    - 'is_a part_of has_part regulates negatively_regulates positively_regulates'
    - 'is_a part_of has_part'
    - 'is_a'
    - null
```
</details>

```bash
python experiments/run.py --method LATTE2GO-2 --config experiments/configs/latte2go.yaml --dataset MULTISPECIES --pred_ntypes molecular_function
```

# Citation
If you use this code for your research, please cite our paper.