# LATTE2GO: Layer-stacking Attention for protein function prediction on Gene Ontology
Protein function prediction by incorporating knowledge graph representation of heterogeneous interactions and Gene Ontology

# Preprequisites
Please execute the following instructions before running the code.

## Install dependencies
```bash
conda install --file requirements.txt
```
## Build dataset
Run the following commands to download necessary files to `data` folder
```bash
python download_data.py
```

# Run experiments

## AFP experiments
Run the following commands to train and evaluate the model on the DeepGraphGO multi-species AFP dataset:

<details><summary>Parameters for `run/train.py`</summary> 

```yml
dataset:
  values: [ "MULTISPECIES" ]
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
python experimentrs/run.py --method LATTE2GO-1 --dataset MULTISPECIES --pred_ntypes molecular_function --seed 1
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
python experiments/run.py --method LATTE2GO-1 --config experiments/configs/latte2go.yaml --dataset MULTISPECIES --pred_ntypes molecular_function
```

# Citation
If you use this code for your research, please cite our paper.