# LATTE2GO: Layer-stacking Attention for protein function prediction on Gene Ontology
Protein function prediction by incorporating knowledge graph representation of heterogeneous interactions and Gene Ontology

# Preprequisites
Please read the following instructions carefully before running the code.

## Install dependencies
```bash
conda install --file requirements.txt
```
## Build dataset
Run the following commands to download necessary files to `data` folder
```bash
python download_data.py
```

# Run LATTE2GO experiments
<details><summary>Parameters for run/train.py</summary> 
```yaml
dataset:
  values: [ "MULTISPECIES" ]
pred_ntypes:
  values: [ 'molecular_function', 'biological_process', 'cellular_component' ]

method:
  values: [ "LATTE2GO-1", "LATTE-1", "LATTE2GO-2", "HGT", "DeepGraphGO", "MLP", "DeepGOZero", 'RGCN' ]

inductive:
  values: [ false ]
seed:
  values: [ 1 ]
```
</details>

```bash
python run/train.py --method LATTE2GO-1 --dataset MULTISPECIES --pred_ntypes molecular_function --seed 1
```

# Citation
If you use this code for your research, please cite our paper.