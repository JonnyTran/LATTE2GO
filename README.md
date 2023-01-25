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
```bash
python run/train.py --config_file config/latte2go.yaml
```

# Citation
If you use this code for your research, please cite our paper.