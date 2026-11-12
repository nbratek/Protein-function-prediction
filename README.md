# Protein function prediction

 
## Data

 
- **GAF (GOA, UniProt-centric)**: `HUMAN`, `DANRE`, `MOUSE` 
- **UniProt Swiss-Prot** (`reviewed` only), columns: Entry, Gene Ontology IDs, Sequence

 
## How to run
 
 
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
 
python3 merge.py   
python3 split.py 
```
 
 
| File | Content |
|---|---|
| `master.tsv` | accession, taxon, go_terms, sequence |
| `go_comparison.tsv` | GO terms only in UniProt / only in GAF |
| `proteins_hq.fasta` | sequences after the evidence code filter |
| `clusters_cluster.tsv` | MMseqs2 output (cluster representative, protein) |
| `train.tsv`, `eval.tsv`, `test.tsv` | final split |
 
## Results

 
Number of clusters: 14164
 
| Organism | train | eval | test |
|---|---|---|---|
| Human | 12,096 | 1,823 | 1,802 |
| Danre | 1,185 | 105 | 114 |
| Mouse | 10,463 | 1,040 | 1,052 |
| Total | 23,744 (80%) | 2,968 (10%) | 2,968 (10%) |
 