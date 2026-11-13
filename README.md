# Protein function prediction

Dataset for predicting Gene Ontology (GO) terms from protein sequences of human, mouse and zebrafish.

## Data

- **GAF (GOA, UniProt-centric)**: `HUMAN`, `DANRE`, `MOUSE`, built with GO release 2026-07-26
- **UniProt Swiss-Prot** (`reviewed` only), columns: Entry, Gene Ontology IDs, Sequence, Entry Name.
  The exports come from free-text queries, so `merge.py` keeps only entries whose Entry Name ends with the species mnemonic (e.g. `OPSD_HUMAN`).
  For a new download use `organism_id:9606`, `organism_id:10090` and `organism_id:7955` together with `reviewed:true`.
- **GO ontology** (`go-basic.obo`, release 2026-11-13): downloaded by `merge.py` on the first run to `data/go-basic-2026-11-13.obo.gz`.
  The GAF files were built with GO 2026-07-26, which is not available for download. Terms obsoleted since then are mapped through `alt_id` / `replaced_by` or dropped (`merge.py` prints how many).

## Pipeline

1. `merge.py`
   - keeps Swiss-Prot proteins of the given species that are present in the GAF, drops `NOT` annotations
   - keeps a GO term only if it is in UniProt and has at least one experimental (`EXP`, `IDA`, `IPI`, `IMP`, `IGI`, `IEP`), high-throughput (`HTP`, `HDA`, `HMP`, `HGI`, `HEP`), `TAS` or `IC` annotation in the GAF
   - removes `GO:0005515` (protein binding) as a direct annotation. It can still appear through propagation from more specific binding terms
   - propagates the terms to all their ancestors over `is_a` and `part_of` (true path rule) and splits them into MF, BP and CC, without the three root terms
2. `split.py` clusters the sequences with MMseqs2 (≥ 50% identity, ≥ 80% coverage of both sequences) and puts whole clusters into train, eval or test (80/10/10).
   Clusters are visited in random order and each goes to the split that is furthest below its target, so all splits get a similar mix of cluster sizes.
3. `vocab.py` builds the label vocabulary of each ontology from the **train split only** (terms with at least 50 train proteins).

Train and evaluate each ontology only on proteins that have at least one label in it.

## How to run

Requires [uv](https://docs.astral.sh/uv/). MMseqs2 is only needed to recompute `clusters_cluster.tsv` (`brew install mmseqs2`).

```bash
uv sync
uv run merge.py
uv run split.py
uv run vocab.py
```

| File | Content |
|---|---|
| `master.tsv` | accession, entry_name, taxon, go_terms (direct annotations), go_mf / go_bp / go_cc (propagated), sequence |
| `go_comparison.tsv` | GO terms only in UniProt / only in GAF |
| `proteins_hq.fasta` | sequences after the evidence code filter |
| `clusters_cluster.tsv` | MMseqs2 output (cluster representative, protein) |
| `train.tsv`, `eval.tsv`, `test.tsv` | final split |
| `vocab_mf.tsv`, `vocab_bp.tsv`, `vocab_cc.tsv` | label vocabulary (go_id, number of train proteins) |

## Results

Number of clusters: 14164

| Organism | train | eval | test |
|---|---|---|---|
| Human | 12,571 | 1,571 | 1,579 |
| Danre | 1,129 | 141 | 134 |
| Mouse | 10,041 | 1,258 | 1,256 |
| Total | 23,741 (80%) | 2,970 (10%) | 2,969 (10%) |

Singletons make up 15-16% of the proteins in every split.
