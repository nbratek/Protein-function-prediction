# Protein function prediction

Predicting Gene Ontology (GO) terms from protein sequences of human, mouse and zebrafish.

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
   Each split is saved as TSV (annotations) and FASTA (sequences, the predictor input).
3. `vocab.py` builds the label vocabulary of each ontology from the **train split only** (terms with at least M = 50 train proteins, change with `--min-count`).

Train and evaluate each ontology only on proteins that have at least one label in it.

## How to run

Requires [uv](https://docs.astral.sh/uv/). MMseqs2 is only needed to recompute `clusters_cluster.tsv` (`brew install mmseqs2`).

```bash
uv sync
uv run merge.py
uv run split.py
uv run vocab.py
uv run predict.py data/processed/test.fasta -o outputs/test_predictions.csv
uv run pytest
```

| File | Content |
|---|---|
| `master.tsv` | accession, entry_name, taxon, go_terms (direct annotations), go_mf / go_bp / go_cc (propagated), sequence |
| `go_comparison.tsv` | GO terms only in UniProt / only in GAF |
| `proteins_hq.fasta` | sequences after the evidence code filter |
| `clusters_cluster.tsv` | MMseqs2 output (cluster representative, protein) |
| `train.tsv`, `eval.tsv`, `test.tsv` | final split |
| `train.fasta`, `eval.fasta`, `test.fasta` | sequences of each split |
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

## Predictor (end-to-end skeleton)

`predict.py` reads a multi-protein FASTA file and writes a score for every protein and every GO term of the three vocabularies.
The model is a placeholder for now (random scores). Input, encoding, batching and output are meant to stay when a real model is plugged in.

```bash
uv run predict.py data/processed/test.fasta -o outputs/test_predictions.csv
```

```
FASTA --skbio--> (ID, sequence) --encode--> x (B, 22, 1024) + mask (B, 1024) --+--> MF model --> (B, 370)
                                                                               +--> BP model --> (B, 1993)  --sigmoid--> CSV
                                                                               +--> CC model --> (B, 365)
```

- **Input**: FASTA parsed with `skbio.io.read(..., constructor=skbio.Protein)`. The protein ID is the header up to the first space.
- **Encoding**: one-hot tensor `(B, N, D)` with N = 22 tokens (20 standard amino acids, `X` for unknown and non-standard residues, `PAD`) and D = 1024 (`--max-len`).
  Shorter sequences are padded, longer ones keep their first D residues. The bool mask `(B, D)` is `True` for residues and `False` for padding.
- **Models**: one per ontology, `forward(x, mask)` returns logits `(B, P)`, where P is the size of the ontology vocabulary read from `vocab_*.tsv` (MF 370, BP 1993, CC 365 for M = 50).
  `DummyModel` ignores the sequence and returns random logits. The predictor applies a sigmoid, so a real model can be trained with `BCEWithLogitsLoss`.
- **Output**: CSV `ProteinID,GOterm,Score`, one row per protein and GO term (2,728 rows per protein, scores rounded to 4 decimals).
  The test split gives 8.1 M rows (~200 MB); an output path ending with `.gz` compresses the file.
- **Options**: `--model`, `--vocab-dir`, `--batch-size` (32), `--max-len` (1024), `--seed` (42, fixes the random scores), `--device` (`cpu`, `cuda`, `mps`).

Rare terms: the vocabulary only has terms annotated to at least M train proteins, so P depends on M.
For training and evaluation, `ProteinDataset.from_split` drops the rarer terms from the labels and skips proteins left without any label (MF: 40 train and 9 eval proteins).
The test split is not filtered and keeps all its GO terms as ground truth.

| Module | Content |
|---|---|
| `pfp/encoding.py` | amino acid alphabet, one-hot encoding with padding, truncation and mask |
| `pfp/data.py` | FASTA reading, vocabularies, label matrices, `ProteinDataset`, batches (`make_loader`) |
| `pfp/models.py` | model interface (`OntologyModel`), `DummyModel`, model registry `MODELS` |
| `pfp/predictor.py` | runs the three models batch by batch and writes the CSV |
| `predict.py` | command line entry point |
| `tests/test_pipeline.py` | tests of every step (`uv run pytest`) |

To add an architecture, subclass `OntologyModel` in `pfp/models.py`, implement `forward(x, mask)` and add the class to `MODELS`. It is then available as `--model <name>`.
