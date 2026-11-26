"""Input data: FASTA files, GO vocabularies, label matrices and batches for PyTorch."""
from collections import Counter
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import skbio
import torch
from torch.utils.data import DataLoader, Dataset

from pfp.encoding import MAX_LEN, encode

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
ONTOLOGIES = ("mf", "bp", "cc")
BATCH_SIZE = 32


def read_fasta(path):
    """Read a multi-protein FASTA file with scikit-bio and return a list of (protein ID, sequence).

    The ID is the header up to the first whitespace. Lowercase letters are converted to uppercase,
    characters outside the IUPAC protein alphabet raise ValueError.
    """
    sequences = skbio.io.read(str(path), format="fasta", constructor=skbio.Protein, lowercase=True)
    records = [(sequence.metadata["id"], str(sequence)) for sequence in sequences]
    if not records:
        raise ValueError(f"No sequences found in {path}")
    duplicates = [protein_id for protein_id, count in Counter(p for p, _ in records).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate protein IDs in {path}: {', '.join(duplicates[:5])}")
    return records


def load_vocab(ontology, data_dir=DATA_DIR):
    """GO terms of the label vocabulary of one ontology (vocab_<ontology>.tsv from vocab.py), in model output order."""
    return pd.read_csv(Path(data_dir) / f"vocab_{ontology}.tsv", sep="\t")["go_id"].tolist()


def load_split(split, data_dir=DATA_DIR):
    """train / eval / test table from split.py (accession, taxon, go_mf, go_bp, go_cc, sequence, ...)."""
    return pd.read_csv(Path(data_dir) / f"{split}.tsv", sep="\t")


def label_matrix(annotations, vocab):
    """Multi-hot bool matrix (n_proteins, len(vocab)) from ";"-separated GO terms per protein.

    Terms outside the vocabulary are dropped. The vocabulary only has terms with at least M train annotations,
    so this is where rare terms leave the training and evaluation labels.
    """
    column = {term: j for j, term in enumerate(vocab)}
    annotations = list(annotations)
    labels = np.zeros((len(annotations), len(vocab)), dtype=bool)
    for row, terms in enumerate(annotations):
        if isinstance(terms, str):  # missing value = no terms in this ontology
            labels[row, [column[term] for term in terms.split(";") if term in column]] = True
    return labels


class ProteinDataset(Dataset):
    """Protein sequences with an optional label matrix of one ontology. Items are (ID, sequence, labels or None)."""

    def __init__(self, ids, sequences, labels=None):
        self.ids = list(ids)
        self.sequences = list(sequences)
        self.labels = labels
        if len(self.sequences) != len(self.ids) or (labels is not None and len(labels) != len(self.ids)):
            raise ValueError("ids, sequences and labels must have the same length")

    @classmethod
    def from_split(cls, df, ontology, vocab):
        """Training or evaluation set of one ontology.

        Labels are restricted to the vocabulary (terms with at least M train annotations) and proteins left
        without any label are skipped. Not for the test set: its annotations stay unfiltered.
        """
        labels = label_matrix(df["go_" + ontology], vocab)
        keep = labels.any(axis=1)
        return cls(df["accession"].to_numpy()[keep], df["sequence"].to_numpy()[keep], labels[keep])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        labels = None if self.labels is None else self.labels[index]
        return self.ids[index], self.sequences[index], labels


@dataclass
class Batch:
    ids: list  # B protein IDs
    x: torch.Tensor  # (B, N, D) one-hot encoded sequences
    mask: torch.Tensor  # (B, D) True for residues, False for padding
    labels: torch.Tensor | None = None  # (B, P) multi-hot, only for training and evaluation

    def to(self, device):
        labels = None if self.labels is None else self.labels.to(device)
        return Batch(self.ids, self.x.to(device), self.mask.to(device), labels)


def collate(items, max_len=MAX_LEN):
    """Turn a list of ProteinDataset items into a Batch; sequences are encoded here, one batch at a time."""
    ids, sequences, labels = zip(*items)
    x, mask = encode(sequences, max_len)
    labels = None if labels[0] is None else torch.from_numpy(np.stack(labels)).float()
    return Batch(list(ids), x, mask, labels)


def make_loader(dataset, batch_size=BATCH_SIZE, max_len=MAX_LEN, shuffle=False, seed=42):
    """DataLoader yielding Batch objects. The seed fixes the order when shuffle=True."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=partial(collate, max_len=max_len),
                      generator=torch.Generator().manual_seed(seed))
