"""One-hot encoding of protein sequences.

A batch of B sequences becomes a float tensor of shape (B, N, D):
    N = 22 tokens: 20 standard amino acids, X (unknown / non-standard residue) and PAD
    D = maximum sequence length; shorter sequences are padded, longer ones are truncated (the first D residues are kept)
and a bool mask of shape (B, D) that is True for residues and False for padding.
"""
import numpy as np
import torch

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
UNKNOWN_INDEX = len(AMINO_ACIDS)  # 20: X and every other character (B, Z, J, U, O, *, -, ...)
PAD_INDEX = UNKNOWN_INDEX + 1  # 21
TOKENS = list(AMINO_ACIDS) + ["X", "<PAD>"]
ALPHABET_SIZE = len(TOKENS)  # N = 22
MAX_LEN = 1024  # D

# Byte value -> token index; lowercase letters are treated like uppercase
_LOOKUP = np.full(256, UNKNOWN_INDEX, dtype=np.int64)
for _index, _aa in enumerate(AMINO_ACIDS):
    _LOOKUP[ord(_aa)] = _LOOKUP[ord(_aa.lower())] = _index


def tokenize(sequence, max_len=MAX_LEN):
    """Token indices of the first max_len residues of a sequence."""
    raw = np.frombuffer(sequence[:max_len].encode("ascii", errors="replace"), dtype=np.uint8)
    return _LOOKUP[raw]


def encode(sequences, max_len=MAX_LEN):
    """One-hot encode a batch of sequences.

    Returns x of shape (B, ALPHABET_SIZE, max_len), float32, and mask of shape (B, max_len), bool (True = residue).
    Padding positions are one-hot on the PAD token.
    """
    indices = np.full((len(sequences), max_len), PAD_INDEX, dtype=np.int64)
    for row, sequence in enumerate(sequences):
        tokens = tokenize(sequence, max_len)
        indices[row, :len(tokens)] = tokens
    indices = torch.from_numpy(indices)
    x = torch.zeros(len(sequences), ALPHABET_SIZE, max_len)
    x.scatter_(1, indices.unsqueeze(1), 1.0)
    return x, indices != PAD_INDEX
