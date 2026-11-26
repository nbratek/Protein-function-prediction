"""Tests of the end-to-end skeleton: FASTA -> one-hot batches + mask -> one model per ontology -> CSV."""
import numpy as np
import pandas as pd
import pytest
import torch

import predict
from pfp.data import ONTOLOGIES, ProteinDataset, label_matrix, load_vocab, make_loader, read_fasta
from pfp.encoding import ALPHABET_SIZE, AMINO_ACIDS, MAX_LEN, PAD_INDEX, UNKNOWN_INDEX, encode
from pfp.models import DummyModel, build_model

FASTA = ">P1 first protein\nMKVL\nLA\n>P2 lowercase, non-standard residues\nmkxbzuo\n>P3 too long\n" + "A" * 1500 + "\n"
VOCABS = {"mf": ["GO:0000001", "GO:0000002"], "bp": ["GO:0000003", "GO:0000004", "GO:0000005"], "cc": ["GO:0000006"]}


@pytest.fixture
def fasta(tmp_path):
    path = tmp_path / "input.fasta"
    path.write_text(FASTA)
    return path


@pytest.fixture
def vocab_dir(tmp_path):
    for ontology, terms in VOCABS.items():
        pd.DataFrame({"go_id": terms, "n_train": 50}).to_csv(tmp_path / f"vocab_{ontology}.tsv", sep="\t", index=False)
    return tmp_path


def test_read_fasta(fasta):
    assert read_fasta(fasta) == [("P1", "MKVLLA"), ("P2", "MKXBZUO"), ("P3", "A" * 1500)]


def test_read_fasta_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "duplicates.fasta"
    path.write_text(">P1\nMK\n>P1\nLA\n")
    with pytest.raises(ValueError, match="Duplicate"):
        read_fasta(path)


def test_encode_pads_and_truncates():
    x, mask = encode(["MKV", "A" * 1500])
    assert x.shape == (2, ALPHABET_SIZE, MAX_LEN) == (2, 22, 1024)
    assert mask.shape == (2, MAX_LEN) and mask.dtype == torch.bool
    assert torch.all(x.sum(dim=1) == 1)  # exactly one token per position
    assert mask.sum(dim=1).tolist() == [3, MAX_LEN]
    assert torch.all(x[0, PAD_INDEX, 3:] == 1) and torch.all(x[0, PAD_INDEX, :3] == 0)
    assert x[0].argmax(dim=0)[:3].tolist() == [AMINO_ACIDS.index(aa) for aa in "MKV"]


def test_encode_unknown_and_lowercase_residues():
    x, _ = encode(["XBZUO*", "mkv"], max_len=8)
    assert torch.all(x[0, UNKNOWN_INDEX, :6] == 1)
    assert torch.equal(x[1], encode(["MKV"], max_len=8)[0][0])


def test_dummy_model_shapes():
    x, mask = encode(["MKV", "LA", "W"], max_len=16)
    model = DummyModel(n_terms=7)
    assert model(x, mask).shape == (3, 7)
    assert model(x).shape == (3, 7)  # the mask is optional
    with pytest.raises(ValueError):
        model(x.transpose(1, 2), mask)  # (B, D, N) instead of (B, N, D)


def test_model_outputs_match_vocabularies():
    # P of each ontology = size of its vocabulary built by vocab.py in data/processed
    x, mask = encode(["MKV"] * 4)
    for ontology in ONTOLOGIES:
        vocab = load_vocab(ontology)
        assert build_model("dummy", n_terms=len(vocab))(x, mask).shape == (4, len(vocab))


def test_label_matrix_drops_terms_outside_vocabulary():
    labels = label_matrix(["GO:1;GO:9", np.nan, "GO:2;GO:1"], ["GO:1", "GO:2"])
    assert labels.tolist() == [[True, False], [False, False], [True, True]]


def test_training_dataset_and_batches():
    df = pd.DataFrame({"accession": ["A", "B", "C"], "sequence": ["MKV", "LLLL", "W"],
                       "go_mf": ["GO:1;GO:rare", "GO:rare", np.nan]})
    dataset = ProteinDataset.from_split(df, "mf", ["GO:1", "GO:2"])
    assert dataset.ids == ["A"]  # B has only a rare term, C has no MF term
    batch = next(iter(make_loader(dataset, batch_size=2, max_len=8)))
    assert batch.x.shape == (1, ALPHABET_SIZE, 8) and batch.mask.shape == (1, 8)
    assert batch.labels.tolist() == [[1.0, 0.0]]


def test_predict_end_to_end(fasta, vocab_dir, tmp_path):
    args = [str(fasta), "--vocab-dir", str(vocab_dir), "--batch-size", "2"]
    predict.main(args + ["-o", str(tmp_path / "predictions.csv")])
    df = pd.read_csv(tmp_path / "predictions.csv")
    assert list(df.columns) == ["ProteinID", "GOterm", "Score"]
    all_terms = [term for ontology in ONTOLOGIES for term in VOCABS[ontology]]
    assert len(df) == 3 * len(all_terms)
    assert df.groupby("ProteinID", sort=False)["GOterm"].apply(list).to_dict() == {p: all_terms for p in ["P1", "P2", "P3"]}
    assert df["Score"].between(0, 1).all()
    # the same seed gives the same scores; .gz output is compressed
    predict.main(args + ["-o", str(tmp_path / "again.csv.gz")])
    pd.testing.assert_frame_equal(df, pd.read_csv(tmp_path / "again.csv.gz"))
