"""Predict GO term scores (MF, BP, CC) for every protein of a FASTA file and save them as CSV.

Example: uv run predict.py data/processed/test.fasta -o outputs/test_predictions.csv
"""
import argparse

import torch

from pfp.data import BATCH_SIZE, DATA_DIR, ONTOLOGIES, load_vocab, read_fasta
from pfp.encoding import MAX_LEN
from pfp.models import MODELS, build_model
from pfp.predictor import Predictor, write_predictions

SEED = 42


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("fasta", help="input FASTA file with one or more protein sequences")
    parser.add_argument("-o", "--output", default="outputs/predictions.csv",
                        help="output CSV with columns ProteinID,GOterm,Score (a .gz suffix compresses it)")
    parser.add_argument("--model", choices=list(MODELS), default="dummy", help="model architecture")
    parser.add_argument("--vocab-dir", default=DATA_DIR, help="directory with vocab_mf.tsv, vocab_bp.tsv, vocab_cc.tsv")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-len", type=int, default=MAX_LEN, help="shorter sequences are padded, longer truncated")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cpu", help="cpu, cuda or mps")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    vocabs = {ontology: load_vocab(ontology, args.vocab_dir) for ontology in ONTOLOGIES}
    models = {ontology: build_model(args.model, n_terms=len(vocab)) for ontology, vocab in vocabs.items()}
    records = read_fasta(args.fasta)
    n_truncated = sum(len(sequence) > args.max_len for _, sequence in records)
    print(f"{len(records)} proteins read from {args.fasta}, {n_truncated} truncated to {args.max_len} residues")
    print(f"Model {args.model!r}, GO terms per ontology: " + ", ".join(f"{o.upper()} {len(v)}" for o, v in vocabs.items()))
    predictor = Predictor(models, vocabs, max_len=args.max_len, batch_size=args.batch_size, device=args.device)
    n_rows = write_predictions(args.output, predictor.predict(records), vocabs)
    print(f"{n_rows} rows (protein, GO term, score) written to {args.output}")


if __name__ == "__main__":
    main()
