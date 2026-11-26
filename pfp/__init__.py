"""Protein function prediction: GO term scores (MF, BP, CC) from amino acid sequences.

Modules:
    encoding   amino acid alphabet and one-hot encoding of batches, (B, N, D) + mask (B, D)
    data       FASTA reading (scikit-bio), GO vocabularies, label matrices, Dataset and batching
    models     per-ontology model interface, dummy model and model registry
    predictor  runs one model per ontology on a FASTA file and writes the scores to CSV
"""
