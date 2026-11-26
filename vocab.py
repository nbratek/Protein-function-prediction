import argparse
import os

import pandas as pd

DATA_DIR = os.path.join("data", "processed")
ONTOLOGIES = ["mf", "bp", "cc"]
# Default M: a term enters the label vocabulary if at least this many TRAIN proteins have it (DeepGOPlus used 50).
# Rarer terms are removed from the training and evaluation labels only; the test annotations are not filtered.
MIN_TRAIN_PROTEINS = 50


def count_terms(df, column):
    terms = df[column].fillna("").str.split(";").explode()
    return terms[terms != ""].value_counts()


def main():
    parser = argparse.ArgumentParser(description="Build the label vocabulary of each ontology from the train split")
    parser.add_argument("--min-count", type=int, default=MIN_TRAIN_PROTEINS,
                        help="M: minimum number of train proteins annotated with a term (default: %(default)s)")
    min_count = parser.parse_args().min_count
    splits = {s: pd.read_csv(os.path.join(DATA_DIR, s + ".tsv"), sep="\t") for s in ["train", "eval", "test"]}
    for ont in ONTOLOGIES:
        column = "go_" + ont
        counts = count_terms(splits["train"], column)
        vocab = counts[counts >= min_count].sort_index().sort_values(ascending=False, kind="stable")
        vocab.rename_axis("go_id").reset_index(name="n_train").to_csv(
            os.path.join(DATA_DIR, "vocab_" + ont + ".tsv"), sep="\t", index=False)
        line = f"{ont.upper()}: {len(vocab)} terms with >= {min_count} train proteins (of {len(counts)} in train)"
        for split, df in splits.items():
            labelled = df[column].fillna("") != ""
            line += f" | {split}: {labelled.sum()} proteins with labels"
            if split != "train":
                split_counts = count_terms(df, column)
                covered = split_counts[split_counts.index.isin(vocab.index)].sum()
                line += f", {100 * covered / split_counts.sum():.1f}% of annotations in vocab"
        print(line)


if __name__ == "__main__":
    main()
