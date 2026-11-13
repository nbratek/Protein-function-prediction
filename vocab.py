import os

import pandas as pd

DATA_DIR = os.path.join("data", "processed")
ONTOLOGIES = ["mf", "bp", "cc"]
# A term enters the label vocabulary if at least this many TRAIN proteins have it (DeepGOPlus used 50)
MIN_TRAIN_PROTEINS = 50


def count_terms(df, column):
    terms = df[column].fillna("").str.split(";").explode()
    return terms[terms != ""].value_counts()


def main():
    splits = {s: pd.read_csv(os.path.join(DATA_DIR, s + ".tsv"), sep="\t") for s in ["train", "eval", "test"]}
    for ont in ONTOLOGIES:
        column = "go_" + ont
        counts = count_terms(splits["train"], column)
        vocab = counts[counts >= MIN_TRAIN_PROTEINS].sort_index().sort_values(ascending=False, kind="stable")
        vocab.rename_axis("go_id").reset_index(name="n_train").to_csv(
            os.path.join(DATA_DIR, "vocab_" + ont + ".tsv"), sep="\t", index=False)
        line = f"{ont.upper()}: {len(vocab)} terms with >= {MIN_TRAIN_PROTEINS} train proteins (of {len(counts)} in train)"
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
