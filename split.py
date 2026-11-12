import os
import random
import subprocess

import pandas as pd

DATA_DIR = "data/processed"
SEED = 42
random.seed(SEED)


def run_mmseqs():
    fasta = DATA_DIR + "/proteins_hq.fasta"
    result = DATA_DIR + "/clusters"
    tmp = DATA_DIR + "/tmp"
    if not os.path.exists(result + "_cluster.tsv"):
        command = ["mmseqs", "easy-cluster", fasta, result, tmp, "--min-seq-id", "0.5", "-c", "0.8", "--cov-mode", "0"]
        subprocess.run(command, check=True)
    clusters = pd.read_csv(result + "_cluster.tsv", sep="\t", header=None)
    clusters.columns = ["cluster", "accession"]
    return clusters


def split_clusters(clusters):
    n_proteins = len(clusters)
    target = {"train": 0.8 * n_proteins, "eval": 0.1 * n_proteins, "test": 0.1 * n_proteins}
    current = {"train": 0, "eval": 0, "test": 0}
    sizes = clusters["cluster"].value_counts().to_dict()
    cluster_ids = list(sizes.keys())
    random.shuffle(cluster_ids)
    cluster_ids.sort(key=lambda c: sizes[c], reverse=True)
    cluster_to_split = {}
    for c in cluster_ids:
        best_split = None
        biggest_missing = None
        for split in ["train", "eval", "test"]:
            missing = target[split] - current[split]
            if biggest_missing is None or missing > biggest_missing:
                biggest_missing = missing
                best_split = split
        cluster_to_split[c] = best_split
        current[best_split] += sizes[c]
    clusters["split"] = clusters["cluster"].map(cluster_to_split)
    return clusters


def main():
    clusters = run_mmseqs()
    clusters = split_clusters(clusters)
    master = pd.read_csv(DATA_DIR + "/master.tsv", sep="\t")
    df = master.merge(clusters, on="accession")
    for split in ["train", "eval", "test"]:
        part = df[df["split"] == split]
        part.to_csv(DATA_DIR + "/" + split + ".tsv", sep="\t", index=False)
        print(split, len(part), "proteins", round(100 * len(part) / len(df), 1), "%")
    print("Number of clusters:", clusters["cluster"].nunique())
    print(pd.crosstab(df["taxon"], df["split"]))


if __name__ == "__main__":
    main()