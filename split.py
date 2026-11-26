import os
import random
import shutil
import subprocess

import pandas as pd

from merge import write_fasta

DATA_DIR = "data/processed"
SEED = 42
FRACTIONS = {"train": 0.8, "eval": 0.1, "test": 0.1}


def run_mmseqs():
    fasta = DATA_DIR + "/proteins_hq.fasta"
    result = DATA_DIR + "/clusters"
    tmp = DATA_DIR + "/tmp"
    if not os.path.exists(result + "_cluster.tsv"):
        if shutil.which("mmseqs") is None:
            raise SystemExit("MMseqs2 not found. Install it (e.g. `brew install mmseqs2`) to recompute the clusters.")
        command = ["mmseqs", "easy-cluster", fasta, result, tmp, "--min-seq-id", "0.5", "-c", "0.8", "--cov-mode", "0"]
        subprocess.run(command, check=True)
    clusters = pd.read_csv(result + "_cluster.tsv", sep="\t", header=None)
    clusters.columns = ["cluster", "accession"]
    return clusters


def split_clusters(clusters):
    # Whole clusters go to one split. Clusters are visited in random order (not sorted by size)
    # and each goes to the split furthest below its target share, so every split gets
    # a similar mix of large and small clusters.
    n_proteins = len(clusters)
    target = {split: fraction * n_proteins for split, fraction in FRACTIONS.items()}
    current = {split: 0 for split in FRACTIONS}
    sizes = clusters["cluster"].value_counts().to_dict()
    cluster_ids = sorted(sizes)  # fixed order before shuffling, so the split is reproducible
    random.Random(SEED).shuffle(cluster_ids)
    cluster_to_split = {}
    for c in cluster_ids:
        best_split = max(target, key=lambda s: (target[s] - current[s]) / target[s])
        cluster_to_split[c] = best_split
        current[best_split] += sizes[c]
    clusters["split"] = clusters["cluster"].map(cluster_to_split)
    return clusters


def main():
    clusters = run_mmseqs()
    clusters = split_clusters(clusters)
    master = pd.read_csv(DATA_DIR + "/master.tsv", sep="\t")
    df = master.merge(clusters, on="accession")
    for split in FRACTIONS:
        part = df[df["split"] == split]
        part.to_csv(DATA_DIR + "/" + split + ".tsv", sep="\t", index=False)
        write_fasta(part, DATA_DIR + "/" + split + ".fasta")  # predictor input
        cluster_sizes = part.groupby("cluster").size()
        print(f"{split}: {len(part)} proteins ({100 * len(part) / len(df):.1f}%), {len(cluster_sizes)} clusters, "
              f"largest cluster {cluster_sizes.max()}, singletons {100 * (cluster_sizes == 1).sum() / len(part):.1f}% of proteins")
    print("Number of clusters:", clusters["cluster"].nunique())
    print(pd.crosstab(df["taxon"], df["split"]))


if __name__ == "__main__":
    main()
