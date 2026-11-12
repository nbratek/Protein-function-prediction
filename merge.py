import glob
import gzip
import os
import pandas as pd

DATA_DIR = "data"
OUT_DIR = os.path.join("data", "processed")

ORGANISMS = {"Human": ("HUMAN-uniprot.gaf.gz", "uniprotkb_Human_*.tsv.gz"),
    "Danre": ("DANRE-uniprot.gaf.gz", "uniprotkb_danre_*.tsv.gz"),
    "Mouse": ("MOUSE-uniprot.gaf.gz", "uniprotkb_Mouse_*.tsv.gz"),}

HQ_EVIDENCE = {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "HTP", "HDA", "HMP", "HGI", "HEP", "TAS", "IC"}


def read_gaf(path):
    rows = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("!"):
                continue
            cols = line.rstrip("\n").split("\t")
            rows.append((cols[1], cols[3], cols[4], cols[6]))
    gaf = pd.DataFrame(rows, columns=["accession", "qualifier", "go_id", "evidence"])
    return gaf[~gaf["qualifier"].str.contains("NOT")]


def read_uniprot(path):
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    return pd.DataFrame({"accession": df["Entry"], "go_uniprot": df["Gene Ontology IDs"].apply(to_set), "sequence": df["Sequence"],})


def to_set(go_string):
    return {g.strip() for g in go_string.split(";") if g.strip()}


def go_per_protein(gaf):
    return gaf.groupby("accession")["go_id"].agg(set).to_dict()


def compare_go(df, gaf_go):
    # Compare GO IDs per protein coming from the GAF and from the Uniprot.
    gaf_sets = df["accession"].map(lambda a: gaf_go.get(a, set()))
    return pd.DataFrame({"accession": df["accession"], "taxon": df["taxon"],
        "only_in_uniprot": [";".join(sorted(u - g)) for u, g in zip(df["go_uniprot"], gaf_sets)],
        "only_in_gaf": [";".join(sorted(g - u)) for u, g in zip(df["go_uniprot"], gaf_sets)],})


def keep_hq_terms(df, gaf):
    # Select only high quality Evidence Codes
    hq_go = go_per_protein(gaf[gaf["evidence"].isin(HQ_EVIDENCE)])
    df = df.copy()
    df["go_terms"] = [u & hq_go.get(a, set()) for a, u in zip(df["accession"], df["go_uniprot"])]
    return df[df["go_terms"].str.len() > 0]


def process_organism(taxon, gaf_file, tsv_pattern):
    gaf = read_gaf(os.path.join(DATA_DIR, gaf_file))
    df = read_uniprot(glob.glob(os.path.join(DATA_DIR, tsv_pattern))[0])
    df["taxon"] = taxon
    df = df[df["accession"].isin(gaf["accession"])]
    comparison = compare_go(df, go_per_protein(gaf))
    df = keep_hq_terms(df, gaf)
    n_diff = (comparison["only_in_uniprot"] != "") | (comparison["only_in_gaf"] != "")
    print(f"{taxon}: {n_diff.sum()}/{len(comparison)} proteins with GAF vs UniProt differences, "
      f"{len(df)} proteins after evidence code filter")
    return df, comparison


def write_fasta(df, path):
    # fasta file 
    with open(path, "w") as fh:
        for acc, taxon, seq in zip(df["accession"], df["taxon"], df["sequence"]):
            fh.write(f">{acc} taxon={taxon}\n{seq}\n")


def save(master, comparison):
    os.makedirs(OUT_DIR, exist_ok=True)
    out = master[["accession", "taxon", "go_terms", "sequence"]].copy()
    out["go_terms"] = out["go_terms"].apply(lambda s: ";".join(sorted(s)))
    out.to_csv(os.path.join(OUT_DIR, "master.tsv"), sep="\t", index=False)
    comparison.to_csv(os.path.join(OUT_DIR, "go_comparison.tsv"), sep="\t", index=False)
    write_fasta(out, os.path.join(OUT_DIR, "proteins_hq.fasta"))


def main():
    results = [process_organism(t, gaf, tsv) for t, (gaf, tsv) in ORGANISMS.items()]
    master = pd.concat([r[0] for r in results], ignore_index=True)
    comparison = pd.concat([r[1] for r in results], ignore_index=True)
    save(master, comparison)


if __name__ == "__main__":
    main()