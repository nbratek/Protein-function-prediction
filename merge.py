import glob
import gzip
import os
import ssl
import urllib.request

import certifi
import pandas as pd

DATA_DIR = "data"
OUT_DIR = os.path.join("data", "processed")

# organism: (GAF file, UniProt TSV pattern, UniProt species mnemonic as in "OPSD_HUMAN")
ORGANISMS = {"Human": ("HUMAN-uniprot.gaf.gz", "uniprotkb_Human_*.tsv.gz", "HUMAN"),
    "Danre": ("DANRE-uniprot.gaf.gz", "uniprotkb_danre_*.tsv.gz", "DANRE"),
    "Mouse": ("MOUSE-uniprot.gaf.gz", "uniprotkb_Mouse_*.tsv.gz", "MOUSE"),}

HQ_EVIDENCE = {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "HTP", "HDA", "HMP", "HGI", "HEP", "TAS", "IC"}

# "protein binding" comes almost only from IPI and says nothing specific about function.
# It is removed as a direct annotation, but can still be reached by propagation from more specific binding terms.
EXCLUDED_TERMS = {"GO:0005515"}

# The GAF files were built with GO 2026-07-26 (see "!go-version" in their headers), which is not available for download.
# We use the release below. Terms obsoleted in between are mapped through alt_id / replaced_by or dropped.
GO_RELEASE = "2026-11-13"
GO_PATH = os.path.join(DATA_DIR, f"go-basic-{GO_RELEASE}.obo.gz")
GO_URLS = [f"https://github.com/geneontology/go-ontology/releases/download/v{GO_RELEASE}/go-basic.obo",
    f"https://release.geneontology.org/{GO_RELEASE}/ontology/go-basic.obo",]
ASPECTS = {"molecular_function": "mf", "biological_process": "bp", "cellular_component": "cc"}
ROOTS = {"GO:0003674", "GO:0008150", "GO:0005575"}


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
    return pd.DataFrame({"accession": df["Entry"], "entry_name": df["Entry Name"],
        "go_uniprot": df["Gene Ontology IDs"].apply(to_set), "sequence": df["Sequence"],})


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
    # Select only high quality Evidence Codes and drop uninformative terms
    hq_go = go_per_protein(gaf[gaf["evidence"].isin(HQ_EVIDENCE)])
    df = df.copy()
    df["go_terms"] = [(u & hq_go.get(a, set())) - EXCLUDED_TERMS for a, u in zip(df["accession"], df["go_uniprot"])]
    return df[df["go_terms"].map(len) > 0]


def process_organism(taxon, gaf_file, tsv_pattern, mnemonic):
    gaf = read_gaf(os.path.join(DATA_DIR, gaf_file))
    df = read_uniprot(glob.glob(os.path.join(DATA_DIR, tsv_pattern))[0])
    # The UniProt exports come from free-text queries, so they also contain other species
    other_species = ~df["entry_name"].str.endswith("_" + mnemonic)
    df = df[~other_species].assign(taxon=taxon)
    df = df[df["accession"].isin(gaf["accession"])]
    comparison = compare_go(df, go_per_protein(gaf))
    df = keep_hq_terms(df, gaf)
    n_diff = (comparison["only_in_uniprot"] != "") | (comparison["only_in_gaf"] != "")
    print(f"{taxon}: {other_species.sum()} UniProt entries of other species dropped, "
      f"{n_diff.sum()}/{len(comparison)} proteins with GAF vs UniProt differences, "
      f"{len(df)} proteins after evidence code filter")
    return df, comparison


def download_go():
    context = ssl.create_default_context(cafile=certifi.where())
    for url in GO_URLS:
        try:
            print(f"Downloading GO {GO_RELEASE} from {url}")
            request = urllib.request.Request(url, headers={"User-Agent": "protein-function-prediction"})
            with urllib.request.urlopen(request, timeout=300, context=context) as response:
                content = response.read()
            if not content.startswith(b"format-version"):
                raise ValueError("the response is not an OBO file")
            with gzip.open(GO_PATH, "wb") as fh:
                fh.write(content)
            return
        except Exception as error:
            print(f"  failed: {error}")
    raise SystemExit(f"Could not download GO. Save go-basic.obo of release {GO_RELEASE} as {GO_PATH[:-3]} and run again.")


def read_go():
    # Parse go-basic.obo: parents through is_a and part_of (true path rule), aspect, alt_id and obsolete terms
    plain_path = GO_PATH[:-3]
    if not os.path.exists(plain_path) and not os.path.exists(GO_PATH):
        download_go()
    go = {"parents": {}, "aspect": {}, "alt_ids": {}, "obsolete": set(), "replaced_by": {}, "version": None}
    term, in_term = None, False
    if os.path.exists(plain_path):
        opener = open(plain_path, encoding="utf-8")
    else:
        opener = gzip.open(GO_PATH, "rt", encoding="utf-8")
    with opener as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("["):
                in_term = line == "[Term]"
                continue
            if ": " not in line:
                continue
            key, value = line.split(": ", 1)
            if key == "data-version":
                go["version"] = value
            elif not in_term:
                continue
            elif key == "id":
                term = value
                go["parents"][term] = set()
            elif key == "namespace":
                go["aspect"][term] = ASPECTS.get(value)
            elif key == "is_a":
                go["parents"][term].add(value.split()[0])
            elif key == "relationship" and value.startswith("part_of "):
                go["parents"][term].add(value.split()[1])
            elif key == "alt_id":
                go["alt_ids"][value] = term
            elif key == "is_obsolete" and value == "true":
                go["obsolete"].add(term)
            elif key == "replaced_by":
                go["replaced_by"][term] = value
    if go["version"] != f"releases/{GO_RELEASE}":
        print(f"Warning: GO file has data-version {go['version']}, expected releases/{GO_RELEASE}")
    return go


def propagate(df, go):
    # Map each direct term to the current GO ID, add all its ancestors and split the result by aspect
    ancestors = {}

    def up(term):
        if term not in ancestors:
            ancestors[term] = {term}.union(*(up(p) for p in go["parents"].get(term, ())))
        return ancestors[term]

    def current_id(term):
        term = go["alt_ids"].get(term, term)
        if term in go["obsolete"]:
            term = go["replaced_by"].get(term)
        return term if term in go["parents"] and term not in go["obsolete"] else None

    df = df.copy()
    mapped = [{current_id(t) for t in terms} - {None} for terms in df["go_terms"]]
    n_changed = sum(len(set(terms) - m) for terms, m in zip(df["go_terms"], mapped))
    full = [set().union(*(up(t) for t in m)) - ROOTS for m in mapped]
    for aspect in ASPECTS.values():
        df["go_" + aspect] = [";".join(sorted(t for t in terms if go["aspect"].get(t) == aspect)) for terms in full]
    print(f"GO {go['version']}: {n_changed} direct annotations remapped or dropped (alt_id / obsolete), "
      + ", ".join(f"{a}: {sum(1 for x in df['go_' + a] if x)} proteins" for a in ASPECTS.values()))
    return df


def write_fasta(df, path):
    # fasta file
    with open(path, "w") as fh:
        for acc, taxon, seq in zip(df["accession"], df["taxon"], df["sequence"]):
            fh.write(f">{acc} taxon={taxon}\n{seq}\n")


def save(master, comparison):
    os.makedirs(OUT_DIR, exist_ok=True)
    out = master[["accession", "entry_name", "taxon", "go_terms", "go_mf", "go_bp", "go_cc", "sequence"]].copy()
    out["go_terms"] = out["go_terms"].apply(lambda s: ";".join(sorted(s)))
    out.to_csv(os.path.join(OUT_DIR, "master.tsv"), sep="\t", index=False)
    comparison.to_csv(os.path.join(OUT_DIR, "go_comparison.tsv"), sep="\t", index=False)
    write_fasta(out, os.path.join(OUT_DIR, "proteins_hq.fasta"))


def main():
    go = read_go()
    results = [process_organism(t, gaf, tsv, mnemonic) for t, (gaf, tsv, mnemonic) in ORGANISMS.items()]
    master = pd.concat([r[0] for r in results], ignore_index=True)
    comparison = pd.concat([r[1] for r in results], ignore_index=True)
    master = propagate(master, go)
    save(master, comparison)


if __name__ == "__main__":
    main()
