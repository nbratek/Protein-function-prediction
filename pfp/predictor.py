"""Predictor: one model per ontology turns sequences into GO term scores, saved as CSV (ProteinID,GOterm,Score)."""
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pfp.data import BATCH_SIZE, ProteinDataset, make_loader
from pfp.encoding import MAX_LEN

CSV_COLUMNS = ["ProteinID", "GOterm", "Score"]


class Predictor:
    """Runs the model of every ontology on batches of sequences.

    models: {ontology: model}, vocabs: {ontology: list of GO terms}. Model outputs follow the vocabulary order.
    """

    def __init__(self, models, vocabs, max_len=MAX_LEN, batch_size=BATCH_SIZE, device="cpu"):
        if set(models) != set(vocabs):
            raise ValueError(f"models {sorted(models)} and vocabularies {sorted(vocabs)} cover different ontologies")
        self.models = {ontology: model.to(device).eval() for ontology, model in models.items()}
        self.vocabs = vocabs
        self.max_len = max_len
        self.batch_size = batch_size
        self.device = device

    def predict(self, records):
        """For records [(protein ID, sequence), ...] yield (protein IDs, {ontology: scores}) per batch.

        Scores of an ontology have shape (B, P) and lie in [0, 1].
        """
        dataset = ProteinDataset([protein_id for protein_id, _ in records], [sequence for _, sequence in records])
        for batch in make_loader(dataset, self.batch_size, self.max_len):
            batch = batch.to(self.device)
            scores = {}
            with torch.inference_mode():
                for ontology, model in self.models.items():
                    logits = model(batch.x, batch.mask)
                    expected = (len(batch.ids), len(self.vocabs[ontology]))
                    if logits.shape != expected:
                        raise ValueError(f"{ontology} model returned shape {tuple(logits.shape)}, expected {expected}")
                    scores[ontology] = torch.sigmoid(logits).cpu().numpy()
            yield batch.ids, scores


def write_predictions(path, batches, vocabs, decimals=4):
    """Write one CSV row per protein and GO term: ProteinID,GOterm,Score.

    batches: iterable of (protein IDs, {ontology: scores of shape (B, P)}), e.g. Predictor.predict(...).
    A path ending with .gz is gzip-compressed. Returns the number of rows written.
    """
    ontologies = list(vocabs)
    terms = np.concatenate([np.asarray(vocabs[ontology], dtype=object) for ontology in ontologies])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    n_rows = 0
    with opener(path, "wt", newline="") as fh:
        fh.write(",".join(CSV_COLUMNS) + "\n")
        for ids, scores in batches:
            matrix = np.concatenate([scores[ontology] for ontology in ontologies], axis=1)  # (B, sum of P)
            rows = pd.DataFrame({"ProteinID": np.repeat(np.asarray(ids, dtype=object), len(terms)),
                                 "GOterm": np.tile(terms, len(ids)), "Score": matrix.ravel()})
            rows.to_csv(fh, header=False, index=False, float_format=f"%.{decimals}f", lineterminator="\n")
            n_rows += len(rows)
    return n_rows
