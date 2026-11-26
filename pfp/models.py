"""Per-ontology models. Every model gets the same interface, so the predictor does not depend on the architecture.

forward(x, mask) takes
    x     float tensor (B, N, D): one-hot encoded sequences (N = 22 tokens, D = maximum length)
    mask  bool tensor (B, D): True for residues, False for padding
and returns logits of shape (B, P), where P is the vocabulary size of the ontology.
The predictor turns logits into scores with a sigmoid, so a real model can be trained with BCEWithLogitsLoss.

To add an architecture: subclass OntologyModel, implement forward and register the class in MODELS.
"""
import torch
from torch import nn


class OntologyModel(nn.Module):
    """Base class: one model predicts the P terms of one ontology."""

    def __init__(self, n_terms):
        super().__init__()
        self.n_terms = n_terms

    @staticmethod
    def check_input(x, mask):
        if x.dim() != 3:
            raise ValueError(f"x must have shape (B, N, D), got {tuple(x.shape)}")
        if mask is not None and mask.shape != (x.shape[0], x.shape[2]):
            raise ValueError(f"mask must have shape (B, D) = {(x.shape[0], x.shape[2])}, got {tuple(mask.shape)}")

    def forward(self, x, mask=None):
        raise NotImplementedError


class DummyModel(OntologyModel):
    """Placeholder model: ignores the sequence content (and the mask) and returns random logits of shape (B, P)."""

    def forward(self, x, mask=None):
        self.check_input(x, mask)
        return torch.randn(x.shape[0], self.n_terms, device=x.device)


MODELS = {"dummy": DummyModel}


def build_model(name, n_terms, **kwargs):
    """Create the model registered under `name` for an ontology with n_terms GO terms."""
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}, available: {', '.join(MODELS)}")
    return MODELS[name](n_terms=n_terms, **kwargs)
