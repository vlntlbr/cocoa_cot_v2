"""
CoCoA-CoT Light module: dual-embedding features and auxiliary MLP model.
"""

from cocoa_cot.light.dual_embedding import DualEmbeddingExtractor
from cocoa_cot.light.aux_model import AuxiliaryModel, CoCoACoTLight

__all__ = ["DualEmbeddingExtractor", "AuxiliaryModel", "CoCoACoTLight"]
