"""
Similarity module: semantic similarity functions for answer and step comparison.
"""

from cocoa_cot.similarity.base import BaseSimilarity
from cocoa_cot.similarity.cross_encoder import CrossEncoderSimilarity
from cocoa_cot.similarity.nli import NLISimilarity
from cocoa_cot.similarity.lexical import RougeL, BLEU
from cocoa_cot.similarity.step_aligned import StepAlignedSimilarity

__all__ = [
    "BaseSimilarity",
    "CrossEncoderSimilarity",
    "NLISimilarity",
    "RougeL",
    "BLEU",
    "StepAlignedSimilarity",
]
