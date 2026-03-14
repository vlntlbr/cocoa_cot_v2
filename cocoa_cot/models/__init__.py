"""
Models module: LLM wrappers for white-box and black-box inference.
"""

from cocoa_cot.models.base import BaseModel
from cocoa_cot.models.hf_model import HFModel
from cocoa_cot.models.blackbox_model import BlackBoxModel

__all__ = ["BaseModel", "HFModel", "BlackBoxModel"]
