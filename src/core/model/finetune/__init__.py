"""Finetuning module for molecular property prediction.

This module provides interfaces and heads for finetuning foundation models
on downstream tasks.
"""

from .interface import FoundationModelInterface
from .mmelon import MMELONFoundationModel
from .heads import (
    FinetuneHead,
    BinaryClassificationHead,
    RegressionHead,
    ProteinAwareBinaryClassificationHead,
    get_activation
)

__all__ = [
    'FoundationModelInterface',
    'MMELONFoundationModel',
    'FinetuneHead',
    'BinaryClassificationHead',
    'RegressionHead',
    'ProteinAwareBinaryClassificationHead',
    'get_activation'
]
