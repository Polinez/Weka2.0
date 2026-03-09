"""
Enums shared across the application.
"""

from django.db import models


class ProblemType(models.TextChoices):
    """Determines supervised/unsupervised mode for a dataset."""

    CLASSIFICATION = "Classification", "Klasyfikacja"
    REGRESSION = "Regression", "Regresja"
    CLUSTERING = "Clustering", "Klasteryzacja"
    DIMENSIONALITY_REDUCTION = "Dimensionality_Reduction", "Redukcja Wymiarowości"


class ParamDataType(models.TextChoices):
    """Data type for model parameter validation."""

    INT = "int", "int"
    FLOAT = "float", "float"
    STR = "str", "str"
    BOOL = "bool", "bool"
