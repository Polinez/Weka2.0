import uuid
from django.db import models
from django.contrib.auth.models import User

from core.enums import ProblemType, ParamDataType
from preprocessing.models import PreprocessingPipeline


class MLModel(models.Model):
    """ML algorithm definition."""

    name = models.CharField(max_length=100, unique=True)
    library = models.CharField(max_length=50, default="scikit-learn")
    type = models.CharField(
        max_length=30,
        choices=ProblemType.choices,
    )
    description = models.TextField(blank=True)

    def __str__(self):
        return self.name


class ModelParameterDef(models.Model):
    """Parameter definition for an ML model."""

    model = models.ForeignKey(
        MLModel, on_delete=models.CASCADE, related_name="parameter_defs"
    )
    name = models.CharField(max_length=100)
    default_value = models.CharField(max_length=200, default="")
    data_type = models.CharField(
        max_length=10,
        choices=ParamDataType.choices,
        default=ParamDataType.STR,
    )
    description = models.TextField(blank=True)
    options = models.JSONField(default=list, blank=True)

    def __str__(self):
        return f"{self.model.name}.{self.name}"


class MLRun(models.Model):
    """Single ML experiment run."""

    run_id = models.UUIDField(
        primary_key=True, default=uuid.uuid4, editable=False, unique=True
    )
    user = models.ForeignKey(User, on_delete=models.CASCADE)

    pipeline = models.ForeignKey(
        PreprocessingPipeline, on_delete=models.CASCADE, related_name="runs"
    )

    model = models.ForeignKey(MLModel, on_delete=models.CASCADE)
    status = models.CharField(max_length=20, default="Success")

    used_parameters = models.JSONField(default=dict)
    metrics = models.JSONField(default=dict)
    plots_paths = models.JSONField(default=dict, blank=True)
    model_binary_path = models.CharField(max_length=500, null=True, blank=True)
    execution_time_ms = models.IntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return (
            f"Run {self.run_id} | {self.user.username} | {self.pipeline.dataset.name}"
        )
