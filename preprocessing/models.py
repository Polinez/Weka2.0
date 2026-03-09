from django.db import models

from data.models import Dataset


class PreprocessingType(models.Model):
    """Registry of preprocessing operation types."""

    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    code_reference = models.CharField(max_length=200)

    def __str__(self):
        return self.name


class PreprocessingPipeline(models.Model):
    """Pipeline for a dataset - contains steps and cached output."""

    dataset = models.ForeignKey(
        Dataset, on_delete=models.CASCADE, related_name="pipelines"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)
    processed_file_path = models.CharField(max_length=500, null=True, blank=True)
    processed_train_path = models.CharField(max_length=500, null=True, blank=True)
    processed_test_path = models.CharField(max_length=500, null=True, blank=True)
    output_columns_metadata = models.JSONField(default=dict, blank=True)
    split_config = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Pipeline {self.id} for {self.dataset.name}"


class PreprocessingStep(models.Model):
    """Single step in a preprocessing pipeline."""

    pipeline = models.ForeignKey(
        PreprocessingPipeline, on_delete=models.CASCADE, related_name="steps"
    )
    order = models.IntegerField()
    type = models.ForeignKey(
        PreprocessingType, on_delete=models.CASCADE, related_name="steps"
    )
    parameters = models.JSONField(default=dict)
    applied_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["order"]
        unique_together = [["pipeline", "order"]]

    def __str__(self):
        return f"Step {self.order}: {self.type.name}"
