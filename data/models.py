import uuid
from django.db import models
from django.contrib.auth.models import User

from core.enums import ProblemType


class Dataset(models.Model):
    """Dataset - raw file metadata. Actual data stored on disk."""
    dataset_id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        unique=True,
    )
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=255, db_index=True)
    file_path = models.CharField(max_length=500)
    row_count = models.IntegerField(default=0)
    column_count = models.IntegerField(default=0)
    file_size_bytes = models.BigIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    is_archived = models.BooleanField(default=False)
    problem_type = models.CharField(
        max_length=30,
        choices=ProblemType.choices,
        null=True,
        blank=True,
    )

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['user', 'name'], name='data_unique_user_dataset')
        ]
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} ({self.row_count} rows)"

    @property
    def target_column(self):
        col = self.columns.filter(is_target=True).first()
        return col.name if col else None


class DatasetColumn(models.Model):
    """Column metadata for a dataset."""
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='columns')
    name = models.CharField(max_length=255)
    inferred_type = models.CharField(max_length=50, default='unknown')
    is_target = models.BooleanField(default=False)

    class Meta:
        ordering = ['id']

    def __str__(self):
        return f"{self.dataset.name}.{self.name}"
