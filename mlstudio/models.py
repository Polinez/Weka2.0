from django.db import models
from loadData.models import Dataset
from django.contrib.auth.models import User




PARAM_TYPES = [
        ('str', 'str'),
        ('int', 'int'),
        ('float', 'float'),
        ('bool', 'bool'),
    ]


class MLModel(models.Model):

    MODEL_TYPE_CHOICES = [
        ('REGRESSION', 'Regresja'),
        ('CLASSIFICATION', 'Klasyfikacja'),
        ('CLUSTERING', 'Klasteryzacja'),
        ('DIM_REDUCTION', 'Redukcja Wymiarowości'),
    ]

    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    model_type = models.CharField(
        max_length=20,
        choices=MODEL_TYPE_CHOICES,
        null=False,
        blank=False,
        default='CLASSIFICATION'
    )

    def __str__(self):
        return self.name


class ModelParameter(models.Model):
    model = models.ForeignKey(MLModel, on_delete=models.CASCADE, related_name="parameters")
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True, default='')
    value = models.CharField(max_length=100)
    data_type = models.CharField(
        max_length=10,
        choices=PARAM_TYPES,
        default='str',
    )

    def __str__(self):
        return f"{self.model.name}: {self.name} = {self.value}"



class CommonParameter(models.Model):
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True, default='')
    value = models.CharField(max_length=100)
    data_type = models.CharField(
        max_length=10,
        choices=PARAM_TYPES,
        default='float'
    )

    def __str__(self):
        return f"{self.name} = {self.value}"


class DatasetModelState(models.Model):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name="model_states")
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    model = models.ForeignKey(MLModel, on_delete=models.CASCADE)
    default_parameters = models.JSONField(default=dict)
    parameters = models.JSONField(default=dict)

    class Meta:
        unique_together = ("dataset", "user")  # onli 1 model state per user per dataset

    def __str__(self):
        return f"{self.user.username} | {self.dataset.name} | {self.model.name}"


class MLRun(models.Model):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name="runs")
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    model = models.ForeignKey("MLModel", on_delete=models.CASCADE)
    common_parameters = models.JSONField(default=dict)
    model_parameters = models.JSONField(default=dict)
    result = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Run {self.id} | {self.user.username} | {self.dataset.name}"