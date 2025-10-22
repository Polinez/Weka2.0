from django.db import models

class MLModel(models.Model):
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)

    def __str__(self):
        return self.name


class ModelParameter(models.Model):
    model = models.ForeignKey(MLModel, on_delete=models.CASCADE, related_name="parameters")
    name = models.CharField(max_length=100)
    value = models.CharField(max_length=100)
    is_common = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.model.name}: {self.name} = {self.value}"

