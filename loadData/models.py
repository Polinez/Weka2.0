from django.db import models
from django.contrib.auth.models import User

LEARNING_TYPE_CHOICES = [
        ('REGRESSION', 'Regresja (Nadzorowane)'),
        ('CLASSIFICATION', 'Klasyfikacja (Nadzorowane)'),
        ('CLUSTERING', 'Klasteryzacja (Nienadzorowane)'),
        ('DIM_REDUCTION', 'Redukcja Wymiarowości (Nienadzorowane)'),
    ]

# Create your models here.
class Dataset(models.Model):
    name = models.CharField(max_length=255, db_index=True, null=False)
    data = models.TextField(null=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    target_column = models.CharField(max_length=255, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    learning_type = models.CharField(
        max_length=20,
        choices=LEARNING_TYPE_CHOICES,
        null=True, # can be null for datasets without specified learning type
        blank=True
    )
    test_size = models.FloatField(null=True, blank=True)
    random_state = models.IntegerField(null=True, blank=True)

    # one user can not have two datasets with the same name
    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['user', 'name'], name='unique_user_dataset')
        ]

    def __str__(self):
        return f"{self.data[:50]}...)>"
