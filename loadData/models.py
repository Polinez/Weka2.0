from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class Dataset(models.Model):
    name = models.CharField(max_length=255, db_index=True, null=False)
    data = models.TextField(null=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE)

    # one user can not have two datasets with the same name
    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['user', 'name'], name='unique_user_dataset')
        ]

    def __str__(self):
        return f"{self.data[:50]}...)>"
