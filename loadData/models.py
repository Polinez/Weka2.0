from django.db import models

# Create your models here.
class Dataset(models.Model):
    name = models.CharField(max_length=255, unique=True, db_index=True, null=False)
    data = models.TextField(null=False)

    def __str__(self):
        return f"<Dataset(id={self.id}, name='{self.name}', data_preview='{self.data[:50]}...')>"
