import os
from django.db.models.signals import post_delete
from django.dispatch import receiver
from django.conf import settings
from .models import Dataset

@receiver(post_delete, sender=Dataset)
def delete_dataset_file(sender, instance, **kwargs):
    """Delete source CSV file when Dataset is deleted."""
    if instance.file_path:
        full_path = os.path.join(settings.MEDIA_ROOT, instance.file_path)
        if os.path.isfile(full_path):
            try:
                os.remove(full_path)
            except OSError:
                pass