import os
import shutil
from django.db.models.signals import post_delete
from django.dispatch import receiver
from django.conf import settings
from .models import MLRun

@receiver(post_delete, sender=MLRun)
def delete_ml_artifacts(sender, instance, **kwargs):
    """Delete model binary file (.joblib) and optional run folder when MLRun is deleted."""
    if instance.model_binary_path:
        full_path = os.path.join(settings.MEDIA_ROOT, instance.model_binary_path)
        if os.path.isfile(full_path):
            try:
                os.remove(full_path)
                # Optional: delete folder if it's empty (media/models/<uuid>/)
                folder = os.path.dirname(full_path)
                if os.path.isdir(folder) and not os.listdir(folder):
                    shutil.rmtree(folder)
            except OSError:
                pass