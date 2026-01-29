import os
from django.db.models.signals import post_delete
from django.dispatch import receiver
from django.conf import settings
from .models import PreprocessingPipeline

@receiver(post_delete, sender=PreprocessingPipeline)
def delete_pipeline_files(sender, instance, **kwargs):
    """Delete cache files (train/test) when PreprocessingPipeline is deleted."""
    paths = [
        instance.processed_file_path,
        instance.processed_train_path,
        instance.processed_test_path
    ]
    for path in paths:
        if path:
            full_path = os.path.join(settings.MEDIA_ROOT, path)
            if os.path.isfile(full_path):
                try:
                    os.remove(full_path)
                except OSError:
                    pass