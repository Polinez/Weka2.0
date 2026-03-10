"""Seed preprocessing types and ML models with parameters."""

from django.core.management.base import BaseCommand
from preprocessing.models import PreprocessingType
from ml.models import MLModel, ModelParameterDef
from ml.management.commands.seed_constants import PREPROCESSING_TYPES, ML_MODELS


class Command(BaseCommand):
    help = "Seed PreprocessingType and ML models with parameter definitions"

    def handle(self, *args, **options):
        self._seed_preprocessing_types()
        self._seed_ml_models()
        self.stdout.write(self.style.SUCCESS("Seed data updated successfully."))

    def _seed_preprocessing_types(self):
        for name, desc in PREPROCESSING_TYPES:
            PreprocessingType.objects.update_or_create(
                name=name, defaults={"description": desc}
            )
        self.stdout.write("PreprocessingType seeded.")

    def _seed_ml_models(self):
        for name, model_type, desc, params in ML_MODELS:
            model, _ = MLModel.objects.update_or_create(
                name=name,
                defaults={
                    "type": model_type,
                    "description": desc,
                    "library": "scikit-learn",
                },
            )
            for param_name, default, dtype, options, description in params:
                ModelParameterDef.objects.update_or_create(
                    model=model,
                    name=param_name,
                    defaults={
                        "default_value": default,
                        "data_type": dtype,
                        "options": options,
                        "description": description,
                    },
                )
        self.stdout.write("ML models and parameters seeded.")
