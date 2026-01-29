"""ML views utilities."""
from django.contrib import messages
from django.shortcuts import get_object_or_404, redirect

from data.models import Dataset
from data.services import load_dataset_dataframe
from preprocessing.models import PreprocessingPipeline
from preprocessing.services import get_train_test_dataframes


def load_dataset_and_pipeline_from_session(request):
    """
    Returns (dataset, pipeline) or redirect response.
    """
    dataset_id = request.session.get('dataset_id')
    if not dataset_id:
        messages.error(request, "Nie wybrano datasetu. Wybierz dataset lub załaduj nowy.")
        return redirect("data:load_data")

    dataset = get_object_or_404(Dataset, dataset_id=dataset_id, user=request.user)
    pipeline_id = request.session.get('pipeline_id')
    pipeline = None
    if pipeline_id:
        pipeline = PreprocessingPipeline.objects.filter(
            id=pipeline_id,
            dataset=dataset,
            is_active=True
        ).first()

    return dataset, pipeline
