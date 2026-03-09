"""ML views utilities."""

from django.contrib import messages
from django.shortcuts import redirect

from data.models import Dataset
from preprocessing.models import PreprocessingPipeline


def load_dataset_and_pipeline_from_session(request):
    """
    Loads Dataset and Pipeline based on session IDs.
    Enforces DB as the source of truth.
    Returns: (dataset, pipeline) or (redirect_response)
    """
    dataset_id = request.session.get("dataset_id")
    pipeline_id = request.session.get("pipeline_id")

    # 1. Checking if Dataset exists in DATABASE
    if not dataset_id:
        messages.warning(request, "Wybierz najpierw zbiór danych.")
        return redirect("data:load_data")

    try:
        dataset = Dataset.objects.get(dataset_id=dataset_id, user=request.user)
    except Dataset.DoesNotExist:
        # Conflict: Session remembers ID, but it doesn't exist in the database (e.g. deleted in another tab)
        # Fix: Clear session
        # Repair: Clear session
        del request.session["dataset_id"]
        if "pipeline_id" in request.session:
            del request.session["pipeline_id"]
        messages.error(request, "Wybrany zbiór danych już nie istnieje.")
        return redirect("data:load_data")

    # 2. Checking if Pipeline exists in DATABASE
    pipeline = None
    if pipeline_id:
        try:
            pipeline = PreprocessingPipeline.objects.get(
                id=pipeline_id, dataset=dataset
            )
        except PreprocessingPipeline.DoesNotExist:
            # Conflict: Session remembers pipeline, but it doesn't exist.
            # Repair: Clear session
            del request.session["pipeline_id"]
            # Optional: Try to find active pipeline in the database (Autorecovery)
            pipeline = dataset.pipelines.filter(is_active=True).first()
            if pipeline:
                request.session["pipeline_id"] = pipeline.id

    return dataset, pipeline
