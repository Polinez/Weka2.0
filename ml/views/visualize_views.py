"""Visualization views."""
from django.contrib.auth.decorators import login_required
from django.shortcuts import render

from data.models import Dataset
from data.services import load_dataset_dataframe
from ml.models import MLRun
from preprocessing.services import get_train_test_dataframes
from .utils import load_dataset_and_pipeline_from_session


@login_required
def visualize(request):
    """Visualize latest ML run results."""
    result = load_dataset_and_pipeline_from_session(request)
    if not isinstance(result, tuple):
        return result
    dataset, pipeline = result

    latest_run = MLRun.objects.filter(
        dataset=dataset,
        user=request.user
    ).order_by('-created_at').first()

    if not latest_run:
        return render(request, "visualize.html", {
            "dataset": dataset,
            "error": "Brak uruchomień. Uruchom model w zakładce Run.",
        })

    metrics = latest_run.metrics or {}
    if metrics.get('error'):
        return render(request, "visualize.html", {
            "dataset": dataset,
            "error": f"Błąd modelu: {metrics.get('error')}",
        })

    train_test = get_train_test_dataframes(pipeline) if pipeline else None
    if train_test:
        df = train_test[0]
    else:
        df = load_dataset_dataframe(dataset)

    plots = metrics.get('plots_base64', [])
    eval_data = latest_run.metrics
    if not plots and eval_data:
        from ml.services.plot_service import (
            generate_classification_plots,
            generate_regression_plots,
            generate_clustering_plots,
            generate_dim_reduction_plots,
        )
        model_type = latest_run.model.type
        generators = {
            'Classification': generate_classification_plots,
            'Regression': generate_regression_plots,
            'Clustering': generate_clustering_plots,
            'Dimensionality_Reduction': generate_dim_reduction_plots,
        }
        gen = generators.get(model_type, lambda *a: [])
        plots = gen(eval_data, df, dataset.target_column) or []

    return render(request, "visualize.html", {
        "dataset": dataset,
        "latest_run": latest_run,
        "plots": plots,
    })
