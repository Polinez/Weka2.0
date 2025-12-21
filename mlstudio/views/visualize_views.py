from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from .utils import load_data_from_session
from ..models import MLRun
from loadData.models import Dataset
import pandas as pd
import io

from .utils import (
    generate_classification_plots,
    generate_regression_plots,
    generate_clustering_plots,
    generate_dim_reduction_plots
)

@login_required()
def visualize(request):
    dataset = load_data_from_session(request)
    if not isinstance(dataset, Dataset):
        return dataset

    latest_run = MLRun.objects.filter(dataset=dataset, user=request.user).order_by('-created_at').first()

    if not latest_run:
        return render(request, "visualize.html", {
            "dataset": dataset,
            "error": "Nie znaleziono żadnych uruchomień modelu dla tego datasetu. Uruchom model w zakładce 'Run'."
        })

    model_type = latest_run.model.model_type
    result_data = latest_run.result or {}

    # Check if there was an error during model execution
    if result_data.get('error'):
        return render(request, "visualize.html", {
            "dataset": dataset,
            "error": f"Model nie został uruchomiony poprawnie: {result_data.get('error')}"
        })

    # Use train_data (contains preprocessed data) - for clustering it's all data, for others it's training set
    train_data_csv = request.session.get('train_data')
    if train_data_csv:
        df = pd.read_csv(io.StringIO(train_data_csv))
    else:
        df = pd.read_csv(io.StringIO(dataset.data))

    plots = []

    try:
        if model_type == 'CLASSIFICATION':
            plots = generate_classification_plots(result_data, df, dataset.target_column) or []
        elif model_type == 'REGRESSION':
            plots = generate_regression_plots(result_data, df, dataset.target_column) or []
        elif model_type == 'CLUSTERING':
            plots = generate_clustering_plots(result_data, df, dataset.target_column) or []
        elif model_type == 'DIM_REDUCTION':
            plots = generate_dim_reduction_plots(result_data, df, dataset.target_column) or []

        # add additional plots for every single model
        for key, value in result_data.items():
            if key.startswith('plot_') and value:
                plots.append(value)

    except Exception as e:
        return render(request, "visualize.html", {
            "dataset": dataset,
            "error": f"Wystąpił błąd podczas generowania wizualizacji: {e}"
        })

    return render(request, "visualize.html", {
        "dataset": dataset,
        "latest_run": latest_run,
        "plots": plots
    })