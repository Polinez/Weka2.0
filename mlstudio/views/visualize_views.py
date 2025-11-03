from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from .utils import load_data_from_session
from ..models import MLRun
from loadData.models import Dataset
import pandas as pd
import io
import numpy as np

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

    working_data_csv = request.session.get('working_data', dataset.data)
    df = pd.read_csv(io.StringIO(working_data_csv))

    model_type = latest_run.model.model_type
    result_data = latest_run.result
    plots = []

    try:
        if model_type == 'CLASSIFICATION':
            plots = generate_classification_plots(result_data, df, dataset.target_column)
        elif model_type == 'REGRESSION':
            plots = generate_regression_plots(result_data, df, dataset.target_column)
        elif model_type == 'CLUSTERING':
            plots = generate_clustering_plots(result_data, df, dataset.target_column)
        elif model_type == 'DIM_REDUCTION':
            plots = generate_dim_reduction_plots(result_data, df, dataset.target_column)

        # add additional plots for every single model
        for key, value in result_data.items():
            # Check if key begins with 'plot_' and value is not None
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