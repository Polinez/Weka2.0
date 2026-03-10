"""Service for handling visualization logic."""

from django.conf import settings
from django.db.models import F
from ml.models import MLRun
from data.services import load_dataset_dataframe
from preprocessing.services import get_train_test_dataframes
from ml.services.plot_service import (
    generate_classification_plots,
    generate_regression_plots,
    generate_clustering_plots,
    generate_dim_reduction_plots,
)


def get_latest_run_visualization_data(user, dataset, pipeline) -> tuple:
    """
    Retrieves the latest MLRun and its plots.
    If plots are missing in the DB but metrics exist, it attempts to regenerate them.
    Returns: (latest_run, plots_urls, error_message)
    """
    latest_run = (
        MLRun.objects.filter(pipeline__dataset=dataset, user=user)
        .order_by(F("created_at").desc())
        .first()
    )

    if not latest_run:
        return None, [], "Brak uruchomień. Uruchom model w zakładce Run."

    metrics = latest_run.metrics or {}

    # 1. Checking errors of the model itself
    if metrics.get("error"):
        return latest_run, [], f"Błąd modelu: {metrics.get('error')}"

    plots_urls = []

    # 2. If plots paths are saved, use them directly
    if latest_run.plots_paths:
        for key, path in latest_run.plots_paths.items():
            plots_urls.append(f"{settings.MEDIA_URL}{path}")

    # 3. Emergency logic: regenerating plots live if they are missing
    if not plots_urls and metrics:
        try:
            # We need to load data to draw plots
            if pipeline:
                res = get_train_test_dataframes(pipeline)
                df = res[0] if res else load_dataset_dataframe(dataset)
            else:
                df = load_dataset_dataframe(dataset)

            # Choosing appropriate generator
            generators = {
                "Classification": generate_classification_plots,
                "Regression": generate_regression_plots,
                "Clustering": generate_clustering_plots,
                "Dimensionality_Reduction": generate_dim_reduction_plots,
            }
            gen = generators.get(latest_run.model.type)

            if gen:
                # Generating plots based on saved metrics and raw data
                plots = gen(metrics, df, dataset.target_column) or []

                # Transform base64 strings to Data URIs
                for p in plots:
                    plots_urls.append(f"data:image/png;base64,{p}")

        except Exception as e:
            return latest_run, [], f"Nie udało się odtworzyć wykresów: {e}"

    return latest_run, plots_urls, None
