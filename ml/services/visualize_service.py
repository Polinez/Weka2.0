"""Service for handling visualization logic."""
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
    Returns: (latest_run, plots_list, error_message)
    """
    latest_run = MLRun.objects.filter(
        dataset=dataset, 
        user=user
    ).order_by('-created_at').first()

    if not latest_run:
        return None, [], "Brak uruchomień. Uruchom model w zakładce Run."

    metrics = latest_run.metrics or {}
    
    # 1. Checking errors of the model itself
    if metrics.get('error'):
        return latest_run, [], f"Błąd modelu: {metrics.get('error')}"

    # 2. Trying to get ready plots from database (Base64)
    plots = metrics.get('plots_base64', [])

    # 3. "Emergency" logic - regenerating plots live if they are missing
    if not plots and metrics:
        try:
            # We need to load data to draw plots
            if pipeline:
                res = get_train_test_dataframes(pipeline)
                df = res[0] if res else load_dataset_dataframe(dataset)
            else:
                df = load_dataset_dataframe(dataset)

            # Choosing appropriate generator
            generators = {
                'Classification': generate_classification_plots,
                'Regression': generate_regression_plots,
                'Clustering': generate_clustering_plots,
                'Dimensionality_Reduction': generate_dim_reduction_plots,
            }
            gen = generators.get(latest_run.model.type)
            
            if gen:
                # Generating plots based on saved metrics and raw data
                plots = gen(metrics, df, dataset.target_column) or []
                
        except Exception as e:
            return latest_run, [], f"Nie udało się odtworzyć wykresów: {e}"

    return latest_run, plots, None