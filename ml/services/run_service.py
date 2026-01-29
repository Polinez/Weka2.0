"""ML experiment run service."""
import time
import uuid
from pathlib import Path

import joblib
import pandas as pd
from django.conf import settings
from django.contrib.auth.models import User

from data.models import Dataset
from data.services import get_target_column_name
from preprocessing.models import PreprocessingPipeline
from preprocessing.services import get_train_test_dataframes
from .model_registry import MODEL_MAPPING
from .plot_service import (
    generate_classification_plots,
    generate_regression_plots,
    generate_clustering_plots,
    generate_dim_reduction_plots,
)


def run_ml_experiment(
    user: User,
    dataset: Dataset,
    pipeline: PreprocessingPipeline | None,
    model,
    split_config: dict,
    used_parameters: dict,
) -> dict:
    """
    Runs ML experiment: load data, split, train, evaluate, save model and plots.
    Returns dict with run_id, status, metrics, error (if failed).
    """
    target_column = get_target_column_name(dataset)
    common_params = {
        'test_size': split_config.get('test_size', 0.2),
        'random_state': split_config.get('random_state', 42),
    }
    model_params = used_parameters.get('model_parameters', used_parameters)
    if 'test_size' in used_parameters:
        common_params['test_size'] = used_parameters['test_size']
    if 'random_state' in used_parameters:
        common_params['random_state'] = used_parameters['random_state']

    try:
        if pipeline:
            result = get_train_test_dataframes(pipeline)
            if not result:
                return {'error': 'Brak danych w pipeline. Skonfiguruj preprocessing.'}
            df_train, df_test = result
        else:
            from data.services import load_dataset_dataframe
            from sklearn.model_selection import train_test_split
            df = load_dataset_dataframe(dataset)
            test_size = split_config.get('test_size', 0.2)
            random_state = split_config.get('random_state', 42)

            stratify = None
            if target_column and target_column in df.columns:
                if df[target_column].value_counts().min() >= 2:
                    stratify = df[target_column]
                    
            df_train, df_test = train_test_split(
                df, test_size=test_size, random_state=random_state, stratify=stratify
            )

        ModelClass = MODEL_MAPPING.get(model.name)
        if not ModelClass:
            return {'error': f"Model '{model.name}' nie jest obsługiwany.'"}

        start = time.perf_counter()
        ml_instance = ModelClass(
            common_parameters=common_params,
            model_parameters=model_params,
            target_column=target_column,
        )
        evaluation = ml_instance.run(df_train, df_test)
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        if evaluation.get('error'):
            return {'error': evaluation['error'], 'status': 'Failed'}

        metrics = _extract_metrics(evaluation)
        plots_base64 = _generate_plots(evaluation, df_train, target_column, model.type)

        run_id = uuid.uuid4()
        model_dir = Path(settings.MEDIA_ROOT) / 'models' / str(run_id)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = f"models/{run_id}/model.joblib"
        joblib.dump(ml_instance.model, Path(settings.MEDIA_ROOT) / model_path)

        return {
            'run_id': run_id,
            'status': 'Success',
            'metrics': metrics,
            'plots_base64': plots_base64,
            'model_binary_path': model_path,
            'execution_time_ms': elapsed_ms,
            'evaluation': evaluation,
        }
    except Exception as e:
        return {
            'error': str(e),
            'status': 'Failed',
        }


def _extract_metrics(evaluation: dict) -> dict:
    """Extract metrics from evaluation result."""
    metrics = {}
    for key in ['accuracy', 'f1', 'mean_absolute_error', 'mean_squared_error', 'r2_score',
                'silhouette_score', 'davies_bouldin_score', 'total_explained_variance']:
        if key in evaluation:
            metrics[key] = evaluation[key]
    return metrics


def _generate_plots(evaluation: dict, df: pd.DataFrame, target_column: str, model_type: str) -> list:
    """Generate plot base64 strings based on model type."""
    model_type_map = {
        'Classification': generate_classification_plots,
        'Regression': generate_regression_plots,
        'Clustering': generate_clustering_plots,
        'Dimensionality_Reduction': generate_dim_reduction_plots,
    }
    generator = model_type_map.get(model_type, lambda *a: [])
    plots = generator(evaluation, df, target_column) or []
    for key, value in evaluation.items():
        if key.startswith('plot_') and value:
            plots.append(value)
    return plots
