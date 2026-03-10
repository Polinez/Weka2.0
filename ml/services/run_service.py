"""ML experiment run service."""

import base64
import time
import uuid
from pathlib import Path

import joblib
import pandas as pd
from django.conf import settings
from django.contrib.auth.models import User

from data.models import Dataset
from data.services import get_target_column_name, load_dataset_dataframe
from preprocessing.models import PreprocessingPipeline
from preprocessing.services import get_train_test_dataframes, split_dataframe
from ml.models import MLRun
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
        "test_size": split_config.get("test_size", 0.2),
        "random_state": split_config.get("random_state", 42),
    }
    model_params = used_parameters.get("model_parameters", used_parameters)

    if "test_size" in used_parameters:
        common_params["test_size"] = used_parameters["test_size"]
    if "random_state" in used_parameters:
        common_params["random_state"] = used_parameters["random_state"]

    try:
        # 1. Loading Data
        if pipeline:
            result = get_train_test_dataframes(pipeline)
            if not result:
                return {
                    "error": "Brak danych w pipeline. Skonfiguruj preprocessing.",
                    "status": "Failed",
                }
            df_train, df_test = result
        else:
            # Fallback for missing pipeline
            df = load_dataset_dataframe(dataset)
            df_train, df_test = split_dataframe(df, split_config, target_column)

        # 2. Initializing Model
        ModelClass = MODEL_MAPPING.get(model.name)
        if not ModelClass:
            return {
                "error": f"Model '{model.name}' nie jest obsługiwany.",
                "status": "Failed",
            }

        start = time.perf_counter()
        ml_instance = ModelClass(
            common_parameters=common_params,
            model_parameters=model_params,
            target_column=target_column,
        )

        # 3. Training & Evaluating
        evaluation = ml_instance.run(df_train, df_test)
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        if evaluation.get("error"):
            return {"error": evaluation["error"], "status": "Failed"}

        # 4. Preparing Metrics & Plots
        metrics = _extract_metrics(evaluation)
        plots_base64 = _generate_plots(evaluation, df_train, target_column, model.type)

        # 5. Saving Binary File AND PLOTS
        run_id = uuid.uuid4()
        model_dir = Path(settings.MEDIA_ROOT) / "models" / str(run_id)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Zapis modelu
        model_path = f"models/{run_id}/model.joblib"
        joblib.dump(ml_instance.model, Path(settings.MEDIA_ROOT) / model_path)

        # Save plot as png files and store relative paths in DB
        plots_paths_dict = {}
        for idx, b64_str in enumerate(plots_base64):
            img_data = base64.b64decode(b64_str)
            plot_rel_path = f"models/{run_id}/plot_{idx}.png"
            with open(Path(settings.MEDIA_ROOT) / plot_rel_path, "wb") as f:
                f.write(img_data)
            plots_paths_dict[f"plot_{idx}"] = plot_rel_path

        # 6. Creating Database Record
        ml_run = MLRun.objects.create(
            run_id=run_id,
            user=user,
            pipeline=pipeline,
            model=model,
            status="Success",
            used_parameters={"model_parameters": model_params},
            metrics=metrics,
            plots_paths=plots_paths_dict,
            model_binary_path=model_path,
            execution_time_ms=elapsed_ms,
        )

        return {
            "status": "Success",
            "run_obj": ml_run,  # Returning ready database object
            "run_id": run_id,
        }

    except Exception as e:
        return {
            "error": str(e),
            "status": "Failed",
        }


def _extract_metrics(evaluation: dict) -> dict:
    """Extract metrics from evaluation result."""
    metrics = {}
    desired_keys = [
        "accuracy",
        "f1",
        "mean_absolute_error",
        "mean_squared_error",
        "r2_score",
        "silhouette_score",
        "davies_bouldin_score",
        "total_explained_variance",
    ]
    for key in desired_keys:
        if key in evaluation:
            metrics[key] = evaluation[key]
    return metrics


def _generate_plots(
    evaluation: dict, df: pd.DataFrame, target_column: str, model_type: str
) -> list:
    """Generate plot base64 strings based on model type."""
    model_type_map = {
        "Classification": generate_classification_plots,
        "Regression": generate_regression_plots,
        "Clustering": generate_clustering_plots,
        "Dimensionality_Reduction": generate_dim_reduction_plots,
    }
    generator = model_type_map.get(model_type, lambda *a: [])
    plots = generator(evaluation, df, target_column) or []

    for key, value in evaluation.items():
        if key.startswith("plot_") and value:
            plots.append(value)
    return plots
