"""Preprocessing services - apply steps, execute pipeline, reset."""
import os 
import io
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from django.conf import settings
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from django.db.models import Max

from data.models import Dataset
from data.services import load_dataset_dataframe, get_target_column_name
from .models import PreprocessingPipeline, PreprocessingStep, PreprocessingType


def get_feature_metadata(df: pd.DataFrame, feature_name: str) -> dict | None:
    """Returns metadata for frontend display (type, missing values, etc)."""
    if not feature_name or feature_name not in df.columns:
        return None
        
    col_series = df[feature_name]
    col_type = col_series.dtype
    is_numeric = pd.api.types.is_numeric_dtype(col_type)
    
    return {
        'name': feature_name,
        'dtype': str(col_type),
        'is_numeric': is_numeric,
        'is_categorical': not is_numeric,
        'missing_count': int(col_series.isnull().sum()),
    }


def handle_preprocessing_request(pipeline: PreprocessingPipeline, feature: str, operation: str, post_data: dict) -> tuple[str | None, str]:
    """
    Parses UI request, validates constraints (e.g. target col), and applies step.
    Returns: (error_message, success_message)
    """
    # 1. Security against deleting target column
    target_col = get_target_column_name(pipeline.dataset)
    if operation == 'delete' and feature == target_col:
        return f"Nie można usunąć kolumny docelowej '{feature}'.", ""

    # 2. Security against scaling target column
    if operation == 'scale' and feature == target_col:
        return f"Nie można skalować kolumny docelowej '{feature}'. Zmieniłoby to wartości predykcji i utrudniło interpretację wyników.", ""

    # 3. Mapping names from form to backend
    type_map = {
        'impute': ('Imputation', lambda: {'method': post_data.get('imputation_method', 'mean')}),
        'encode': ('Encoding', lambda: {'method': post_data.get('encoding_method', 'label_encoder')}),
        'scale': ('Scaling', lambda: {'method': post_data.get('scaling_method', 'standardization')}),
        'delete': ('DropColumn', lambda: {}),
    }

    step_type_name, params_fn = type_map.get(operation, (None, None))
    if not step_type_name:
        return "Nieznana operacja.", ""

    params = params_fn()
    
    # 4. Calling business logic
    err = apply_preprocessing_step(pipeline, step_type_name, feature, params)
    
    if err:
        return err, ""
        
    # 5. Generating success message (Polish names)
    pol_names = {
        'mean': 'Średnia', 'median': 'Mediana', 'mode': 'Najczęstsza wartość',
        'label_encoder': 'Kodowanie etykiet', 'one_hot_encoder': 'Kodowanie binarne (One-Hot)',
        'standardization': 'Standaryzacja (Z-score)', 'normalization': 'Normalizacja (Min-Max)'
    }
    
    if operation == 'delete':
        msg = f"Usunięto kolumnę '{feature}'"
    else:
        method_name = pol_names.get(params.get('method', ''), operation)
        msg = f"{method_name} na '{feature}'"
        
    return None, msg


def get_or_create_active_pipeline(dataset: Dataset, split_config: dict) -> PreprocessingPipeline:
    """Get or create active pipeline for dataset."""
    pipeline = dataset.pipelines.filter(is_active=True).first()
    if not pipeline:
        dataset.pipelines.update(is_active=False)
        pipeline = PreprocessingPipeline.objects.create(
            dataset=dataset,
            is_active=True,
            split_config=split_config,
        )
    else:
        pipeline.split_config = split_config
        pipeline.save()
    return pipeline


def get_data_source_path(pipeline: PreprocessingPipeline) -> str | None:
    """Returns path to load data from: processed train (for full df) or dataset raw."""
    if pipeline.processed_train_path:
        return pipeline.processed_train_path
    return pipeline.dataset.file_path


def split_dataframe(df: pd.DataFrame, split_config: dict, target_col: str | None):
    """Split df into train/test using split_config (Centralized Logic)."""
    test_size = split_config.get('test_size', 0.2)
    random_state = split_config.get('random_state', 42)
    
    stratify = None
    if target_col and target_col in df.columns:
        if df[target_col].value_counts().min() >= 2:
            stratify = df[target_col]
            
    return train_test_split(df, test_size=test_size, random_state=random_state, stratify=stratify)


def _apply_imputation(df_train: pd.DataFrame, df_test: pd.DataFrame, col: str, params: dict):
    method = params.get('method', 'mean')
    strategy = 'mean' if method == 'mean' else ('median' if method == 'median' else 'most_frequent')
    imputer = SimpleImputer(strategy=strategy)
    imputer.fit(df_train[[col]])
    df_train[col] = imputer.transform(df_train[[col]])
    df_test[col] = imputer.transform(df_test[[col]])


def _apply_encoding(df_train: pd.DataFrame, df_test: pd.DataFrame, col: str, params: dict):
    method = params.get('method', 'label_encoder')
    if method == 'label_encoder':
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
        df_train[[col]] = enc.fit_transform(df_train[[col]].astype(str))
        df_test[[col]] = enc.transform(df_test[[col]].astype(str))
        if df_test[col].isnull().any():
            imputer = SimpleImputer(strategy='most_frequent')
            imputer.fit(df_train[[col]])
            df_test[col] = imputer.transform(df_test[[col]])
    elif method == 'one_hot_encoder':
        ohe = OneHotEncoder(handle_unknown='ignore', drop=None, sparse_output=False, dtype=int)
        ohe.fit(df_train[[col]])
        new_cols = ohe.get_feature_names_out([col])
        tr_train = ohe.transform(df_train[[col]])
        tr_test = ohe.transform(df_test[[col]])
        df_train.drop(columns=[col], inplace=True)
        df_test.drop(columns=[col], inplace=True)
        for i, c in enumerate(new_cols):
            df_train[c] = tr_train[:, i]
            df_test[c] = tr_test[:, i]


def _apply_scaling(df_train: pd.DataFrame, df_test: pd.DataFrame, col: str, params: dict):
    method = params.get('method', 'standardization')
    scaler = StandardScaler() if method == 'standardization' else MinMaxScaler()
    scaler.fit(df_train[[col]])
    df_train[col] = scaler.transform(df_train[[col]])
    df_test[col] = scaler.transform(df_test[[col]])


def _apply_drop_column(df_train: pd.DataFrame, df_test: pd.DataFrame, col: str):
    df_train.drop(columns=[col], inplace=True)
    df_test.drop(columns=[col], inplace=True)


STEP_HANDLERS = {
    'Imputation': _apply_imputation,
    'Encoding': _apply_encoding,
    'Scaling': _apply_scaling,
    'DropColumn': _apply_drop_column,
}


def execute_pipeline_steps(pipeline: PreprocessingPipeline) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data from dataset, split, apply all steps in order, return (df_train, df_test).
    """
    dataset = pipeline.dataset
    split_config = pipeline.split_config or {}
    target_col = get_target_column_name(dataset)

    df = load_dataset_dataframe(dataset)
    df_train, df_test = split_dataframe(df, split_config, target_col)

    steps = pipeline.steps.all().order_by('order')
    for step in steps:
        handler = STEP_HANDLERS.get(step.type.name)
        if not handler:
            continue
        col = step.parameters.get('column')
        if not col or col not in df_train.columns:
            continue
        if step.type.name == 'DropColumn':
            _apply_drop_column(df_train, df_test, col)
        else:
            handler(df_train, df_test, col, step.parameters)

    return df_train, df_test


def apply_preprocessing_step(
    pipeline: PreprocessingPipeline,
    step_type_name: str,
    column: str,
    params: dict,
) -> str | None:
    """
    Add step to pipeline, execute, save train/test files. Returns error message or None.
    """
    step_type = PreprocessingType.objects.filter(name=step_type_name).first()
    if not step_type:
        return f"Nieznany typ operacji: {step_type_name}"

    next_order = (pipeline.steps.aggregate(Max('order'))['order__max'] or 0) + 1
    step_params = {'column': column, **params}

    PreprocessingStep.objects.create(
        pipeline=pipeline,
        order=next_order,
        type=step_type,
        parameters=step_params,
    )

    try:
        df_train, df_test = execute_pipeline_steps(pipeline)
    except Exception as e:
        pipeline.steps.filter(order=next_order).delete()
        return str(e)

    base = Path(settings.MEDIA_ROOT) / 'pipelines' / str(pipeline.id)
    base.mkdir(parents=True, exist_ok=True)
    train_path = f"pipelines/{pipeline.id}/train_{uuid.uuid4()}.csv"
    test_path = f"pipelines/{pipeline.id}/test_{uuid.uuid4()}.csv"
    df_train.to_csv(Path(settings.MEDIA_ROOT) / train_path, index=False)
    df_test.to_csv(Path(settings.MEDIA_ROOT) / test_path, index=False)

    pipeline.processed_train_path = train_path
    pipeline.processed_test_path = test_path
    pipeline.processed_file_path = train_path
    pipeline.output_columns_metadata = {'columns': list(df_train.columns)}
    pipeline.save()
    return None


def reset_pipeline(pipeline: PreprocessingPipeline) -> None:
    """Remove steps and clear processed paths AND DELETE FILES FROM DISK."""
    
    paths_to_remove = [
        pipeline.processed_file_path,
        pipeline.processed_train_path,
        pipeline.processed_test_path
    ]
    
    for relative_path in paths_to_remove:
        if relative_path:
            full_path = Path(settings.MEDIA_ROOT) / relative_path
            if full_path.exists():
                try:
                    os.remove(full_path)
                    print(f"Usunięto plik: {full_path}")
                except OSError as e:
                    print(f"Błąd usuwania pliku przy resecie: {e}")

    pipeline.steps.all().delete()
    pipeline.processed_file_path = None
    pipeline.processed_train_path = None
    pipeline.processed_test_path = None
    pipeline.output_columns_metadata = {}
    pipeline.save()


def get_train_test_dataframes(pipeline: PreprocessingPipeline) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Returns (df_train, df_test) or None if data not available."""
    # 1. Try to read from cache
    if pipeline.processed_train_path and pipeline.processed_test_path:
        base = Path(settings.MEDIA_ROOT)
        try:
            return (
                pd.read_csv(base / pipeline.processed_train_path),
                pd.read_csv(base / pipeline.processed_test_path),
            )
        except FileNotFoundError:
            pass 

    # 2. Generate live using centralized function
    dataset = pipeline.dataset
    split_config = pipeline.split_config or {}
    target_col = get_target_column_name(dataset)
    df = load_dataset_dataframe(dataset)
    
    return split_dataframe(df, split_config, target_col)
