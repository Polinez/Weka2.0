"""Service for dataset exploration and statistics."""

import pandas as pd
from data.services import load_dataset_dataframe
from preprocessing.services import get_train_test_dataframes
from ml.services.plot_service import generate_exploration_histogram


def get_active_dataframe(dataset, pipeline) -> pd.DataFrame:
    """
    Loads the most relevant dataframe:
    1. Pipeline processed train set (if available).
    2. Raw dataset file (fallback).
    """
    if pipeline:
        try:
            train_test = get_train_test_dataframes(pipeline)
            if train_test:
                return train_test[0]  # Returning training set
        except Exception:
            pass  # Ignoring cache errors, fallback to raw

    return load_dataset_dataframe(dataset)


def get_exploration_context(
    dataset, pipeline, selected_column_session: str | None
) -> dict:
    """
    Prepares all data needed for the Explore view:
    - Dataframe preview
    - Statistics
    - Histogram
    - Selected column logic
    """
    df = get_active_dataframe(dataset, pipeline)
    columns = df.columns.tolist()

    # 1. Intelligent column selection (Priority: Session -> Target -> First)
    selected_column = selected_column_session
    if selected_column not in columns:
        if dataset.target_column and dataset.target_column in columns:
            selected_column = dataset.target_column
        else:
            selected_column = columns[0] if columns else None

    # 2. Generating statistics (Pandas describe)
    try:
        stats_df = df.describe()
        # Conversion to format easy for Django template:
        # [['mean', 5.4, 3.2], ['std', 1.1, 0.5], ...]
        statistics = stats_df.reset_index().values.tolist()
        stat_columns = ["Statystyka"] + stats_df.columns.tolist()
    except Exception:
        # Security for empty/strange df
        statistics = []
        stat_columns = []

    # 3. Plot and preview
    graph = generate_exploration_histogram(df, selected_column)
    data_preview = df.head().values.tolist()

    return {
        "data": data_preview,
        "columns": columns,
        "statistics": statistics,
        "stat_columns": stat_columns,
        "graph": graph,
        "selected_column": selected_column,  # Returning, because it could have changed (auto-select)
    }
