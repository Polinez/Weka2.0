"""Data management services."""
import io
import mimetypes
import uuid
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.contrib.auth.models import User

from .models import Dataset, DatasetColumn


def validate_csv_file(uploaded_file):
    """
    Validates uploaded file as CSV.
    Returns (df, error) - df is DataFrame or None, error is str or None.
    """
    if not uploaded_file.name.lower().endswith('.csv'):
        return None, "Plik musi mieć rozszerzenie .csv."

    mime_type, _ = mimetypes.guess_type(uploaded_file.name)
    if mime_type not in ['text/csv', 'application/vnd.ms-excel', 'text/plain', None]:
        return None, "Nieprawidłowy typ pliku. Dozwolone są tylko pliki CSV."

    try:
        file_data = uploaded_file.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(file_data))
    except Exception:
        return None, "Plik nie jest prawidłowym CSV."

    return df, None


def save_uploaded_file(user: User, uploaded_file, df: pd.DataFrame) -> Dataset:
    """
    Saves CSV to disk and creates Dataset + DatasetColumn records.
    """
    dataset_id = uuid.uuid4()
    user_dir = Path(settings.MEDIA_ROOT) / 'datasets' / str(user.id)
    user_dir.mkdir(parents=True, exist_ok=True)
    file_path = user_dir / f"{dataset_id}.csv"
    df.to_csv(file_path, index=False)

    relative_path = f"datasets/{user.id}/{dataset_id}.csv"
    dataset = Dataset.objects.create(
        dataset_id=dataset_id,
        user=user,
        name=uploaded_file.name,
        file_path=relative_path,
        row_count=len(df),
        column_count=len(df.columns),
        file_size_bytes=file_path.stat().st_size,
    )
    infer_and_save_columns(dataset, df)
    return dataset


def infer_and_save_columns(dataset: Dataset, df: pd.DataFrame) -> None:
    """Infers column types and creates DatasetColumn records."""
    DatasetColumn.objects.filter(dataset=dataset).delete()
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            inferred = 'numeric'
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            inferred = 'datetime'
        else:
            inferred = 'categorical'
        DatasetColumn.objects.create(
            dataset=dataset,
            name=col,
            inferred_type=inferred,
            is_target=False,
        )


def load_dataset_dataframe(dataset: Dataset, file_path_override: str = None) -> pd.DataFrame:
    """Loads DataFrame from dataset file path or override."""
    path = file_path_override or dataset.file_path
    full_path = Path(settings.MEDIA_ROOT) / path
    return pd.read_csv(full_path)


def get_target_column_name(dataset: Dataset) -> str | None:
    """Returns the name of the target column, or None."""
    col = dataset.columns.filter(is_target=True).first()
    return col.name if col else None


def set_target_column(dataset: Dataset, column_name: str | None) -> None:
    """Sets is_target for the given column, clears others."""
    dataset.columns.update(is_target=False)
    if column_name:
        dataset.columns.filter(name=column_name).update(is_target=True)
