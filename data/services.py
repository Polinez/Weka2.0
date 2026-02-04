"""Data management services."""
import io
import mimetypes
import os
import uuid
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.contrib.auth.models import User
from django.db import transaction
import logging
from django.core.mail import EmailMessage, BadHeaderError
from smtplib import SMTPException

from core.enums import ProblemType

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
    
    # Save file
    df.to_csv(file_path, index=False)
    relative_path = f"datasets/{user.id}/{dataset_id}.csv"

    # Database operations in transaction
    try:
        with transaction.atomic():
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
    except Exception:
        # If database fails, delete file to avoid clutter
        if file_path.exists():
            os.remove(file_path)
        raise # Raise error to view to handle it

def infer_and_save_columns(dataset: Dataset, df: pd.DataFrame) -> None:
    """Infers column types and creates DatasetColumn records.
        Saves type of each column as 'numeric', 'categorical', or 'datetime'."""
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

def set_target_column(dataset: Dataset, column_name: str | None) -> None:
    """Sets is_target for the given column, clears others."""
    dataset.columns.update(is_target=False)
    if column_name:
        dataset.columns.filter(name=column_name).update(is_target=True)

def configure_analysis_settings(dataset: Dataset, post_data: dict) -> tuple:
    """
    Validates inputs, maps problem types, sets target, updates Dataset
    and creates/updates PreprocessingPipeline.
    Returns: (pipeline_id, split_config)
    Raises: ValueError if validation fails.
    """
    df = load_dataset_dataframe(dataset)
    columns = list(df.columns)

    # Maps problem types from form to enum
    raw_type = post_data.get("problem_type") or post_data.get("learning_type")
    problem_type_map = {
        'Classification': ProblemType.CLASSIFICATION,
        'Regression': ProblemType.REGRESSION,
        'Clustering': ProblemType.CLUSTERING,
        'Dimensionality_Reduction': ProblemType.DIMENSIONALITY_REDUCTION,
        'CLASSIFICATION': ProblemType.CLASSIFICATION,
        'REGRESSION': ProblemType.REGRESSION,
        'CLUSTERING': ProblemType.CLUSTERING,
        'DIM_REDUCTION': ProblemType.DIMENSIONALITY_REDUCTION,
    }
    problem_type = problem_type_map.get(raw_type, raw_type)

    # Download and validate parameters from form
    target_col = post_data.get("target_column")
    try:
        test_size = float(post_data.get("test_size", 0.2))
        random_state = int(post_data.get("random_state", 42))
        if not 0.1 <= test_size <= 0.9:
            raise ValueError("Rozmiar zbioru testowego musi być między 0.1 a 0.9.")
    except ValueError as e:
        raise ValueError(f"Nieprawidłowe parametry liczbowe: {e}")

    # Validate target column based on problem type. Regression/Classification need target.
    if problem_type in [ProblemType.REGRESSION, ProblemType.CLASSIFICATION]:
        if not target_col:
            raise ValueError("Kolumna decyzyjna jest wymagana dla Regresji i Klasyfikacji.")
        elif target_col not in columns:
            raise ValueError("Wybrana kolumna decyzyjna nie istnieje w zbiorze.")

    # Save settings to Dataset and Pipeline
    dataset.problem_type = problem_type
    dataset.save()
    set_target_column(dataset, target_col)

    split_config = {'test_size': test_size, 'random_state': random_state}

    # use preprocessing service to get/create pipeline
    from preprocessing.services import get_or_create_active_pipeline
    pipeline = get_or_create_active_pipeline(dataset, split_config)
    pipeline.split_config = split_config
    pipeline.save()

    return pipeline.id, split_config



def get_target_column_name(dataset: Dataset) -> str | None:
    """Returns the name of the target column, or None."""
    col = dataset.columns.filter(is_target=True).first()
    return col.name if col else None




def archive_dataset_and_cleanup(dataset: Dataset) -> None:
    """Soft delete dataset and hard delete heavy related objects."""
    dataset.pipelines.all().delete()
    dataset.runs.all().delete()
    dataset.is_archived = True
    dataset.save()

def restore_dataset_from_archive(dataset: Dataset) -> None:
    """Restores dataset visibility."""
    dataset.is_archived = False
    dataset.save()

def send_contact_email(name: str, user_email: str, message: str) -> None:
    """Sends contact email to admin."""
    logger = logging.getLogger(__name__)

    if not all([name, user_email, message]):
        logger.warning("Empty message form.")
        return False

    # 2. Treść
    subject = f'[Weka2.0] Message from {name}'
    body = (
        f"User: {name}\n"
        f"Email: {user_email}\n\n"
        f"--- Message ---\n"
        f"{message}"
    )

    try:
        email_msg = EmailMessage(
            subject=subject,
            body=body,
            from_email=settings.EMAIL_HOST_USER,

            to=[settings.EMAIL_HOST_USER],

            reply_to=[user_email]
        )

        email_msg.send(fail_silently=False)
        return True

    except Exception as e:
        logger.error(f"Error: {e}")
        return False
