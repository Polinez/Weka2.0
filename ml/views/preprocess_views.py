"""Preprocessing views."""
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_POST

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
import numpy as np

from data.models import Dataset
from data.services import load_dataset_dataframe, get_target_column_name
from preprocessing.models import PreprocessingType
from preprocessing.services import (
    get_or_create_active_pipeline,
    get_train_test_dataframes,
    apply_preprocessing_step,
    reset_pipeline,
    execute_pipeline_steps,
)
from .utils import load_dataset_and_pipeline_from_session


POL_NAMES = {
    'mean': 'Średnia',
    'median': 'Mediana',
    'mode': 'Najczęstsza wartość',
    'label_encoder': 'Kodowanie etykiet',
    'one_hot_encoder': 'Kodowanie binarne (One-Hot)',
    'standardization': 'Standaryzacja (Z-score)',
    'normalization': 'Normalizacja (Min-Max)',
}


@login_required
def preprocess(request):
    """GET: Show preprocessing page. POST: Select feature."""
    result = load_dataset_and_pipeline_from_session(request)
    if not isinstance(result, tuple):
        return result
    dataset, pipeline = result

    if not pipeline:
        messages.error(request, "Brak aktywnego pipeline. Skonfiguruj zadanie w Konfiguracja.")
        return redirect("data:set_target", dataset_id=dataset.dataset_id)

    try:
        df_train, df_test = get_train_test_dataframes(pipeline)
        if df_train is None:
            full_df = load_dataset_dataframe(dataset)
        else:
            full_df = df_train
    except Exception as e:
        messages.error(request, f"Błąd odczytu danych: {e}")
        return redirect("data:load_data")

    if request.method == 'POST':
        selected_feature = request.POST.get('feature')
        request.session['selected_feature'] = selected_feature or None
        return redirect('ml:preprocess')

    data_for_table = full_df.head().values.tolist()
    columns = full_df.columns.tolist()
    history = [s.type.name + ": " + str(s.parameters) for s in pipeline.steps.all().order_by('order')]

    context = {
        "dataset": dataset,
        "data": data_for_table,
        "columns": columns,
        "preprocessing_history": history,
        "selected_feature_name": None,
        "selected_feature_metadata": None,
    }

    selected_feature_name = request.session.get('selected_feature')
    if selected_feature_name and selected_feature_name in full_df.columns:
        col_series = full_df[selected_feature_name]
        col_type = col_series.dtype
        is_numeric = pd.api.types.is_numeric_dtype(col_type)
        context['selected_feature_name'] = selected_feature_name
        context['selected_feature_metadata'] = {
            'name': selected_feature_name,
            'dtype': str(col_type),
            'is_numeric': is_numeric,
            'is_categorical': not is_numeric,
            'missing_count': int(col_series.isnull().sum()),
        }

    return render(request, "preprocess.html", context)


@login_required
def preprocess_apply(request):
    """Apply preprocessing step."""
    if request.method != 'POST':
        return redirect('ml:preprocess')

    result = load_dataset_and_pipeline_from_session(request)
    if not isinstance(result, tuple):
        return result
    dataset, pipeline = result

    if not pipeline:
        messages.error(request, "Brak aktywnego pipeline.")
        return redirect("data:set_target", dataset_id=dataset.dataset_id)

    selected_feature = request.session.get('selected_feature')
    if not selected_feature:
        messages.error(request, "Nie wybrano żadnej cechy!")
        return redirect('ml:preprocess')

    operation = request.POST.get('apply_operation')
    if not operation:
        messages.warning(request, "Nie wybrano operacji.")
        return redirect('ml:preprocess')

    type_map = {
        'impute': ('Imputation', lambda: {'method': request.POST.get('imputation_method', 'mean')}),
        'encode': ('Encoding', lambda: {'method': request.POST.get('encoding_method', 'label_encoder')}),
        'scale': ('Scaling', lambda: {'method': request.POST.get('scaling_method', 'standardization')}),
        'delete': ('DropColumn', lambda: {}),
    }
    step_type_name, params_fn = type_map.get(operation, (None, None))
    if not step_type_name:
        messages.warning(request, "Nieznana operacja.")
        return redirect('ml:preprocess')

    target_col = get_target_column_name(dataset)
    if operation == 'delete' and selected_feature == target_col:
        messages.error(request, f"Nie można usunąć kolumny docelowej '{selected_feature}'.")
        return redirect('ml:preprocess')

    params = params_fn()
    err = apply_preprocessing_step(pipeline, step_type_name, selected_feature, params)
    if err:
        messages.error(request, err)
        return redirect('ml:preprocess')

    history_entry = POL_NAMES.get(params.get('method', ''), operation) + f" na '{selected_feature}'"
    if operation == 'delete':
        history_entry = f"Usunięto kolumnę '{selected_feature}'"
        request.session['selected_feature'] = None
    messages.success(request, history_entry)
    return redirect('ml:preprocess')


@login_required
@require_POST
def preprocess_reset(request):
    """Reset pipeline to initial state."""
    result = load_dataset_and_pipeline_from_session(request)
    if not isinstance(result, tuple):
        return result
    dataset, pipeline = result

    if pipeline:
        reset_pipeline(pipeline)
        request.session['selected_feature'] = None
        messages.info(request, "Przywrócono stan danych sprzed preprocessingu.")
    else:
        messages.error(request, "Brak pipeline do resetowania.")
        return redirect("data:load_data")

    return redirect('ml:preprocess')
