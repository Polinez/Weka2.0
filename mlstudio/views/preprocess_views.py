from django.shortcuts import render, redirect
from django.contrib import messages
from .utils import load_data_from_session
import pandas as pd
import io
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from loadData.models import Dataset
import numpy as np


def preprocess(request):
    """
    GET: Show user a page
    POST: Saves new features in session
    """
    dataset = load_data_from_session(request)
    if not isinstance(dataset, Dataset):
        return dataset

    # --- Logic for undo ---
    original_data = request.session.get('original_data')
    working_data = request.session.get('working_data')
    session_dataset_id = request.session.get('dataset_id_for_data')

    if not original_data or session_dataset_id != dataset.id:
        request.session['original_data'] = dataset.data
        request.session['working_data'] = dataset.data
        request.session['dataset_id_for_data'] = dataset.id
        request.session['preprocessing_history'] = []
        request.session['selected_feature'] = None
        working_data = dataset.data

    # --- POST ---
    if request.method == 'POST':
        # User selects feature from left side
        selected_feature = request.POST.get('feature')
        if selected_feature:
            request.session['selected_feature'] = selected_feature
        else:
            request.session['selected_feature'] = None

        return redirect('mlstudio:preprocess')

        # --- GET ---
    try:
        full_df = pd.read_csv(io.StringIO(working_data))
    except Exception as e:
        messages.error(request, f"Błąd odczytu danych roboczych: {e}. Resetowanie do oryginału.")
        request.session['working_data'] = request.session['original_data']
        request.session.pop('selected_feature', None)
        full_df = pd.read_csv(io.StringIO(request.session['original_data']))

    data_for_table = full_df.head().values.tolist()
    columns = full_df.columns.tolist()

    context = {
        "dataset": dataset,
        "data": data_for_table,
        "columns": columns,
        "preprocessing_history": request.session.get('preprocessing_history', []),
        "selected_feature_name": None,
        "selected_feature_metadata": None,
    }

    # --- Download feature form session and make preprocessing  ---
    selected_feature_name = request.session.get('selected_feature')

    if selected_feature_name and selected_feature_name in full_df.columns:
        col_series = full_df[selected_feature_name]
        col_type = col_series.dtype
        is_numeric = pd.api.types.is_numeric_dtype(col_type)

        metadata = {
            'name': selected_feature_name,
            'dtype': str(col_type),
            'is_numeric': is_numeric,
            'is_categorical': not is_numeric,
            'missing_count': int(col_series.isnull().sum())
        }
        context['selected_feature_name'] = selected_feature_name
        context['selected_feature_metadata'] = metadata

    return render(request, "preprocess.html", context)


def preprocess_apply(request):
    """
    View to Post operations to apply preprocessing
    """
    if request.method != 'POST':
        return redirect('mlstudio:preprocess')

    working_data = request.session.get('working_data')
    selected_feature = request.session.get('selected_feature')

    if not working_data:
        messages.error(request, "Brak danych roboczych w sesji. Spróbuj odświeżyć stronę.")
        return redirect('mlstudio:preprocess')

    if not selected_feature:
        messages.error(request, "Nie wybrano żadnej cechy! Wybierz cechę z listy po lewej.")
        return redirect('mlstudio:preprocess')

    df = pd.read_csv(io.StringIO(working_data))

    if selected_feature not in df.columns:
        messages.error(request, f"Wybrana cecha '{selected_feature}' (z sesji) nie istnieje w danych.")
        request.session['selected_feature'] = None
        return redirect('mlstudio:preprocess')

    operation = request.POST.get('apply_operation')  # 'impute', 'encode', 'scale'
    history_entry = ""

    try:
        if operation == 'impute':
            method = request.POST.get('imputation_method')
            if not method:
                messages.warning(request, "Nie wybrano metody imputacji.")
                return redirect('mlstudio:preprocess')

            if method == 'mean':
                fill_value = df[selected_feature].mean()
                df[selected_feature].fillna(fill_value, inplace=True)
            elif method == 'median':
                fill_value = df[selected_feature].median()
                df[selected_feature].fillna(fill_value, inplace=True)
            elif method == 'mode':
                fill_value = df[selected_feature].mode()[0]
                df[selected_feature].fillna(fill_value, inplace=True)
            history_entry = f"Wypełniono braki w '{selected_feature}' metodą: {method}"

        elif operation == 'encode':
            method = request.POST.get('encoding_method')
            if not method:
                messages.warning(request, "Nie wybrano metody kodowania.")
                return redirect('mlstudio:preprocess')

            if method == 'label_encoder':
                le = LabelEncoder()
                df[selected_feature] = le.fit_transform(df[selected_feature].astype(str))
            history_entry = f"Zastosowano '{method}' na kolumnie '{selected_feature}'"

        elif operation == 'scale':
            method = request.POST.get('scaling_method')
            if not method:
                messages.warning(request, "Nie wybrano metody skalowania.")
                return redirect('mlstudio:preprocess')

            if method == 'standardization':
                scaler = StandardScaler()
                df[[selected_feature]] = scaler.fit_transform(df[[selected_feature]])
            elif method == 'normalization':
                scaler = MinMaxScaler()
                df[[selected_feature]] = scaler.fit_transform(df[[selected_feature]])
            history_entry = f"Zastosowano '{method}' na kolumnie '{selected_feature}'"

        else:
            messages.warning(request, "Nieznana operacja.")
            return redirect('mlstudio:preprocess')

        # Saves new data to session
        new_working_data = io.StringIO()
        df.to_csv(new_working_data, index=False)
        request.session['working_data'] = new_working_data.getvalue()

        # Add preprocessing to history
        history = request.session.get('preprocessing_history', [])
        history.append(history_entry)
        request.session['preprocessing_history'] = history

        messages.success(request, history_entry)

    except Exception as e:
        messages.error(request, f"Wystąpił błąd podczas operacji na '{selected_feature}': {e}")

    return redirect('mlstudio:preprocess')


def preprocess_reset(request):
    """
    View to reset a dataset in session to dataset from FB
    """
    original_data = request.session.get('original_data')
    if original_data:
        request.session['working_data'] = original_data
        request.session['preprocessing_history'] = []
        request.session['selected_feature'] = None
        messages.info(request, "Przywrócono oryginalny stan datasetu.")

    return redirect('mlstudio:preprocess')