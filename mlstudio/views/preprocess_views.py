from django.shortcuts import render, redirect
from django.contrib import messages
from .utils import load_data_from_session
import pandas as pd
import io
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, OrdinalEncoder
from loadData.models import Dataset
import numpy as np

pol_names = {
    'mean': 'Średnia',
    'median': 'Mediana',
    'mode': 'Najczęstsza wartość',
    'label_encoder': 'Kodowanie etykiet',
    'one_hot_encoder': 'Kodowanie binarne (One-Hot)',
    'standardization': 'Standaryzacja (Z-score)',
    'normalization': 'Normalizacja (Min-Max)'
}

def preprocess(request):
    """
    GET: Show user a page
    POST: Saves new features in session
    """
    dataset = load_data_from_session(request)
    if not isinstance(dataset, Dataset):
        return dataset

    # --- POST ---
    if request.method == 'POST':
        selected_feature = request.POST.get('feature')
        if selected_feature:
            request.session['selected_feature'] = selected_feature
        else:
            request.session['selected_feature'] = None

        return redirect('mlstudio:preprocess')

    # --- GET ---
    try:
        full_df = pd.read_csv(io.StringIO(dataset.data))
    except Exception as e:
        messages.error(request, f"Błąd odczytu danych treningowych z sesji: {e}.")
        return redirect("loadData:load_data")

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

    dataset = load_data_from_session(request)
    if not isinstance(dataset, Dataset):
        return dataset

    # load train and test data from session
    train_data_csv = request.session.get('train_data')
    test_data_csv = request.session.get('test_data')
    selected_feature = request.session.get('selected_feature')

    if not train_data_csv or not test_data_csv:
        messages.error(request,"Brak danych treningowych lub testowych w sesji. Spróbuj ponownie skonfigurować zadanie.")
        return redirect('loadData:set_target', dataset_id=dataset.id)

    if not selected_feature:
        messages.error(request, "Nie wybrano żadnej cechy! Wybierz cechę z listy po lewej.")
        return redirect('mlstudio:preprocess')

    df_train = pd.read_csv(io.StringIO(train_data_csv))
    df_test = pd.read_csv(io.StringIO(test_data_csv))

    if selected_feature not in df_train.columns:
        messages.error(request, f"Wybrana cecha '{selected_feature}' (z sesji) nie istnieje w danych.")
        request.session['selected_feature'] = None
        return redirect('mlstudio:preprocess')

    operation = request.POST.get('apply_operation')
    history_entry = ""

    try:
        if operation == 'impute':
            method = request.POST.get('imputation_method')
            if not method:
                messages.warning(request, "Nie wybrano metody imputacji.")
                return redirect('mlstudio:preprocess')

            if method == 'mean':
                imputer = SimpleImputer(strategy='mean')
            elif method == 'median':
                imputer = SimpleImputer(strategy='median')
            elif method == 'mode':
                imputer = SimpleImputer(strategy='most_frequent')
            else:
                messages.warning(request, "Nieznana metoda imputacji.")
                return redirect('mlstudio:preprocess')

            # Fit fir ONLY on training data
            imputer.fit(df_train[[selected_feature]])

            # transform BOTH datasets
            df_train[selected_feature] = imputer.transform(df_train[[selected_feature]])
            df_test[selected_feature] = imputer.transform(df_test[[selected_feature]])

            history_entry = f"Wypełniono braki w '{selected_feature}' metodą: {pol_names.get(method, method)}"

        elif operation == 'encode':
            method = request.POST.get('encoding_method')
            if not method:
                messages.warning(request, "Nie wybrano metody kodowania.")
                return redirect('mlstudio:preprocess')

            if method == 'label_encoder':
                if method == 'label_encoder':
                    # ordinary encoder but its makes everything that same as LabelEncoder
                    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)

                    df_train[[selected_feature]] = enc.fit_transform(df_train[[selected_feature]].astype(str))

                    df_test[[selected_feature]] = enc.transform(df_test[[selected_feature]].astype(str))

                    if df_test[selected_feature].isnull().any():
                        messages.warning(request,f"Wykryto nowe etykiety w '{selected_feature}' w zbiorze testowym — zastąpiono je najczęstszą wartością.")
                        imputer_for_nan = SimpleImputer(strategy='most_frequent')
                        imputer_for_nan.fit(df_train[[selected_feature]])
                        df_test[selected_feature] = imputer_for_nan.transform(df_test[[selected_feature]])


            elif method == 'one_hot_encoder':
                ohe = OneHotEncoder(handle_unknown='ignore', drop=None, sparse_output=False, dtype=int)

                ohe.fit(df_train[[selected_feature]])

                new_col_names = ohe.get_feature_names_out([selected_feature])

                transformed_train = ohe.transform(df_train[[selected_feature]])
                transformed_test = ohe.transform(df_test[[selected_feature]])

                df_train_new_cols = pd.DataFrame(transformed_train, columns=new_col_names, index=df_train.index)
                df_test_new_cols = pd.DataFrame(transformed_test, columns=new_col_names, index=df_test.index)

                df_train = pd.concat([df_train.drop(columns=[selected_feature]), df_train_new_cols], axis=1)
                df_test = pd.concat([df_test.drop(columns=[selected_feature]), df_test_new_cols], axis=1)

            history_entry = f"Zastosowano '{pol_names.get(method, method)}' na kolumnie '{selected_feature}'"
        elif operation == 'scale':
            method = request.POST.get('scaling_method')
            if not method:
                messages.warning(request, "Nie wybrano metody skalowania.")
                return redirect('mlstudio:preprocess')

            if method == 'standardization':
                scaler = StandardScaler()
            elif method == 'normalization':
                scaler = MinMaxScaler()
            else:
                messages.warning(request, "Nieznana metoda skalowania.")
                return redirect('mlstudio:preprocess')

            # fit ONLY on training data
            scaler.fit(df_train[[selected_feature]])

            # transform BOTH datasets
            df_train[selected_feature] = scaler.transform(df_train[[selected_feature]])
            df_test[selected_feature] = scaler.transform(df_test[[selected_feature]])

            history_entry = f"Zastosowano '{pol_names.get(method, method)}' na kolumnie '{selected_feature}'"

        elif operation == 'delete':
            if selected_feature == dataset.target_column:
                messages.error(request,
                               f"Nie można usunąć kolumny '{selected_feature}', ponieważ jest ona ustawiona jako kolumna docelowa.")
                return redirect('mlstudio:preprocess')
            else:
                # delete column from BOTH datasets
                df_train.drop(columns=[selected_feature], inplace=True)
                df_test.drop(columns=[selected_feature], inplace=True)
                history_entry = f"Usunięto kolumnę: '{selected_feature}'"
                request.session['selected_feature'] = None

        else:
            messages.warning(request, "Nieznana operacja.")
            return redirect('mlstudio:preprocess')

        # Save modified datasets back to session
        new_train_data = io.StringIO()
        df_train.to_csv(new_train_data, index=False)
        request.session['train_data'] = new_train_data.getvalue()

        # save test data back to session
        new_test_data = io.StringIO()
        df_test.to_csv(new_test_data, index=False)
        request.session['test_data'] = new_test_data.getvalue()

        history = request.session.get('preprocessing_history', [])
        history.append(history_entry)
        request.session['preprocessing_history'] = history

        messages.success(request, history_entry)

    except Exception as e:
        messages.error(request, f"Wystąpił błąd podczas operacji na '{selected_feature}': {e}")

        request.session['train_data'] = train_data_csv
        request.session['test_data'] = test_data_csv

    return redirect('mlstudio:preprocess')


def preprocess_reset(request):
    """
    View to reset a dataset in session to dataset from FB
    """
    original_train = request.session.get('original_train_data')
    original_test = request.session.get('original_test_data')

    dataset_id = request.session.get('dataset_id')

    if original_train and original_test:
        request.session['train_data'] = original_train
        request.session['test_data'] = original_test

        request.session['preprocessing_history'] = []
        request.session['selected_feature'] = None

        messages.info(request,"Przywrócono stan danych sprzed preprocessingu. Podział na zbiór treningowy i testowy został zachowany.")

    else:
        # Fallback if original data not in session
        messages.error(request,"Błąd resetowania. Brak oryginalnych danych podziału w sesji. Przekierowuję do ponownej konfiguracji.")
        if dataset_id:
            return redirect("loadData:set_target", dataset_id=dataset_id)
        return redirect("loadData:load_data")

    return redirect('mlstudio:preprocess')