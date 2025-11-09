from django.shortcuts import render, redirect, get_object_or_404
from .models import Dataset, LEARNING_TYPE_CHOICES
import pandas as pd
import io
from django.db import IntegrityError  # Correct import for catching integrity errors
from django.contrib.auth.decorators import login_required
from django.core.mail import EmailMessage
import mimetypes

from django.contrib import messages
from django.conf import settings

from sklearn.model_selection import train_test_split

def validate_csv_file(uploaded_file):
    """
    Checks if the uploaded file is a valid CSV.
    Returns a tuple: (df, error), where df is the loaded DataFrame or None,
    and error is an error message string or None.
    """
    #  Check file extension
    if not uploaded_file.name.lower().endswith('.csv'):
        return None, "File must have a .csv extension."

    # Check MIME type
    mime_type, _ = mimetypes.guess_type(uploaded_file.name)
    if mime_type not in ['text/csv', 'application/vnd.ms-excel', 'text/plain', None]:
        return None, "Invalid file type. Only CSV files are allowed."

    # Try reading the file content as CSV
    try:
        file_data = uploaded_file.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(file_data))
    except Exception:
        return None, "The file is not a valid CSV."

    return df, None


@login_required
def load_data(request):
    """
    View function to load datasets and render the index page.
    """
    error = None

    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']

        # Validate uploaded CSV file
        df, error = validate_csv_file(uploaded_file)

        if not error:
            try:
                # Convert DataFrame back to CSV text for database storage
                csv_text = df.to_csv(index=False)
                Dataset.objects.create(
                    name=uploaded_file.name,
                    data=csv_text,
                    user=request.user,
                )
                return redirect('loadData:load_data')
            except IntegrityError:
                error = f"Zestaw danych o tej nazwie {uploaded_file.name} już istnieje."
            except Exception as e:
                error = f"Błąd podczas przetwarzania pliku: {str(e)}"

    try:
        # Filter datasets by current user
        datasets = Dataset.objects.filter(user=request.user).order_by('-id')
    except Exception as e:
        datasets = []
        error = f"Błąd podczas pobierania danych: {str(e)}"

    return render(request, 'loadData.html', {"error": error, "datasets": datasets})

@login_required
def set_target(request, dataset_id):
    """
    Render of site to set target column for a dataset.
    """
    dataset = get_object_or_404(Dataset, id=dataset_id, user=request.user)
    df = dataset_to_dataframe(dataset)
    columns = list(df.columns)
    error = None

    default_test_size = 0.2
    default_random_state = 42

    if request.method == "POST":
        learning_type = request.POST.get("learning_type") or 'CLASSIFICATION'
        target_col = request.POST.get("target_column")

        try:
            test_size_str = request.POST.get("test_size", str(default_test_size))
            test_size = float(test_size_str)
            if not (0.1 <= test_size <= 0.9):
                raise ValueError("Proporcja zbioru testowego musi być między 0.1 a 0.9.")

            random_state_str = request.POST.get("random_state", str(default_random_state))
            random_state = int(random_state_str)

            default_test_size = test_size
            default_random_state = random_state

        except ValueError as e:
            error = f"Nieprawidłowe parametry podziału: {e}"

        # Validation logic
        if learning_type in ['REGRESSION', 'CLASSIFICATION']:
            if not target_col:
                error = "Kolumna decyzyjna jest wymagana dla Regresji i Klasyfikacji."
            elif target_col not in columns:
                error = "Wybrana kolumna decyzyjna jest nieprawidłowa."
            else:
                dataset.target_column = target_col

        elif learning_type in ['CLUSTERING', 'DIM_REDUCTION']:
            dataset.target_column = None
            target_col = None


        # Save if no errors
        if not error:
            dataset.learning_type = learning_type
            dataset.save()
            try:
                # 1. check if stratification is needed
                stratify_col = None
                if learning_type == 'CLASSIFICATION' and target_col:
                    if df[target_col].nunique() > 1:
                        stratify_col = df[target_col]
                    else:
                        messages.warning(request,f"Nie można wykonać stratyfikacji: kolumna '{target_col}' ma tylko jedną unikalną wartość.")

                train_df, test_df = train_test_split(df,test_size=test_size,  random_state=random_state,  stratify=stratify_col)

                train_csv = train_df.to_csv(index=False)
                test_csv = test_df.to_csv(index=False)

                request.session['train_data'] = train_csv
                request.session['test_data'] = test_csv

                # copy for reset purposes
                request.session['original_train_data'] = train_csv
                request.session['original_test_data'] = test_csv

                request.session['original_data'] = dataset.data
                request.session['dataset_id_for_data'] = dataset.id
                request.session['dataset_id'] = dataset.id

                request.session['preprocessing_history'] = []
                request.session['selected_feature'] = None

                messages.success(request,f"Konfiguracja zapisana. Dane podzielone na zbiór treningowy ({len(train_df)} wierszy) i testowy ({len(test_df)} wierszy).")

                return redirect("loadData:load_data")

            except Exception as e:
                error = f"Błąd podczas podziału danych: {e}"

    context = {
        "dataset": dataset,
        "columns": columns,
        "learning_type_choices": LEARNING_TYPE_CHOICES,
        "error": error,
        "default_test_size": default_test_size,
        "default_random_state": default_random_state
    }
    return render(request, "decisionColumn.html", context)


def dataset_to_dataframe(dataset):
    """
    Converts the 'data' field (CSV as text) to a pandas DataFrame.
    """
    return pd.read_csv(io.StringIO(dataset.data))

@login_required
def delete_dataset(request, dataset_id):
    """
    View for deleting a dataset by its ID.
    """
    if request.method == 'POST':
        try:
            dataset = Dataset.objects.get(id=dataset_id)
            dataset.delete()
        except Dataset.DoesNotExist:
            pass  # Optionally handle not found
    return redirect('loadData:load_data')

# Simple view to render a contact page
def contact(request):
    context = {}
    if request.method == 'POST':
        name = request.POST.get('name')
        email_from_user = request.POST.get('email')
        message = request.POST.get('message')

        if name and email_from_user and message:
            try:
                full_message = f"""
                Nowa wiadomość ze strony AI Analyzer:

                Od: {name}
                Email zwrotny: {email_from_user}

                Treść wiadomości:
                {message}
                """

                subject = f'Błąd na stronie od {name}'

                email = EmailMessage(subject,full_message,settings.EMAIL_HOST_USER,['sebastian.wandzel@uekat.edu.pl'],reply_to=[email_from_user])
                email.send()

                context['success'] = True

            except Exception as e:
                # Dobra praktyka: logowanie błędu
                print(f"BŁĄD WYSYŁANIA EMAILA: {e}")
                context['error'] = 'Nie udało się wysłać wiadomości. Spróbuj później.'
        else:
            context['error'] = 'Wypełnij wszystkie pola.'

    return render(request, 'contact.html', context)

# Simple view to render an about page
def about(request):
    return render(request, "about.html")


