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
                Dataset.objects.create(
                    name=uploaded_file.name,
                    data=df.to_csv(index=False),
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


def dataset_to_dataframe(dataset):
    """Convert Dataset object to pandas DataFrame."""
    # Reading CSV stored as text in the database
    return pd.read_csv(io.StringIO(dataset.data))


@login_required
def set_target(request, dataset_id):
    """Set target column, test size, and random seed."""
    dataset = get_object_or_404(Dataset, id=dataset_id, user=request.user)
    df = dataset_to_dataframe(dataset)
    columns = list(df.columns)

    error = None
    test_size = dataset.test_size or 0.2
    random_state = dataset.random_state or 42

    if request.method == "POST":
        learning_type = request.POST.get("learning_type", "CLASSIFICATION")
        target_col = request.POST.get("target_column")

        try:
            # Parse test_size and random_state from POST, fallback to defaults
            test_size = float(request.POST.get("test_size", test_size))
            random_state = int(request.POST.get("random_state", random_state))
            if not 0.1 <= test_size <= 0.9:
                raise ValueError("Rozmiar zbioru testowego musi być między 0.1 a 0.9.")
        except ValueError as e:
            error = f"Nieprawidłowe parametry podziału: {e}"

        # Validate target column for supervised learning
        if not error and learning_type in ['REGRESSION', 'CLASSIFICATION']:
            if not target_col:
                error = "Kolumna decyzyjna jest wymagana dla Regresji i Klasyfikacji."
            elif target_col not in columns:
                error = "Wybrana kolumna decyzyjna jest nieprawidłowa."
            else:
                dataset.target_column = target_col
        else:
            dataset.target_column = None  # unsupervised learning does not need target

        if not error:
            dataset.learning_type = learning_type
            dataset.test_size = test_size
            dataset.random_state = random_state
            dataset.save()

            try:
                # Decide whether to stratify: only for classification and if target has >1 unique values
                stratify_col = None
                if learning_type == 'CLASSIFICATION' and target_col and df[target_col].nunique() > 1:
                    stratify_col = df[target_col]
                elif learning_type == 'CLASSIFICATION' and target_col:
                    messages.warning(request, f"Nie można wykonać stratyfikacji: kolumna '{target_col}' ma tylko jedną unikalną wartość.")

                # Split dataset
                train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=stratify_col )

                # Store CSVs in session for later processing
                for key, data in {
                    'train_data': train_df,
                    'test_data': test_df,
                    'original_train_data': train_df,
                    'original_test_data': test_df
                }.items():
                    request.session[key] = data.to_csv(index=False)

                # Additional session info for preprocessing
                request.session.update({
                    'original_data': dataset.data,
                    'dataset_id': dataset.id,
                    'preprocessing_history': [],
                    'selected_feature': None
                })

                messages.success(request, f"Konfiguracja zapisana. Dane podzielone na zbiór treningowy ({len(train_df)} wierszy) i testowy ({len(test_df)} wierszy).")
                return redirect("loadData:load_data")
            except Exception as e:
                error = f"Błąd podczas podziału danych: {e}"

    context = {
        "dataset": dataset,
        "columns": columns,
        "learning_type_choices": LEARNING_TYPE_CHOICES,
        "error": error,
        "default_test_size": test_size,
        "default_random_state": random_state
    }
    return render(request, "decisionColumn.html", context)



@login_required
def delete_dataset(request, dataset_id):
    """Delete dataset by ID if POST request."""
    if request.method == 'POST':
        Dataset.objects.filter(id=dataset_id).delete()  # safe deletion
    return redirect('loadData:load_data')

# Simple view to render a contact page
def contact(request):
    """Handle contact form submission."""
    context = {}
    if request.method == 'POST':
        name, email_from_user, message = map(request.POST.get, ['name', 'email', 'message'])
        if all([name, email_from_user, message]):
            try:
                # Construct email and send
                email_msg = EmailMessage(
                    subject=f'Error on site from {name}',
                    body=f"Od: {name}\nEmail zwrotny: {email_from_user}\n\nTreść wiadomości:\n{message}",
                    from_email=settings.EMAIL_HOST_USER,
                    to=['sebastian.wandzel@uekat.edu.pl'],
                    reply_to=[email_from_user]
                )
                email_msg.send()
                context['success'] = True
            except Exception as e:
                print(f"BŁĄD WYSYŁANIA EMAILA: {e}")  # logging error
                context['error'] = 'Failed to send email.'
        else:
            context['error'] = 'Please fill in all fields.'
    return render(request, 'contact.html', context)

# Simple view to render an about page
def about(request):
    """Render the about page."""
    return render(request, "about.html")


