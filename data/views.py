"""Data management views."""
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.db import IntegrityError
from django.conf import settings
from django.core.mail import EmailMessage

from core.enums import ProblemType
from .models import Dataset
from .services import (
    validate_csv_file,
    save_uploaded_file,
    load_dataset_dataframe,
    set_target_column,
)
from preprocessing.models import PreprocessingPipeline
from preprocessing.services import get_or_create_active_pipeline


PROBLEM_TYPE_CHOICES = [
    (ProblemType.CLASSIFICATION, 'Klasyfikacja (Nadzorowane)'),
    (ProblemType.REGRESSION, 'Regresja (Nadzorowane)'),
    (ProblemType.CLUSTERING, 'Klasteryzacja (Nienadzorowane)'),
    (ProblemType.DIMENSIONALITY_REDUCTION, 'Redukcja Wymiarowości (Nienadzorowane)'),
]


@login_required
def load_data(request):
    """Load datasets and render the index page."""
    error = None
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        df, error = validate_csv_file(uploaded_file)
        if not error:
            try:
                save_uploaded_file(request.user, uploaded_file, df)
                return redirect('data:load_data')
            except IntegrityError:
                error = f"Zestaw danych o tej nazwie {uploaded_file.name} już istnieje."
            except Exception as e:
                error = f"Błąd podczas przetwarzania pliku: {str(e)}"

    datasets = Dataset.objects.filter(user=request.user, is_archived=False).order_by('-created_at')
    return render(request, 'loadData.html', {"error": error, "datasets": datasets})


@login_required
def set_target(request, dataset_id):
    """Set target column, problem type, and create pipeline with split config."""
    dataset = get_object_or_404(Dataset, dataset_id=dataset_id, user=request.user)
    df = load_dataset_dataframe(dataset)
    columns = list(df.columns)
    error = None
    test_size = 0.2
    random_state = 42

    if request.method == "POST":
        raw_type = request.POST.get("problem_type") or request.POST.get("learning_type")
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
        target_col = request.POST.get("target_column")
        try:
            test_size = float(request.POST.get("test_size", test_size))
            random_state = int(request.POST.get("random_state", random_state))
            if not 0.1 <= test_size <= 0.9:
                raise ValueError("Rozmiar zbioru testowego musi być między 0.1 a 0.9.")
        except ValueError as e:
            error = f"Nieprawidłowe parametry podziału: {e}"

        if problem_type in [ProblemType.REGRESSION, ProblemType.CLASSIFICATION]:
            if not target_col:
                error = "Kolumna decyzyjna jest wymagana dla Regresji i Klasyfikacji."
            elif target_col not in columns:
                error = "Wybrana kolumna decyzyjna jest nieprawidłowa."

        if not error:
            dataset.problem_type = problem_type
            dataset.save()
            set_target_column(dataset, target_col)

            split_config = {'test_size': test_size, 'random_state': random_state}
            pipeline = get_or_create_active_pipeline(dataset, split_config)
            pipeline.split_config = split_config
            pipeline.save()

            request.session['dataset_id'] = str(dataset.dataset_id)
            request.session['pipeline_id'] = pipeline.id
            request.session['split_config'] = split_config

            messages.success(request, f"Konfiguracja zapisana. Możesz przejść do ML Studio.")
            return redirect("data:load_data")

    context = {
        "dataset": dataset,
        "columns": columns,
        "problem_type_choices": PROBLEM_TYPE_CHOICES,
        "error": error,
        "default_test_size": test_size,
        "default_random_state": random_state,
    }
    return render(request, "decisionColumn.html", context)


@login_required
def delete_dataset(request, dataset_id):
    """Soft delete dataset (archive)."""
    if request.method == 'POST':
        dataset = get_object_or_404(Dataset, dataset_id=dataset_id, user=request.user)
        dataset.is_archived = True
        dataset.save()
    return redirect('data:load_data')


def contact(request):
    """Handle contact form submission."""
    context = {}
    if request.method == 'POST':
        name = request.POST.get('name')
        email_from_user = request.POST.get('email')
        message = request.POST.get('message')
        if all([name, email_from_user, message]):
            try:
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
                print(f"BŁĄD WYSYŁANIA EMAILA: {e}")
                context['error'] = 'Failed to send email.'
        else:
            context['error'] = 'Please fill in all fields.'
    return render(request, 'contact.html', context)


def about(request):
    """Render the about page."""
    return render(request, "about.html")
