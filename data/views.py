"""Data management views."""

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.db import IntegrityError

from core.enums import ProblemType
from .models import Dataset

# Importujemy zaktualizowane serwisy
from .services import (
    validate_csv_file,
    save_uploaded_file,
    load_dataset_dataframe,
    configure_analysis_settings,
    archive_dataset_and_cleanup,
    restore_dataset_from_archive,
    send_contact_email,
)

PROBLEM_TYPE_CHOICES = [
    (ProblemType.CLASSIFICATION, "Klasyfikacja (Nadzorowane)"),
    (ProblemType.REGRESSION, "Regresja (Nadzorowane)"),
    (ProblemType.CLUSTERING, "Klasteryzacja (Nienadzorowane)"),
    (ProblemType.DIMENSIONALITY_REDUCTION, "Redukcja Wymiarowości (Nienadzorowane)"),
]


@login_required
def load_data(request):
    """Load datasets and render the index page."""
    error = None
    if request.method == "POST" and request.FILES.get("file"):
        uploaded_file = request.FILES["file"]
        df, error = validate_csv_file(uploaded_file)
        if not error:
            try:
                save_uploaded_file(request.user, uploaded_file, df)
                return redirect("data:load_data")
            except IntegrityError:
                error = f"Zestaw danych o nazwie '{uploaded_file.name}' już istnieje (sprawdź też archiwum poniżej)."
            except Exception as e:
                error = f"Błąd podczas przetwarzania pliku: {str(e)}"

    datasets = Dataset.objects.filter(user=request.user, is_archived=False).order_by(
        "-created_at"
    )
    archived_datasets = Dataset.objects.filter(
        user=request.user, is_archived=True
    ).order_by("-created_at")

    return render(
        request,
        "loadData.html",
        {"error": error, "datasets": datasets, "archived_datasets": archived_datasets},
    )


@login_required
def set_target(request, dataset_id):
    """Set target column using service logic."""
    dataset = get_object_or_404(Dataset, dataset_id=dataset_id, user=request.user)
    error = None

    if request.method == "POST":
        try:
            # Use service to configure analysis settings
            pipeline_id, split_config = configure_analysis_settings(
                dataset, request.POST
            )

            # session settings
            request.session["dataset_id"] = str(dataset.dataset_id)
            request.session["pipeline_id"] = pipeline_id
            request.session["split_config"] = split_config

            messages.success(
                request, "Konfiguracja zapisana. Możesz przejść do ML Studio."
            )
            return redirect("data:load_data")
        except ValueError as e:
            error = str(e)
        except Exception as e:
            error = f"Wystąpił nieoczekiwany błąd: {str(e)}"

    # Download dataframe to get columns for rendering
    df = load_dataset_dataframe(dataset)
    columns = list(df.columns)

    context = {
        "dataset": dataset,
        "columns": columns,
        "problem_type_choices": PROBLEM_TYPE_CHOICES,
        "error": error,
        "default_test_size": 0.2,
        "default_random_state": 42,
    }
    return render(request, "decisionColumn.html", context)


@login_required
def delete_dataset(request, dataset_id):
    """
    Soft delete dataset (archive) AND Hard delete related heavy artifacts.
    """
    if request.method == "POST":
        dataset = get_object_or_404(Dataset, dataset_id=dataset_id, user=request.user)
        archive_dataset_and_cleanup(dataset)
        messages.success(request, "Zbiór został przeniesiony do archiwum.")
    return redirect("data:load_data")


@login_required
def restore_dataset(request, dataset_id):
    """
    Restore dataset from archive (is_archived = False).
    """
    dataset = get_object_or_404(Dataset, dataset_id=dataset_id, user=request.user)
    if request.method == "POST":
        restore_dataset_from_archive(dataset)
        messages.success(request, f"Zbiór '{dataset.name}' został przywrócony.")
    return redirect("data:load_data")


def contact(request):
    """Handle contact form."""
    context = {}
    if request.method == "POST":
        name = request.POST.get("name", "").strip()
        email = request.POST.get("email", "").strip()
        message = request.POST.get("message", "").strip()

        success = send_contact_email(name=name, user_email=email, message=message)

        if success:
            context["success"] = True
        else:
            context["error"] = "Failed to send email."

    return render(request, "contact.html", context)


def about(request):
    """Render the about page."""
    return render(request, "about.html")
