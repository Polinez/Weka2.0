from django.shortcuts import render, redirect
from .models import Dataset
import pandas as pd
import io
from django.db import IntegrityError  # Correct import for catching integrity errors
from django.contrib.auth.decorators import login_required
from django.core.mail import send_mail

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# Views for loading and processing datasets


@login_required
def load_data(request):
    """
    View function to load datasets and render the index page.
    """
    error = None
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']

        # Check file extension before processing
        if not uploaded_file.name.endswith('.csv'):
            error = "Proszę przesłać plik CSV."
        else:
            try:
                # Read file content as text and load into pandas DataFrame
                file_data = uploaded_file.read().decode('utf8')
                df = pd.read_csv(io.StringIO(file_data))
                # Save DataFrame as CSV text to the database
                csv_text = df.to_csv(index=False)
                Dataset.objects.create(
                    name=uploaded_file.name,
                    data=csv_text,
                    user=request.user,  # Associate dataset with the logged-in user
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
    dataset = Dataset.objects.get(id=dataset_id, user=request.user)
    df = dataset_to_dataframe(dataset)
    columns = list(df.columns)

    if request.method == "POST":
        target_col = request.POST.get("target_column")
        if target_col in columns:
            dataset.target_column = target_col
            dataset.save()
            return redirect("loadData:load_data")
        else:
            error = "Nieprawidłowa kolumna"
            return render(request, "decisionColumn.html", {"dataset": dataset, "columns": columns, "error": error})

    return render(request, "decisionColumn.html", {"dataset": dataset, "columns": columns})


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
    return render(request, "contact.html")

# Simple view to render an about page
def about(request):
    return render(request, "about.html")


# send report mail
def contact_view(request):
    context = {}
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        message = request.POST.get('message')

        if name and email and message:
            try:
                send_mail(
                    f'Błąd na stronie od {name}',
                    message,
                    email,
                    ['sebastian.wandzel@uekat.edu.pl'],
                )
                context['success'] = True
            except Exception as e:
                context['error'] = 'Nie udało się wysłać wiadomości. Spróbuj później.'
        else:
            context['error'] = 'Wypełnij wszystkie pola.'

    return render(request, 'contact.html', context)