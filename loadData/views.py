from django.shortcuts import render, redirect
from .models import Dataset
import pandas as pd
import io

# Views for loading and processing datasets


def load_data(request):
    """
    View function to load datasets and render the index page.
    """
    error = None
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        try:
            # Read file content as text and load into pandas DataFrame
            file_data = uploaded_file.read().decode('utf8')
            df = pd.read_csv(io.StringIO(file_data))
            # Save DataFrame as CSV text to the database
            csv_text = df.to_csv(index=False)
            Dataset.objects.create(
                name=uploaded_file.name,
                data=csv_text,
            )
            return redirect('load_data')
        except Exception as e:
            error = f"Błąd podczas przetwarzania pliku: {str(e)}"

    try:
        datasets = Dataset.objects.all()
    except Exception as e:
        datasets = []
        error = str(e)

    return render(request, 'index.html', {"error": error, "datasets": datasets})


def dataset_to_dataframe(dataset):
    """
    Converts the 'data' field (CSV as text) to a pandas DataFrame.
    """
    return pd.read_csv(io.StringIO(dataset.data))
