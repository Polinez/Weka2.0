from django.shortcuts import render, get_object_or_404
from loadData.models import Dataset
import pandas as pd
import io

def studio(request, dataset_id):
    dataset = get_object_or_404(Dataset, id=dataset_id, user=request.user)
    df = pd.read_csv(io.StringIO(dataset.data))  # konwersja do DataFrame

    preview = df.head().to_html()
    return render(request, "explore.html", {"dataset": dataset, "preview": preview})
