from .utils import load_data_from_sesion
from django.shortcuts import render
import pandas as pd, io
from loadData.models import Dataset

def preprocess(request):
    dataset = load_data_from_sesion(request)
    if not isinstance(dataset, Dataset):
        return dataset

    df = pd.read_csv(io.StringIO(dataset.data))

    data = df.head().values.tolist()
    columns = df.columns.tolist()

    return render(request, "preprocess.html", {
        "dataset": dataset,
        "data": data,
        "columns": columns})