from .utils import load_data_from_sesion
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from loadData.models import Dataset

@login_required()
def visualize(request):
    dataset = load_data_from_sesion(request)
    if not isinstance(dataset, Dataset):
        return dataset

    return render(request, "visualize.html", {
        "dataset": dataset
    })