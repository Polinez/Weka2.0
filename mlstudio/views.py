from django.contrib.auth.decorators import login_required
from django.shortcuts import render, get_object_or_404, redirect
from django.views.decorators.http import require_POST
from loadData.models import Dataset
import pandas as pd
import io

def load_data_from_sesion(request):
    dataset_id = request.session.get('dataset_id')  # Retrieve dataset_id from session

    if not dataset_id:
        return render(request, "loadData:loadData.html", {"error": "No dataset selected."})

    dataset = get_object_or_404(Dataset, id=dataset_id, user=request.user)
    return dataset


@login_required
@require_POST
def select_dataset(request, dataset_id):
    dataset = get_object_or_404(Dataset, id=dataset_id, user=request.user)
    request.session['dataset_id'] = dataset.id  # Store dataset_id in session
    return redirect("mlstudio:studio")


def studio(request):
    dataset = load_data_from_sesion(request)

    df = pd.read_csv(io.StringIO(dataset.data))  # konwersja do DataFrame

    preview = df.head().to_html()
    return render(request, "explore.html", {"dataset": dataset, "preview": preview})


def preprocess(request):
    dataset = load_data_from_sesion(request)

    return render(request, "preprocess.html", {"dataset": dataset})

def models(request):
    dataset = load_data_from_sesion(request)

    return render(request, "models.html", {"dataset": dataset})

def run_model(request):
    dataset = load_data_from_sesion(request)

    return render(request, "run.html", {"dataset": dataset})

def visualize(request):
    dataset = load_data_from_sesion(request)

    return render(request, "visualize.html", {"dataset": dataset})


