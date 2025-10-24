from .utils import get_plot

from django.contrib.auth.decorators import login_required
from django.shortcuts import render, get_object_or_404, redirect
from django.views.decorators.http import require_POST
from loadData.models import Dataset
from django.contrib import messages
import pandas as pd
import io

from .models import MLModel, ModelParameter, CommonParameter, DatasetModelState


def load_data_from_sesion(request):
    dataset_id = request.session.get('dataset_id')  # Retrieve dataset_id from session

    if not dataset_id:
        messages.error(request, "Nie wybrano datasetu. Wybierz dataset lub załaduj nowy.")
        return redirect("loadData:loadData")

    dataset = get_object_or_404(Dataset, id=dataset_id, user=request.user)

    # if no target column, redirect to set target
    if not dataset.target_column:
        messages.error(request, "Brak ustawionej kolumny docelowej dla wybranego datasetu. Ustaw ją najpierw.")
        return redirect("loadData:set_target", dataset.id)

    return dataset


@login_required
@require_POST
def select_dataset(request, dataset_id):
    dataset = get_object_or_404(Dataset, id=dataset_id, user=request.user)

    # if no target column, redirect to set target
    if not dataset.target_column:
        messages.error(request, "Nie można wybrać datasetu bez ustawionej kolumny docelowej. Ustaw ją najpierw.")
        return redirect("loadData:set_target", dataset.id)

    request.session['dataset_id'] = dataset.id  # Store dataset_id in session
    return redirect("mlstudio:studio")

@login_required()
def studio(request):
    dataset = load_data_from_sesion(request)
    # if not dataset, load_data_from_session has already handled the response
    if not isinstance(dataset, Dataset):
        return dataset

    df = pd.read_csv(io.StringIO(dataset.data))  # convert to DataFrame
    data = df.head().values.tolist()
    columns = df.columns.tolist()

    # selected column from POST or session
    if request.method == "POST":
        selected_column = request.POST.get("selected_column")
        request.session["selected_column"] = selected_column
    else:
        selected_column = request.session.get("selected_column")

    # if not selected
    if not selected_column or selected_column not in columns:
        selected_column = dataset.target_column
        request.session["selected_column"] = selected_column

    # show everything if no column is selected
    graph = get_plot(df, columns=[selected_column])

    stats_df = df.describe()
    statistics = stats_df.reset_index().values.tolist()
    stat_columns = ["Statystyka"] + stats_df.columns.tolist()

    return render(request, "explore.html", {
        "dataset": dataset,
        "data": data,
        "columns": columns,
        "statistics": statistics,
        "stat_columns": stat_columns,
        "graph": graph,
        "selected_column": selected_column})


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

def models(request):
    dataset = load_data_from_sesion(request)
    if not isinstance(dataset, Dataset):
        return dataset

    user = request.user
    models_list = MLModel.objects.all()
    selected_model_id = request.POST.get("selected_model")
    selected_model = None

    default_common_parameters = {param.name: param.value for param in CommonParameter.objects.all()}

    # Import state from Db if exists
    try:
        state = DatasetModelState.objects.get(dataset=dataset, user=user)
        selected_model = state.model
        saved_default_parameters = state.default_parameters if isinstance(state.default_parameters, dict) else {}
        saved_model_parameters = state.parameters if isinstance(state.parameters, dict) else {}
    except DatasetModelState.DoesNotExist:
        state = None
        saved_default_parameters = {}
        saved_model_parameters = {}

    # SELECT MODEL from list handling if no state exists
    if selected_model_id and "save_all" not in request.POST:
        selected_model = MLModel.objects.filter(id=selected_model_id).first()
        if selected_model:
            # Loads default parameters for the selected model
            default_model_parameters = {p.name: p.value for p in selected_model.parameters.all()}

            # Saves initial state to DB
            DatasetModelState.objects.update_or_create(
                dataset=dataset,
                user=user,
                defaults={
                    "model": selected_model,
                    "default_parameters": default_common_parameters,
                    "parameters": default_model_parameters
                }
            )

            messages.info(request, f"Załadowano domyślne parametry dla modelu {selected_model.name}.")
            return redirect("mlstudio:models")

    # Save parameters if button submitted
    if "save_all" in request.POST:
        if not selected_model:
            messages.error(request, "Proszę najpierw wybrać model przed zapisaniem ustawień.")
            return redirect("mlstudio:models")

        posted_common_params = {
            k.replace("common_", ""): v for k, v in request.POST.items() if k.startswith("common_")
        }
        posted_model_params = {
            k.replace("param_", ""): v for k, v in request.POST.items() if k.startswith("param_")
        }

        DatasetModelState.objects.update_or_create(
            dataset=dataset,
            user=user,
            defaults={
                "model": selected_model,
                "default_parameters": posted_common_params,
                "parameters": posted_model_params
            }
        )

        messages.success(request, "Parametry zostały zapisane poprawnie ✅")
        return redirect("mlstudio:models")

    # Load parameters to display
    # if there are ssaved
    common_parameters = saved_default_parameters or default_common_parameters

    # if no saved model parameters
    if selected_model:
        model_parameters = saved_model_parameters or {p.name: p.value for p in selected_model.parameters.all()}
    else:
        model_parameters = {}

    return render(request, "models.html", {
        "dataset": dataset,
        "models_list": models_list,
        "selected_model": selected_model,
        "common_parameters": common_parameters,
        "model_parameters": model_parameters
    })

def run_model(request):
    dataset = load_data_from_sesion(request)
    if not isinstance(dataset, Dataset):
        return dataset

    return render(request, "run.html", {
        "dataset": dataset
    })

def visualize(request):
    dataset = load_data_from_sesion(request)
    if not isinstance(dataset, Dataset):
        return dataset

    return render(request, "visualize.html", {
        "dataset": dataset
    })
