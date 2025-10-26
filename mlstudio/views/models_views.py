from .utils import load_data_from_sesion
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.contrib import messages
from ..models import MLModel, CommonParameter, DatasetModelState
from loadData.models import Dataset

@login_required()
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
    # if there are saved
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