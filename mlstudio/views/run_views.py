from .utils import load_data_from_sesion
from django.contrib.auth.decorators import login_required
from loadData.models import Dataset
from django.shortcuts import render, redirect , get_object_or_404
from django.contrib import messages

from ..models import DatasetModelState, MLRun

from mlstudio.views.ml_models.run_ml_model import run_ml_model


@login_required()
def run_model(request):
    dataset = load_data_from_sesion(request)
    if not isinstance(dataset, Dataset):
        return dataset

    user = request.user

    # Import state from Db if exists
    try:
        state = DatasetModelState.objects.get(dataset=dataset, user=user)
    except DatasetModelState.DoesNotExist:
        messages.error(request, "Najpierw wybierz model w zakładce Models.")
        return redirect("mlstudio:models")

    # Load previous runs
    runs = MLRun.objects.filter(dataset=dataset, user=user).order_by('-created_at')

    # Get selected model and parameters
    selected_run = None
    displayed_common_params = {}
    displayed_model_params = {}
    model_to_display = None

    #if run clicked on previous runs
    run_id = request.GET.get("run_id")
    if run_id:
        selected_run = get_object_or_404(MLRun, id=run_id, user=user)
        displayed_common_params = selected_run.common_parameters
        displayed_model_params = selected_run.model_parameters
        model_to_display = selected_run.model
    elif state:
        displayed_common_params = state.default_parameters
        displayed_model_params = state.parameters
        model_to_display = state.model

    # run model button clicked
    if request.method == "POST" and "run_model" in request.POST:

        # run model with current parameters
        result = run_ml_model(
            dataset,
            state.model.name,
            state.default_parameters,
            state.parameters
        )
        # save model
        ml_run = MLRun.objects.create(
            dataset=dataset,
            user=user,
            model=state.model,
            common_parameters=state.default_parameters,
            model_parameters=state.parameters,
            result=result
        )
        messages.success(request, f"Model został uruchomiony. ID uruchomienia: {ml_run.id}")
        return redirect("mlstudio:run_model")

    return render(request, "run.html", {
        "dataset": dataset,
        "runs": runs,
        "selected_run": selected_run,
        "model_to_display": model_to_display,
        "displayed_common_params": displayed_common_params,
        "displayed_model_params": displayed_model_params,
    })
