"""Model selection and parameters views."""

from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.contrib import messages

from ml.models import MLModel
from .utils import load_dataset_and_pipeline_from_session

# Importing  services
from ml.services.models_service import (
    get_models_for_dataset,
    extract_parameters_from_post,
    prepare_model_params_for_display,
)


@login_required
def models(request):
    """View for selecting ML model and configuring hyperparameters."""
    result = load_dataset_and_pipeline_from_session(request)
    if not isinstance(result, tuple):
        return result
    dataset, pipeline = result

    # Checking if any problem type was selected
    if not dataset.problem_type:
        messages.error(request, "Ustaw najpierw typ problemu w Konfiguracji.")
        return redirect("data:set_target", dataset_id=dataset.dataset_id)

    #  Getting models available for this problem type
    models_list = get_models_for_dataset(dataset)

    #  Getting current model from session
    session_model_id = request.session.get("last_model_id")
    selected_model = None
    if session_model_id:
        selected_model = MLModel.objects.filter(id=session_model_id).first()

    # Default settings (from pipeline in database, to be consistent)
    defaults = pipeline.split_config or {"test_size": 0.2, "random_state": 42}

    # 4. Handling form (POST)
    if request.method == "POST":
        new_model_id = request.POST.get("selected_model")

        if new_model_id:
            selected_model = MLModel.objects.filter(id=new_model_id).first()

            #  Saving NEW model to session immediately
            request.session["last_model_id"] = new_model_id

            #  If it's a different model than the previous one, clear specific parameters in session
            if str(new_model_id) != str(session_model_id):
                request.session["last_params"] = {}

        # If user changed model in dropdown
        if new_model_id:
            selected_model = MLModel.objects.filter(id=new_model_id).first()
            # If it's a different model than the previous one, clear specific parameters in session
            if str(new_model_id) != str(session_model_id):
                request.session["last_params"] = {}

        # If "Save parameters" button was clicked
        if "save_all" in request.POST and selected_model:
            try:
                #  Parsing and validation
                extracted = extract_parameters_from_post(request.POST, selected_model)

                # Saving UI state in session
                request.session["last_model_id"] = selected_model.id
                request.session["last_params"] = {
                    "common_parameters": extracted["common"],
                    "model_parameters": extracted["model"],
                }

                # IMPORTANT: Updating pipeline in database (Single Source of Truth)
                pipeline.split_config = extracted["common"]
                pipeline.save()

                messages.success(request, "Konfiguracja modelu została zapisana.")
            except ValueError as e:
                messages.error(request, str(e))

        # Reload to avoid "Resubmit form" problem
        return redirect("ml:models")

    # 4. Preparing data to display (GET)
    session_params = request.session.get("last_params", {})

    #  Creating structures for HTML
    common_p, model_p = prepare_model_params_for_display(
        session_params, defaults, selected_model
    )

    return render(
        request,
        "models.html",
        {
            "dataset": dataset,
            "models_list": models_list,
            "selected_model": selected_model,
            "common_parameters": common_p,
            "model_parameters": model_p,
        },
    )
