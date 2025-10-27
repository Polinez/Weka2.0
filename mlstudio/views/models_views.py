from .utils import load_data_from_sesion
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.contrib import messages
from ..models import MLModel, CommonParameter, DatasetModelState
from loadData.models import Dataset


def convert_value(value_str, data_type):
    try:
        if data_type == 'int':
            return int(value_str)
        elif data_type == 'float':
            value_str = value_str.replace(',', '.')
            return float(value_str)
        elif data_type == 'bool':
            return value_str.lower() == 'true'
        elif data_type == 'str':
            return value_str
    except (ValueError, TypeError, AttributeError):
        raise ValueError(f"Niepoprawna wartość '{value_str}' dla typu {data_type}")

@login_required()
def models(request):
    dataset = load_data_from_sesion(request)
    if not isinstance(dataset, Dataset):
        return dataset

    user = request.user
    models_list = MLModel.objects.all()
    selected_model_id = request.POST.get("selected_model")
    selected_model = None

    common_param_definitions = {p.name: p for p in CommonParameter.objects.all()}

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
            default_model_params_defs = {p.name: p.value for p in selected_model.parameters.all()}
            default_common_params_defs = {p.name: p.value for p in common_param_definitions.values()}

            final_model_params = {}
            for p in selected_model.parameters.all():
                final_model_params[p.name] = convert_value(p.value, p.data_type)

            final_common_params = {}
            for p in common_param_definitions.values():
                final_common_params[p.name] = convert_value(p.value, p.data_type)

            common_params_to_save = saved_default_parameters or final_common_params

            # Saves initial state to DB
            DatasetModelState.objects.update_or_create(
                dataset=dataset,
                user=user,
                defaults={
                    "model": selected_model,
                    "default_parameters": common_params_to_save,
                    "parameters": final_model_params
                }
            )
            return redirect("mlstudio:models")

    # Save parameters if button submitted
    if "save_all" in request.POST:
        if not selected_model:
            messages.error(request, "Proszę najpierw wybrać model przed zapisaniem ustawień.")
            return redirect("mlstudio:models")

        try:
            model_param_types = {p.name: p.data_type for p in selected_model.parameters.all()}
            common_param_types = {p.name: p.data_type for p in common_param_definitions.values()}

            posted_common_params = {}
            for k, v_str in request.POST.items():
                if k.startswith("common_"):
                    name = k.replace("common_", "")
                    data_type = common_param_types.get(name)
                    # convert string from form to correct type
                    posted_common_params[name] = convert_value(v_str, data_type)

            posted_model_params = {}
            for k, v_str in request.POST.items():
                if k.startswith("param_"):
                    name = k.replace("param_", "")
                    data_type = model_param_types.get(name)
                    # convert string from form to correct type
                    posted_model_params[name] = convert_value(v_str, data_type)

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
        except ValueError as e:
            messages.error(request, f"Błąd zapisu: {e}")
        return redirect("mlstudio:models")

    # logic for common_parameters
    common_parameters_for_template = []
    # Load saved values if exist
    common_values_to_display = saved_default_parameters or {}
    if not common_values_to_display and not selected_model:
        # if no saved and no model selected, take from DB defaults
        common_values_to_display = {p.name: p.value for p in common_param_definitions.values()}

    for definition in common_param_definitions.values():
        common_parameters_for_template.append({
            'name': definition.name,
            'value': common_values_to_display.get(definition.name, definition.value),
            'data_type': definition.data_type
        })

    # Logic for model_parameters
    model_parameters_for_template = []
    if selected_model:
        # Load saved values if exist
        model_values_to_display = saved_model_parameters or {}
        if not model_values_to_display:
            # if no saved and no model selected, take from DB defaults
            model_values_to_display = {p.name: p.value for p in selected_model.parameters.all()}

        for definition in selected_model.parameters.all():
            model_parameters_for_template.append({
                'name': definition.name,
                'value': model_values_to_display.get(definition.name, definition.value),
                'data_type': definition.data_type
            })

    return render(request, "models.html", {
        "dataset": dataset,
        "models_list": models_list,
        "selected_model": selected_model,
        "common_parameters": common_parameters_for_template,
        "model_parameters": model_parameters_for_template
    })