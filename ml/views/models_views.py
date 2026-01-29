"""Model selection and parameters views."""
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.contrib import messages

from ml.models import MLModel, ModelParameterDef
from data.models import Dataset
from core.enums import ParamDataType
from .utils import load_dataset_and_pipeline_from_session


def convert_value(value_str, data_type):
    """Convert form string to Python type."""
    if value_str is None or str(value_str).strip() == "":
        if data_type in ['int', 'float']:
            return None
        if data_type == 'bool':
            return False
        return ""
    try:
        if data_type == 'int':
            return int(value_str)
        if data_type == 'float':
            return float(str(value_str).replace(',', '.'))
        if data_type == 'bool':
            return str(value_str).lower() == 'true'
        return str(value_str)
    except (ValueError, TypeError, AttributeError):
        raise ValueError(f"Nieprawidłowa wartość '{value_str}' dla typu {data_type}")


@login_required
def models(request):
    """Model and parameter selection. Session cache for last model/params."""
    result = load_dataset_and_pipeline_from_session(request)
    if not isinstance(result, tuple):
        return result
    dataset, pipeline = result

    if not dataset.problem_type:
        messages.error(request, "Ustaw typ problemu w Konfiguracja.")
        return redirect("data:set_target", dataset_id=dataset.dataset_id)

    models_list = MLModel.objects.filter(type=dataset.problem_type)
    split_config = request.session.get('split_config', {})
    common_params = {
        'test_size': split_config.get('test_size', 0.2),
        'random_state': split_config.get('random_state', 42),
    }

    session_model_id = request.session.get('last_model_id')
    session_params = request.session.get('last_params', {})
    selected_model = MLModel.objects.filter(id=session_model_id).first() if session_model_id else None
    displayed_common = session_params.get('common_parameters', common_params)
    displayed_model_params = session_params.get('model_parameters', {})

    if request.method != "POST":
        common_parameters = [
            {'name': 'test_size', 'value': displayed_common.get('test_size', 0.2), 'data_type': 'float', 'description': 'Proporcja zbioru testowego'},
            {'name': 'random_state', 'value': displayed_common.get('random_state', 42), 'data_type': 'int', 'description': 'Ziarno losowości'},
        ]
        model_parameters = []
        if selected_model:
            for pdef in selected_model.parameter_defs.all():
                val = displayed_model_params.get(pdef.name)
                if val is None:
                    val = convert_value(pdef.default_value, pdef.data_type)
                model_parameters.append({
                    'name': pdef.name,
                    'value': val,
                    'data_type': pdef.data_type,
                    'description': pdef.description,
                })

        return render(request, "models.html", {
            "dataset": dataset,
            "models_list": models_list,
            "selected_model": selected_model,
            "common_parameters": common_parameters,
            "model_parameters": model_parameters,
        })

    selected_model_id = request.POST.get("selected_model")
    selected_model = MLModel.objects.filter(id=selected_model_id).first() if selected_model_id else None

    if "save_all" in request.POST:
        try:
            common_parameters = {}
            common_parameters['test_size'] = convert_value(request.POST.get("common_test_size"), 'float') or 0.2
            common_parameters['random_state'] = convert_value(request.POST.get("common_random_state"), 'int') or 42

            model_parameters = {}
            if selected_model:
                for pdef in selected_model.parameter_defs.all():
                    raw = request.POST.get(f"param_{pdef.name}")
                    model_parameters[pdef.name] = convert_value(raw, pdef.data_type)

            request.session['last_model_id'] = selected_model.id if selected_model else None
            request.session['last_params'] = {
                'common_parameters': common_parameters,
                'model_parameters': model_parameters,
            }
            request.session['split_config'] = common_parameters
            messages.success(request, "Parametry zapisane w sesji.")
        except ValueError as e:
            messages.error(request, str(e))

    common_parameters = [
        {'name': 'test_size', 'value': request.POST.get("common_test_size", displayed_common.get('test_size', 0.2)), 'data_type': 'float', 'description': 'Proporcja zbioru testowego'},
        {'name': 'random_state', 'value': request.POST.get("common_random_state", displayed_common.get('random_state', 42)), 'data_type': 'int', 'description': 'Ziarno losowości'},
    ]
    model_parameters = []
    if selected_model:
        for pdef in selected_model.parameter_defs.all():
            val = request.POST.get(f"param_{pdef.name}")
            if val is None:
                val = convert_value(pdef.default_value, pdef.data_type)
            model_parameters.append({
                'name': pdef.name,
                'value': val,
                'data_type': pdef.data_type,
                'description': pdef.description,
            })

    return render(request, "models.html", {
        "dataset": dataset,
        "models_list": models_list,
        "selected_model": selected_model,
        "common_parameters": common_parameters,
        "model_parameters": model_parameters,
    })
