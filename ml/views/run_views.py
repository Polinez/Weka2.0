"""ML run views."""
import pandas as pd
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.views.decorators.http import require_POST

from data.models import Dataset
from ml.models import MLModel, MLRun
from preprocessing.models import PreprocessingPipeline
from preprocessing.services import get_train_test_dataframes
from ml.services.run_service import run_ml_experiment
from .utils import load_dataset_and_pipeline_from_session


@login_required
def run_model(request):
    """Run ML model and display results."""
    result = load_dataset_and_pipeline_from_session(request)
    if not isinstance(result, tuple):
        return result
    dataset, pipeline = result

    session_model_id = request.session.get('last_model_id')
    session_params = request.session.get('last_params', {})
    split_config = request.session.get('split_config', {})

    if not session_model_id:
        messages.error(request, "Najpierw wybierz model i parametry w zakładce Modele.")
        return redirect("ml:models")

    model = get_object_or_404(MLModel, id=session_model_id)
    common_params = session_params.get('common_parameters', {})
    model_params = session_params.get('model_parameters', {})
    split_config = split_config or common_params or {'test_size': 0.2, 'random_state': 42}

    runs = MLRun.objects.filter(dataset=dataset, user=request.user).order_by('-created_at')

    selected_run = None
    run_id = request.GET.get("run_id")
    if run_id:
        selected_run = MLRun.objects.filter(run_id=run_id, user=request.user).first()
        if selected_run:
            common_params = selected_run.split_config or {}
            model_params = selected_run.used_parameters.get('model_parameters', selected_run.used_parameters)
            model = selected_run.model

    displayed_common_params = common_params
    displayed_model_params = model_params
    model_to_display = model

    if request.method == "POST" and "run_model" in request.POST:
        train_test = get_train_test_dataframes(pipeline) if pipeline else None
        if not train_test:
            messages.error(request, "Brak danych. Skonfiguruj zadanie i ewentualnie preprocessing.")
            return redirect("data:set_target", dataset_id=dataset.dataset_id)

        df_train, df_test = train_test
        used_params = {
            'test_size': split_config.get('test_size', 0.2),
            'random_state': split_config.get('random_state', 42),
            'model_parameters': model_params,
        }

        result = run_ml_experiment(
            user=request.user,
            dataset=dataset,
            pipeline=pipeline,
            model=model,
            split_config=split_config,
            used_parameters=used_params,
        )

        if result.get('error'):
            messages.error(request, result['error'])
            return redirect("ml:run_model")

        eval_data = result.get('evaluation', {})
        plots_list = list(result.get('plots_base64', []))
        for key, val in eval_data.items():
            if key.startswith('plot_') and val:
                plots_list.append(val)

        metrics = result.get('metrics', {})
        metrics['plots_base64'] = plots_list

        ml_run = MLRun.objects.create(
            run_id=result['run_id'],
            user=request.user,
            dataset=dataset,
            pipeline=pipeline,
            model=model,
            status=result.get('status', 'Success'),
            split_config=split_config,
            used_parameters={'model_parameters': model_params, **split_config},
            metrics=metrics,
            plots_paths={},
            model_binary_path=result.get('model_binary_path'),
            execution_time_ms=result.get('execution_time_ms'),
        )

        messages.success(request, f"Model uruchomiony. ID: {ml_run.run_id}")
        return redirect("ml:run_model")

    return render(request, "run.html", {
        "dataset": dataset,
        "runs": runs,
        "selected_run": selected_run,
        "model_to_display": model_to_display,
        "displayed_common_params": displayed_common_params,
        "displayed_model_params": displayed_model_params,
    })


@login_required
@require_POST
def delete_run(request):
    """Delete ML run."""
    run_id = request.POST.get('run_to_delete')
    if not run_id:
        messages.error(request, "Nie podano ID przebiegu.")
        return redirect('ml:run_model')
    run = MLRun.objects.filter(run_id=run_id, user=request.user).first()
    if run:
        run.delete()
        messages.success(request, "Przebieg usunięty.")
    return redirect('ml:run_model')
