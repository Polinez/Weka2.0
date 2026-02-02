"""ML run views."""
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.views.decorators.http import require_POST

from ml.models import MLModel, MLRun
from ml.services.run_service import run_ml_experiment
from .utils import load_dataset_and_pipeline_from_session


@login_required
def run_model(request):
    """Run ML model and display results."""
    # 1. Session validation
    result = load_dataset_and_pipeline_from_session(request)
    if not isinstance(result, tuple):
        return result
    dataset, pipeline = result

    # 2. Getting selected model
    session_model_id = request.session.get('last_model_id')
    if not session_model_id:
        messages.error(request, "Najpierw wybierz model i parametry w zakładce Modele.")
        return redirect("ml:models")

    model = get_object_or_404(MLModel, id=session_model_id)
    
    # 3. Configuring parameters
    session_params = request.session.get('last_params', {})
    common_params = session_params.get('common_parameters', {})
    model_params = session_params.get('model_parameters', {})
    
    # Determining split_config (priority: session -> pipeline -> default)
    split_config = request.session.get('split_config') or pipeline.split_config or {'test_size': 0.2, 'random_state': 42}

    # 4. Handling POST (Training execution)
    if request.method == "POST" and "run_model" in request.POST:
        used_params = {
            'model_parameters': model_params,
            **split_config
        }

        # This call does everything: loads data, trains, saves files and creates DB entry
        result = run_ml_experiment(
            user=request.user,
            dataset=dataset,
            pipeline=pipeline,
            model=model,
            split_config=split_config,
            used_parameters=used_params,
        )

        if result.get('status') == 'Failed':
            messages.error(request, f"Błąd treningu: {result.get('error')}")
        else:
            # Success - getting ID from object returned by service
            run_obj = result.get('run_obj')
            messages.success(request, f"Model uruchomiony pomyślnie. ID: {run_obj.run_id}")
            return redirect("ml:run_model")

    # 5. Preparing data to display (GET)
    runs = MLRun.objects.filter(dataset=dataset, user=request.user).order_by('-created_at')

    selected_run = None
    displayed_common = common_params
    displayed_model = model_params
    model_to_display = model

    # Handling click on history (run_id in URL)
    run_id = request.GET.get("run_id")
    if run_id:
        selected_run = runs.filter(run_id=run_id).first()
        if selected_run:
            displayed_common = selected_run.split_config or {}
            displayed_model = selected_run.used_parameters.get('model_parameters', {})
            model_to_display = selected_run.model

    return render(request, "run.html", {
        "dataset": dataset,
        "runs": runs,
        "selected_run": selected_run,
        "model_to_display": model_to_display,
        "displayed_common_params": displayed_common,
        "displayed_model_params": displayed_model,
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
