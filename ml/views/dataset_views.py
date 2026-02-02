"""Dataset selection and studio views."""
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, get_object_or_404, redirect
from django.views.decorators.http import require_POST

from data.models import Dataset
from preprocessing.services import get_or_create_active_pipeline
from .utils import load_dataset_and_pipeline_from_session
from ml.services.dataset_service import get_exploration_context


@login_required
@require_POST
def select_dataset(request, dataset_id):
    """Store selected dataset and pipeline in session."""
    dataset = get_object_or_404(Dataset, dataset_id=dataset_id, user=request.user)
    
    # Creating default pipeline when selecting dataset
    pipeline = get_or_create_active_pipeline(dataset, {'test_size': 0.2, 'random_state': 42})
    
    # Setting session 
    request.session['dataset_id'] = str(dataset.dataset_id)
    request.session['pipeline_id'] = pipeline.id
    request.session['split_config'] = pipeline.split_config or {}
    
    # Resetting UI choices dependent on data
    if 'selected_column' in request.session:
        del request.session['selected_column']
    if 'selected_feature' in request.session: # Resetting selection in Preprocessing also worth doing
        del request.session['selected_feature']
        
    return redirect("ml:studio")


@login_required
def studio(request):
    """Explore dataset view."""
    # 1. Session validation
    result = load_dataset_and_pipeline_from_session(request)
    if not isinstance(result, tuple):
        return result
    dataset, pipeline = result

    # 2. Handling column change (POST)
    if request.method == "POST":
        selected_column = request.POST.get("selected_column")
        request.session["selected_column"] = selected_column
        return redirect("ml:studio")

    # Service returns dictionary ready to put in context
    context_data = get_exploration_context(
        dataset=dataset,
        pipeline=pipeline,
        selected_column_session=request.session.get("selected_column")
    )

    # Updating session if service selected default column (auto-select)
    if context_data['selected_column'] != request.session.get("selected_column"):
        request.session["selected_column"] = context_data['selected_column']

    # 4. Rendering template
    return render(request, "explore.html", {
        "dataset": dataset,
        **context_data # Unpacking dictionary (data, statistics, graph, etc.)
    })
