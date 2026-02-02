"""Dataset selection and studio views."""
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, get_object_or_404, redirect
from django.views.decorators.http import require_POST

from data.models import Dataset
from data.services import load_dataset_dataframe
from ml.services.plot_service import generate_exploration_histogram
from preprocessing.services import get_or_create_active_pipeline
from .utils import load_dataset_and_pipeline_from_session
from preprocessing.services import get_or_create_active_pipeline, get_train_test_dataframes
from .utils import load_dataset_and_pipeline_from_session


@login_required
@require_POST
def select_dataset(request, dataset_id):
    """Store selected dataset and pipeline in session."""
    dataset = get_object_or_404(Dataset, dataset_id=dataset_id, user=request.user)
    
    pipeline = get_or_create_active_pipeline(dataset, {'test_size': 0.2, 'random_state': 42})
    
    # Setting session
    request.session['dataset_id'] = str(dataset.dataset_id)
    request.session['pipeline_id'] = pipeline.id
    request.session['split_config'] = pipeline.split_config or {}
    
    # Clearing selected column when changing dataset
    if 'selected_column' in request.session:
        del request.session['selected_column']
        
    return redirect("ml:studio")


@login_required
def studio(request):
    """Explore dataset view."""
    # Validation and getting objects from session (helper from utils)
    result = load_dataset_and_pipeline_from_session(request)
    if not isinstance(result, tuple):
        return result
    dataset, pipeline = result

    # Loading data
    df = None

    if pipeline:
        try:
            train_test = get_train_test_dataframes(pipeline)
            if train_test:
                # Displaying training set, because it contains changes (e.g. dropped columns)
                df = train_test[0]
        except Exception as e:
            # In case of error reading pipeline, we log (optionally) and go to fallback
            print(f"Warning: Could not load pipeline data: {e}")

    # If there is no pipeline or steps, we load raw file
    if df is None:
        df = load_dataset_dataframe(dataset)

    # Getting columns
    columns = df.columns.tolist()

    # Handling column selection (POST or Session)
    if request.method == "POST":
        selected_column = request.POST.get("selected_column")
        request.session["selected_column"] = selected_column
        return redirect("ml:studio")

    selected_column = request.session.get("selected_column")

    # Intelligent default setting of column
    if selected_column not in columns:
        if dataset.target_column and dataset.target_column in columns:
            selected_column = dataset.target_column
        else:
            selected_column = columns[0] if columns else None
        request.session["selected_column"] = selected_column

    # Generating histogram of selected column
    graph = generate_exploration_histogram(df, selected_column)

    # Generating statistics of selected column
    stats_df = df.describe()
    statistics = stats_df.reset_index().values.tolist()
    stat_columns = ["Statystyka"] + stats_df.columns.tolist()

    # Generating data preview
    data_preview = df.head().values.tolist()

    return render(request, "explore.html", {
        "dataset": dataset,
        "data": data_preview,
        "columns": columns,
        "statistics": statistics,
        "stat_columns": stat_columns,
        "graph": graph,
        "selected_column": selected_column,
    })
