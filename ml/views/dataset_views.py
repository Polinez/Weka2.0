"""Dataset selection and studio views."""
import pandas as pd
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, get_object_or_404, redirect
from django.views.decorators.http import require_POST

from data.models import Dataset
from data.services import load_dataset_dataframe
from .utils import load_dataset_and_pipeline_from_session
from ml.services.plot_service import plot_to_base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_plot(df, columns=None):
    """Generate histogram plot for selected columns."""
    buffer = __import__('io').BytesIO()
    plt.figure(figsize=(10, 6))
    if columns is None or len(columns) == 0:
        plt.text(0.5, 0.5, "Wybierz kolumny do wizualizacji", ha='center', va='center', fontsize=14)
        plt.axis('off')
    else:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            plt.text(0.5, 0.5, f"Nie znaleziono kolumn: {', '.join(missing)}", ha='center', va='center', fontsize=12)
            plt.axis('off')
        else:
            non_numeric = [c for c in columns if not pd.api.types.is_numeric_dtype(df[c])]
            if non_numeric:
                plt.text(0.5, 0.5, f"Kolumna {', '.join(non_numeric)} jest nienumeryczna", ha='center', va='center', fontsize=13)
                plt.axis('off')
            else:
                for col in columns:
                    plt.hist(df[col].dropna(), bins=30, alpha=0.6, label=col, edgecolor='black', linewidth=1)
                plt.title("Rozkład wybranych cech numerycznych")
                plt.xlabel("Wartość cechy")
                plt.ylabel("Liczba wystąpień")
                plt.legend()
                plt.grid(alpha=0.3)
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    import base64
    graph = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()
    plt.close()
    return graph


@login_required
@require_POST
def select_dataset(request, dataset_id):
    """Store selected dataset and pipeline in session."""
    from preprocessing.services import get_or_create_active_pipeline
    dataset = get_object_or_404(Dataset, dataset_id=dataset_id, user=request.user)
    pipeline = dataset.pipelines.filter(is_active=True).first()
    if not pipeline:
        pipeline = get_or_create_active_pipeline(dataset, {'test_size': 0.2, 'random_state': 42})
    request.session['dataset_id'] = str(dataset.dataset_id)
    request.session['pipeline_id'] = pipeline.id
    request.session['split_config'] = pipeline.split_config or {}
    if 'selected_column' in request.session:
        del request.session['selected_column']
    return redirect("ml:studio")


@login_required
def studio(request):
    """Explore dataset view."""
    result = load_dataset_and_pipeline_from_session(request)
    if not isinstance(result, tuple):
        return result
    dataset, pipeline = result

    df = load_dataset_dataframe(dataset)
    data = df.head().values.tolist()
    columns = df.columns.tolist()
    selected_column = None

    if request.method == "POST":
        selected_column = request.POST.get("selected_column")
        request.session["selected_column"] = selected_column
        return redirect("ml:studio")

    selected_column = request.session.get("selected_column")
    if selected_column not in columns:
        if dataset.target_column and dataset.target_column in columns:
            selected_column = dataset.target_column
        else:
            selected_column = columns[0] if columns else None
        request.session["selected_column"] = selected_column

    graph = get_plot(df, columns=[selected_column]) if selected_column else None
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
        "selected_column": selected_column,
    })
