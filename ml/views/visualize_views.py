"""Visualization views."""

from django.contrib.auth.decorators import login_required
from django.shortcuts import render

from .utils import load_dataset_and_pipeline_from_session
from ml.services.visualize_service import get_latest_run_visualization_data


@login_required
def visualize(request):
    """Visualize latest ML run results."""
    # 1. Session validation
    result = load_dataset_and_pipeline_from_session(request)
    if not isinstance(result, tuple):
        return result
    dataset, pipeline = result

    # Service returns ready objects or error to display
    latest_run, plots, error_msg = get_latest_run_visualization_data(
        user=request.user, dataset=dataset, pipeline=pipeline
    )

    return render(
        request,
        "visualize.html",
        {
            "dataset": dataset,
            "latest_run": latest_run,
            "plots": plots,
            "error": error_msg,
        },
    )
