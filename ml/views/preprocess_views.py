"""Preprocessing views."""

from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_POST

from data.services import load_dataset_dataframe
from preprocessing.services import (
    get_train_test_dataframes,
    reset_pipeline,
    get_feature_metadata,
    handle_preprocessing_request,
)
from .utils import load_dataset_and_pipeline_from_session

POL_NAMES = {
    "mean": "Średnia",
    "median": "Mediana",
    "mode": "Najczęstsza wartość",
    "label_encoder": "Kodowanie etykiet",
    "one_hot_encoder": "Kodowanie binarne (One-Hot)",
    "standardization": "Standaryzacja (Z-score)",
    "normalization": "Normalizacja (Min-Max)",
}


@login_required
def preprocess(request):
    """GET: Show preprocessing page. POST: Select feature."""
    result = load_dataset_and_pipeline_from_session(request)
    if not isinstance(result, tuple):
        return result
    dataset, pipeline = result

    if not pipeline:
        messages.error(
            request, "Brak aktywnego pipeline. Skonfiguruj zadanie w Konfiguracja."
        )
        return redirect("data:set_target", dataset_id=dataset.dataset_id)

    # 1. Loading data
    try:
        df_train, _ = get_train_test_dataframes(pipeline) or (None, None)
        full_df = df_train if df_train is not None else load_dataset_dataframe(dataset)
    except Exception as e:
        messages.error(request, f"Błąd spójności danych: {e}")
        return redirect("data:load_data")

    # 2. Handling Feature Selection (POST)
    if request.method == "POST":
        selected_feature = request.POST.get("feature")
        request.session["selected_feature"] = selected_feature or None
        return redirect("ml:preprocess")

    # 3. Getting current selection from Session
    selected_feature_name = request.session.get("selected_feature")

    # 4. UI validation: If column in session doesn't exist in data (because it was deleted), clear it.
    if selected_feature_name and selected_feature_name not in full_df.columns:
        selected_feature_name = None
        request.session["selected_feature"] = None

    # 5. Getting metadata (Stateless - calculated on the fly)
    feature_metadata = get_feature_metadata(full_df, selected_feature_name)

    # History: Always fetched from DATABASE
    history = [
        f"{s.type.name}: {s.parameters}" for s in pipeline.steps.all().order_by("order")
    ]

    return render(
        request,
        "preprocess.html",
        {
            "dataset": dataset,
            "data": full_df.head().values.tolist(),
            "columns": full_df.columns.tolist(),
            "preprocessing_history": history,
            "selected_feature_name": selected_feature_name,
            "selected_feature_metadata": feature_metadata,
        },
    )


@login_required
@require_POST
def preprocess_apply(request):
    """Apply preprocessing step."""
    # 1. Context consistency
    result = load_dataset_and_pipeline_from_session(request)
    if not isinstance(result, tuple):
        return result
    dataset, pipeline = result

    # 2. Getting parameters from UI/Session
    selected_feature = request.session.get("selected_feature")
    operation = request.POST.get("apply_operation")

    if not selected_feature or not operation:
        messages.warning(request, "Nie wybrano cechy lub operacji.")
        return redirect("ml:preprocess")

    # 3. Delegation to Service (Transactionality)
    # Service will handle:
    # a) Checking if operation is legal
    # b) Updating DATABASE (adding step)
    # c) Updating FILES on disk
    # d) Updating DATABASE (new file paths)
    error_msg, success_msg = handle_preprocessing_request(
        pipeline=pipeline,
        feature=selected_feature,
        operation=operation,
        post_data=request.POST,
    )

    if error_msg:
        messages.error(request, error_msg)
    else:
        messages.success(request, success_msg)
        # If operation changed structure (deleting), update UI in session
        if operation == "delete":
            request.session["selected_feature"] = None

    return redirect("ml:preprocess")


@login_required
@require_POST
def preprocess_reset(request):
    """Reset pipeline to initial state."""
    result = load_dataset_and_pipeline_from_session(request)
    if not isinstance(result, tuple):
        return result
    _, pipeline = result  # Dataset is not needed here

    if pipeline:
        # Service clears files and records in DATABASE
        reset_pipeline(pipeline)
        # I only clear the UI state
        request.session["selected_feature"] = None
        messages.info(request, "Przywrócono stan danych sprzed preprocessingu.")

    return redirect("ml:preprocess")
