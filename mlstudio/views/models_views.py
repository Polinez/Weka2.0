from .utils import load_data_from_session
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.contrib import messages
from ..models import MLModel, CommonParameter, DatasetModelState
from loadData.models import Dataset


# --- Helper function for type conversion ---

def convert_value(value_str, data_type):
    """
    Converts a string value (from an HTML form) to its specified Python data type.

    This handles common form submission issues, like empty strings for numerical
    types (converting them to None) and decimal separators (dots/commas).
    """
    # 1. Handle empty/None values first
    # Use str(value_str).strip() to handle potential whitespace
    if value_str is None or str(value_str).strip() == "":
        if data_type == 'int':
            return None  # Allow empty (optional) integer fields
        elif data_type == 'float':
            return None  # Allow empty (optional) float fields
        elif data_type == 'bool':
            return False  # Default empty boolean to False
        else:  # 'str'
            return ""  # Default empty string to ""

    try:
        # 2. Perform type conversion
        if data_type == 'int':
            return int(value_str)
        elif data_type == 'float':
            # Standardize decimal separator to '.' before conversion
            value_str = str(value_str).replace(',', '.')
            return float(value_str)
        elif data_type == 'bool':
            # HTML <select> sends 'True' or 'False' as strings
            return str(value_str).lower() == 'true'
        elif data_type == 'str':
            return str(value_str)
    except (ValueError, TypeError, AttributeError):
        # 3. Handle failed conversion
        raise ValueError(f"Invalid value '{value_str}' for type {data_type}")


# --- Helper functions for the 'models' view ---

def _get_current_state(dataset, user):
    """
    Retrieves the last *saved* configuration (model and parameters)
    for a specific user and dataset from the database.

    Returns (None, None, {}, {}) if no state is found.
    """
    try:
        state = DatasetModelState.objects.get(dataset=dataset, user=user)
        selected_model = state.model
        # Ensure parameters are dicts, even if null/invalid in DB
        saved_common_params = state.default_parameters if isinstance(state.default_parameters, dict) else {}
        saved_model_params = state.parameters if isinstance(state.parameters, dict) else {}
        return state, selected_model, saved_common_params, saved_model_params
    except DatasetModelState.DoesNotExist:
        # No prior state saved for this user/dataset combo.
        return None, None, {}, {}


def _handle_save_parameters(request, dataset, user, common_param_definitions, selected_model):
    """
    Validates and saves all parameters from the POST request to the DatasetModelState in the database.

    This function iterates through POST data, identifies known parameters, converts them using `convert_value`, and saves them.

    If any conversion fails, it raises a `ValueError` which is expected to be caught by the calling view.
    """
    if not selected_model:
        # This error is caught by the main 'models' view.
        raise ValueError("Please select a model before saving settings.")

    # --- 1. Get type definitions for validation ---
    model_param_types = {p.name: p.data_type for p in selected_model.parameters.all()}
    common_param_types = {p.name: p.data_type for p in common_param_definitions.values()}

    # --- 2. Process and convert COMMON parameters from POST data ---
    posted_common_params = {}
    for k, v_str in request.POST.items():
        if k.startswith("common_"):
            name = k.replace("common_", "")
            if name in common_param_types:
                data_type = common_param_types.get(name)
                # This line will raise ValueError if v_str is invalid
                posted_common_params[name] = convert_value(v_str, data_type)

    # --- 3. Process and convert MODEL-SPECIFIC parameters from POST data ---
    posted_model_params = {}
    for k, v_str in request.POST.items():
        if k.startswith("param_"):
            name = k.replace("param_", "")
            if name in model_param_types:
                data_type = model_param_types.get(name)
                # This line will also raise ValueError
                posted_model_params[name] = convert_value(v_str, data_type)

    # --- 4. Save the validated parameters to the database ---
    DatasetModelState.objects.update_or_create(
        dataset=dataset,
        user=user,
        defaults={
            "model": selected_model,
            "default_parameters": posted_common_params,  # Saved common params
            "parameters": posted_model_params  # Saved model params
        }
    )
    # Success is indicated by the absence of an error.


def _get_parameters_for_template(definitions, values_dict):
    """
    Builds a list of parameter dictionaries formatted for the template.

    This function is used when we already have Python-native values (e.g.,
    loaded from the DB or from model defaults) and need to ensure
    they are correctly formatted for rendering.

    It merges parameter definitions (for name, type, and *default* value)
    with a dictionary of *actual* values (e.g., from the database).
    """
    params_for_template = []
    # Handle both dicts (for common params) and QuerySets (for model params)
    definitions_list = definitions.values() if isinstance(definitions, dict) else definitions

    for definition in definitions_list:
        # Get the saved value (e.g., from DB)
        value = values_dict.get(definition.name)

        # If the saved value is None (e.g., field was empty when saved),
        # fall back to the parameter's *defined* default value.
        if value is None:
            value = convert_value(definition.value, definition.data_type)

        params_for_template.append({
            'name': definition.name,
            'value': value,
            'data_type': definition.data_type
        })
    return params_for_template


def _build_params_from_post(definitions, post_data, prefix):
    """
    Prepares a list of parameters for the template *directly from POST data*.

    This is used when a POST request fails validation or when the user
    changes the model. It ensures that the exact (and potentially invalid)
    values the user submitted are "sticky" and shown back to them in the form.
    """
    params_for_template = []
    # Handle both dicts and QuerySets
    definitions_list = definitions.values() if isinstance(definitions, dict) else definitions

    for definition in definitions_list:
        param_name_in_post = f"{prefix}{definition.name}"
        # Gets the raw string value from request.POST (e.g., "abc" or "0.2" or "True")
        raw_value = post_data.get(param_name_in_post)

        params_for_template.append({
            'name': definition.name,
            'value': raw_value,  # Pass the raw string to the template
            'data_type': definition.data_type
        })
    return params_for_template


# --- Main View Function ---

@login_required()
def models(request):
    """
    Manages the ML model and parameter selection page for a dataset.

    GET:
        - Loads the user's last *saved* state (model + params) from DatasetModelState.
        - If no state exists, loads defaults (no model, default common params).
        - Merges saved values over defaults and renders the page.

    POST:
        Two possible actions, distinguished by the 'save_all' button:

        1. "Save All" (if 'save_all' in request.POST):
            - Tries to validate and save *all* form parameters using _handle_save_parameters.
            - On success: Redirects back to the (GET) view.
            - On validation error: Re-renders the form, showing the *invalid* data
              submitted by the user (using _build_params_from_post).

        2. "Change Model" (if 'save_all' *not* in request.POST):
            - This is triggered by the model dropdown's onchange event.
            - It does *not* save anything.
            - It re-renders the form, keeping the user's *unsaved* common parameters
              (using _build_params_from_post).
            - It loads the *default* parameters for the *newly selected* model
              (using _get_parameters_for_template).
    """
    # 1. Load dataset and user
    # (Corrected typo: sesion -> session)
    dataset = load_data_from_session(request)
    if not isinstance(dataset, Dataset):
        # load_data_from_session returns a redirect if dataset not found
        return dataset

    user = request.user

    # 2. Get definitions (these are constant for the page)
    models_list = MLModel.objects.all()
    common_param_definitions = {p.name: p for p in CommonParameter.objects.all()}

    # --- 3. Handle GET request (initial page load) ---
    if request.method != "POST":
        # Load the last *saved* state from the database.
        state, selected_model, saved_common_params, saved_model_params = _get_current_state(dataset, user)

        # 3a. Prepare Common Parameters for display
        # Start with the global defaults...
        default_common_vals = {
            p.name: convert_value(p.value, p.data_type)
            for p in common_param_definitions.values()
        }
        # ...then override with user's *saved* values.
        common_values_to_display = default_common_vals.copy()
        common_values_to_display.update(saved_common_params)

        common_parameters_for_template = _get_parameters_for_template(
            common_param_definitions,
            common_values_to_display
        )

        # 3b. Prepare Model-Specific Parameters for display
        model_parameters_for_template = []
        if selected_model:
            # Start with the *selected model's* defaults...
            default_model_vals = {
                p.name: convert_value(p.value, p.data_type)
                for p in selected_model.parameters.all()
            }
            # ...then override with user's *saved* values for this model.
            model_values_to_display = default_model_vals.copy()
            model_values_to_display.update(saved_model_params)

            model_parameters_for_template = _get_parameters_for_template(
                selected_model.parameters.all(),
                model_values_to_display
            )

        context = {
            "dataset": dataset, "models_list": models_list,
            "selected_model": selected_model,
            "common_parameters": common_parameters_for_template,
            "model_parameters": model_parameters_for_template
        }
        return render(request, "models.html", context)

    # --- 4. Handle POST requests ---

    # Get the model selected in the form
    selected_model_id = request.POST.get("selected_model")
    if selected_model_id:
        selected_model = MLModel.objects.filter(id=selected_model_id).first()
    else:
        selected_model = None

    # --- 4a. POST Branch 1: User clicked "Save All" ---
    if "save_all" in request.POST:
        try:
            # Attempt to validate all form data and save to DB
            _handle_save_parameters(request, dataset, user, common_param_definitions, selected_model)

            messages.success(request, "Parameters saved successfully ✅")
            # Redirect to the GET view to show the clean, saved state
            return redirect("mlstudio:models")

        except ValueError as e:
            # This happens if convert_value() or _handle_save_parameters() raises an error
            messages.error(request, f"Save error: {e}")

            # Rebuild the form, showing the *invalid* data the user submitted.
            # Use _build_params_from_post to pass raw strings back to the template.
            common_parameters_for_template = _build_params_from_post(
                common_param_definitions,
                request.POST,
                "common_"
            )
            model_parameters_for_template = []
            if selected_model:
                model_parameters_for_template = _build_params_from_post(
                    selected_model.parameters.all(),
                    request.POST,
                    "param_"
                )

    # --- 4b. POST Branch 2: User just changed the selected model (no save) ---
    else:
        # This branch triggers when the <select> for the model is changed,
        # which re-submits the form (but without the 'save_all' button).

        # 1. Prepare Common Parameters:
        #    Show what the user *currently* has in the form (from POST data).
        common_parameters_for_template = _build_params_from_post(
            common_param_definitions,
            request.POST,
            "common_"
        )

        # 2. Prepare Model Parameters:
        #    The model has changed, so we *discard* any previous model-specific
        #    parameters and load the *defaults* for the *newly selected* model.
        model_parameters_for_template = []
        if selected_model:
            model_params_definitions = selected_model.parameters.all()
            # Get the defaults for the new model
            model_values_to_display = {
                p.name: convert_value(p.value, p.data_type)
                for p in model_params_definitions
            }
            # Use _get_parameters_for_template as we are using clean, converted values
            model_parameters_for_template = _get_parameters_for_template(
                model_params_definitions,
                model_values_to_display
            )

    # --- 5. Render template (for all POST cases: "Model Change" or "Save Failed") ---
    # This code is reached if:
    #   1. The 'save_all' failed validation.
    #   2. The user just changed the model (POST Branch 2).
    context = {
        "dataset": dataset,
        "models_list": models_list,
        "selected_model": selected_model,
        "common_parameters": common_parameters_for_template,
        "model_parameters": model_parameters_for_template
    }
    return render(request, "models.html", context)