"""Service for handling ML Models logic."""
from ml.models import MLModel


def convert_param_value(value_str, data_type):
    """
    Safely converts string values from HTML form to Python types.
    Handles 'int', 'float', 'bool'.
    """
    if value_str is None or str(value_str).strip() == "":
        if data_type == 'bool':
            return False
        # For numbers empty string is None
        return None

    try:
        if data_type == 'int':
            return int(value_str)
        if data_type == 'float':
            # Handling Polish comma
            return float(str(value_str).replace(',', '.'))
        if data_type == 'bool':
            val_lower = str(value_str).lower()
            return val_lower in ['true', '1', 'on', 'yes']
        
        # By default we return as text
        return str(value_str)
    except (ValueError, TypeError):
        # We raise a clear error that can be displayed to the user
        raise ValueError(f"Wartość '{value_str}' nie jest poprawnym typem {data_type}")

def get_models_for_dataset(dataset):
    """Returns available models matching the dataset problem type."""
    if not dataset.problem_type:
        return MLModel.objects.none()
    return MLModel.objects.filter(type=dataset.problem_type)


def extract_parameters_from_post(post_data: dict, model: MLModel | None) -> dict:
    """
    Extracts and validates parameters from POST request.
    Returns nested dict: {'common': {...}, 'model': {...}}
    """
    # 1. Common parameters (Split configuration)
    common = {}
    try:
        common['test_size'] = convert_param_value(post_data.get("common_test_size"), 'float') or 0.2
        common['random_state'] = convert_param_value(post_data.get("common_random_state"), 'int') or 42
        
        # Business validation
        if not (0.01 < common['test_size'] < 1.0):
            raise ValueError("Test size musi być z zakresu (0, 1).")
    except ValueError as e:
        raise ValueError(f"Błąd parametrów ogólnych: {e}")

    # 2. Model specific parameters
    model_params = {}
    if model:
        for pdef in model.parameter_defs.all():
            # In HTML form names are like "param_max_depth"
            raw_val = post_data.get(f"param_{pdef.name}")
            try:
                val = convert_param_value(raw_val, pdef.data_type)
                # If empty, try to use default value from definition
                if val is None and pdef.default_value is not None:
                     val = convert_param_value(pdef.default_value, pdef.data_type)
                
                model_params[pdef.name] = val
            except ValueError as e:
                raise ValueError(f"Błąd w parametrze '{pdef.name}': {e}")
            
    return {'common': common, 'model': model_params}


def prepare_model_params_for_display(session_params: dict, common_defaults: dict, selected_model: MLModel | None) -> tuple[list, list]:
    """
    Prepares lists of dictionaries for rendering in Django Template.
    Merges defaults with session values.
    """
    # 1. Common Params Setup
    d_common = session_params.get('common_parameters', common_defaults)
    
    # Creating list of dictionaries ready for loop in template
    common_list = [
        {
            'name': 'test_size', 
            'value': d_common.get('test_size', 0.2), 
            'label': 'Rozmiar zbioru testowego (0.1 - 0.9)', 
            'data_type': 'float',
            'step': '0.05'
        },
        {
            'name': 'random_state', 
            'value': d_common.get('random_state', 42), 
            'label': 'Ziarno losowości (Seed)', 
            'data_type': 'int',
            'step': '1'
        },
    ]

    # 2. Model Params Setup
    d_model = session_params.get('model_parameters', {})
    model_list = []
    
    if selected_model:
        for pdef in selected_model.parameter_defs.all():
            # Session value has priority, then default from database
            val = d_model.get(pdef.name)
            if val is None:
                val = convert_param_value(pdef.default_value, pdef.data_type)
                
            model_list.append({
                'name': pdef.name,
                'value': val,
                'label': pdef.name,
                'data_type': pdef.data_type,
                'description': pdef.description,
            })
            
    return common_list, model_list