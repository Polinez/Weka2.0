import io
import pandas as pd
from loadData.models import Dataset

from mlstudio.views.ml_models.implementations import (
    LogisticRegressionModel,
    LinearRegressionModel,
    DecisionTreeRegressorModel,
    DecisionTreeClassificationModel,
    KNNClassifierModel,
    SVMModel,
    NaiveBayesModel,
    RandomForestClassifierModel,
)

MODEL_MAPPING = {
    # "modelName": RealMOdel from implementations.py

    # Modele Regresyjne
    "Regresja Liniowa": LinearRegressionModel,
    "Drzewo Regresyjne": DecisionTreeRegressorModel,

    # Modele Klasyfikacyjne
    "Regresja Logistyczna": LogisticRegressionModel,
    "Drzewo Decyzyjne": DecisionTreeClassificationModel,
    "k-Najbliższych Sąsiadów (KNN)": KNNClassifierModel,
    "Maszyny Wektorów Nośnych (SVM)": SVMModel,
    "Naiwny Klasyfikator Bayesa": NaiveBayesModel,
    "Las Losowy": RandomForestClassifierModel,
}

def run_ml_model(dataset:Dataset, modelName:str, common_parameters:dict, model_parameters:dict):
    """
        Function to run a machine learning model on a given dataset.
        finds the appropriate model class based on modelName,
    """
    try:
        ModelClass = MODEL_MAPPING.get(modelName)
        if not ModelClass:
            raise ValueError(f"Model '{modelName}' is not supported.")


        # Load dataset into a DataFrame
        df = pd.read_csv(io.StringIO(dataset.data))

        target_column = dataset.target_column
        if not target_column:
            raise ValueError(f"Target column '{target_column}' not found.")

        # Initialize and run the model
        ml_model_instance = ModelClass(
            common_parameters=common_parameters,
            model_parameters=model_parameters,
            target_column=target_column
        )

        # evaliate model
        evaluation_result = ml_model_instance.run(df)

        return evaluation_result

    except Exception as e:

        return {
            "error": str(e),
            "model_name": modelName,
        }