import io
import pandas as pd
from loadData.models import Dataset

from mlstudio.views.ml_models.implementations import (
    LogisticRegressionModel,
    LinearRegressionModel,
    DecisionTreeRegressorModel,
    DecisionTreeClassificationModel,
    KNNClassifierModel,
    SVCModel,
    SVRModel,
    NaiveBayesModel,
    RandomForestClassifierModel,
    RandomForestRegressorModel,
    KMeansModel,
    DBSCANModel,
    PCAModel,
)

MODEL_MAPPING = {
    # "modelName": RealMOdel from implementations.py

    # Regression Models
    "Regresja Liniowa": LinearRegressionModel,
    "Drzewo decyzyjne (Regresja)": DecisionTreeRegressorModel,  # Pełna nazwa z UI
    "Lasy Losowe (Regresja)": RandomForestRegressorModel,  # Pełna nazwa z UI
    "Maszyna wektorów nosnych (SVM) (Regresja)": SVRModel,  # SVR

    # Classification Models
    "Regresja Logistyczna": LogisticRegressionModel,
    "Drzewo decyzyjne (Klasyfikacja)": DecisionTreeClassificationModel,  # Pełna nazwa z UI
    "K-najblizszych sasiadów (KNN)": KNNClassifierModel,  # Pełna nazwa z UI
    "Naiwny Klasyfikator Bayesa": NaiveBayesModel,
    "Lasy Losowe (Klasyfikacja)": RandomForestClassifierModel,  # Pełna nazwa z UI
    "Maszyna wektorów nosnych (SVM) (Klasyfikacja)": SVCModel,  # SVC

    # Cluster Models
    "K-Średnich (K-Means)": KMeansModel,
    "DBSCAN": DBSCANModel,

    # Dimensionality Reduction Models
    "Redukcja Wymiarowości (PCA)": PCAModel,
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