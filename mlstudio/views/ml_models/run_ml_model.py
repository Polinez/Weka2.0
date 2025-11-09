import pandas as pd

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
    "Drzewo decyzyjne (Regresja)": DecisionTreeRegressorModel,
    "Lasy Losowe (Regresja)": RandomForestRegressorModel,
    "Maszyna wektorów nosnych (SVM) (Regresja)": SVRModel,

    # Classification Models
    "Regresja Logistyczna": LogisticRegressionModel,
    "Drzewo decyzyjne (Klasyfikacja)": DecisionTreeClassificationModel,
    "K-najblizszych sasiadów (KNN)": KNNClassifierModel,
    "Naiwny Klasyfikator Bayesa": NaiveBayesModel,
    "Lasy Losowe (Klasyfikacja)": RandomForestClassifierModel,
    "Maszyna wektorów nosnych (SVM) (Klasyfikacja)": SVCModel,

    # Cluster Models
    "K-Średnich (K-Means)": KMeansModel,
    "DBSCAN": DBSCANModel,

    # Dimensionality Reduction Models
    "Redukcja Wymiarowości (PCA)": PCAModel,
}

def run_ml_model(df:pd.DataFrame, model_name:str, target_column:str, common_parameters:dict, model_parameters:dict):
    """
        Function to run a machine learning model on a given dataset.
        finds the appropriate model class based on modelName,
    """
    try:
        ModelClass = MODEL_MAPPING.get(model_name)
        if not ModelClass:
            raise ValueError(f"Model '{model_name}' is not supported.")


        # Initialize and run the model
        ml_model_instance = ModelClass(
            common_parameters=common_parameters,
            model_parameters=model_parameters,
            target_column=target_column
        )

        # evaluate model
        evaluation_result = ml_model_instance.run(df)

        return evaluation_result

    except Exception as e:

        return {
            "error": str(e),
            "model_name": model_name,
        }