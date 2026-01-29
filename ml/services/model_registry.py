"""Model name to implementation class mapping."""
from ml.services.ml_models.implementations import (
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
    "Regresja Liniowa": LinearRegressionModel,
    "Drzewo decyzyjne (Regresja)": DecisionTreeRegressorModel,
    "Lasy Losowe (Regresja)": RandomForestRegressorModel,
    "Maszyna wektorów nosnych (SVM) (Regresja)": SVRModel,
    "Regresja Logistyczna": LogisticRegressionModel,
    "Drzewo decyzyjne (Klasyfikacja)": DecisionTreeClassificationModel,
    "K-najblizszych sasiadów (KNN)": KNNClassifierModel,
    "Naiwny Klasyfikator Bayesa": NaiveBayesModel,
    "Lasy Losowe (Klasyfikacja)": RandomForestClassifierModel,
    "Maszyna wektorów nosnych (SVM) (Klasyfikacja)": SVCModel,
    "K-Średnich (K-Means)": KMeansModel,
    "DBSCAN": DBSCANModel,
    "Redukcja Wymiarowości (PCA)": PCAModel,
}
