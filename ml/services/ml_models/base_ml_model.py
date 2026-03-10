from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    r2_score,
    silhouette_score,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
)


class BaseMLModel(ABC):
    def __init__(
        self, common_parameters: dict, model_parameters: dict, target_column: str
    ):
        self.model = None
        self.common_parameters = common_parameters
        self.model_parameters = model_parameters
        self.target_column = target_column
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X = None

    def prepare_data(self, df_train, df_test):
        if self.target_column:
            if (
                self.target_column not in df_train.columns
                or self.target_column not in df_test.columns
            ):
                raise ValueError(
                    f"Kolumna docelowa '{self.target_column}' nie została znaleziona w danych."
                )
            self.X_train = df_train.drop(columns=[self.target_column])
            self.y_train = df_train[self.target_column]
            self.X_test = df_test.drop(columns=[self.target_column])
            self.y_test = df_test[self.target_column]
        else:
            self.X_train = df_train
            self.X_test = df_test
            self.y_train = None
            self.y_test = None
            self.X = pd.concat([df_train, df_test], ignore_index=True)

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def process_data(self):
        pass

    @abstractmethod
    def evaluate_model(self, y_pred):
        pass

    def run(self, df_train, df_test):
        self.prepare_data(df_train, df_test)
        self.create_model()
        self.train_model()
        result = self.process_data()
        return self.evaluate_model(result)


class BaseClassificationModel(BaseMLModel):
    def train_model(self):
        self.model.fit(self.X_train, self.y_train)

    def process_data(self):
        return self.model.predict(self.X_test)

    def evaluate_model(self, y_pred):
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average="weighted", zero_division=0)
        report = classification_report(
            self.y_test, y_pred, output_dict=True, zero_division=0
        )
        y_pred_proba = None
        try:
            if hasattr(self.model, "predict_proba"):
                y_pred_proba = self.model.predict_proba(self.X_test)
        except Exception:
            pass
        return {
            "model_type": "Classification",
            "accuracy": accuracy,
            "f1": f1,
            "classification_report": report,
            "y_test": self.y_test.tolist(),
            "y_pred": y_pred.tolist(),
            "y_pred_proba": y_pred_proba.tolist() if y_pred_proba is not None else None,
        }


class BaseRegressionModel(BaseMLModel):
    def train_model(self):
        self.model.fit(self.X_train, self.y_train)

    def process_data(self):
        return self.model.predict(self.X_test)

    def evaluate_model(self, y_pred):
        mse = mean_squared_error(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        return {
            "model_type": "Regression",
            "mean_absolute_error": mae,
            "mean_squared_error": mse,
            "r2_score": r2,
            "y_test": self.y_test.tolist(),
            "y_pred": y_pred.tolist(),
        }


class BaseUnsupervisedModel(BaseMLModel):
    def __init__(
        self, common_parameters: dict, model_parameters: dict, target_column: str = None
    ):
        super().__init__(common_parameters, model_parameters, target_column=None)

    def prepare_data(self, df_train, df_test):
        self.X = pd.concat([df_train, df_test], ignore_index=True)
        self.X_train = self.X
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def train_model(self):
        pass

    def process_data(self):
        return self.model.fit_predict(self.X_train)

    def evaluate_model(self, labels):
        try:
            unique_labels = np.unique(labels)
            if len(unique_labels) > 1 and -1 not in unique_labels:
                silhouette = silhouette_score(self.X, labels)
                db_score = davies_bouldin_score(self.X, labels)
            else:
                silhouette = None
                db_score = None
        except Exception:
            silhouette = None
            db_score = None
        return {
            "model_type": "Clustering",
            "labels": labels.tolist() if labels is not None else [],
            "number_of_clusters": len(np.unique(labels)) if labels is not None else 0,
            "silhouette_score": silhouette,
            "davies_bouldin_score": db_score,
        }

    def run(self, df_train, df_test):
        self.prepare_data(df_train, df_test)
        self.create_model()
        self.train_model()
        result = self.process_data()
        return self.evaluate_model(result)


class BaseDimensionalityReduction(BaseMLModel):
    def __init__(
        self, common_parameters: dict, model_parameters: dict, target_column: str = None
    ):
        super().__init__(common_parameters, model_parameters, target_column=None)

    def prepare_data(self, df_train, df_test):
        self.X = pd.concat([df_train, df_test], ignore_index=True)
        self.X_train = self.X
        self.X_test = self.X

    def train_model(self):
        self.model.fit(self.X_train)

    def process_data(self):
        return self.model.transform(self.X_test)

    def evaluate_model(self, X_transformed):
        explained_variance_ratio_ = self.model.explained_variance_ratio_
        total_explained_variance = sum(explained_variance_ratio_)
        return {
            "model_type": "Dimensionality Reduction",
            "original_features": (
                self.X_train.shape[1] if self.X_train is not None else 0
            ),
            "reduced_features": (
                X_transformed.shape[1] if X_transformed is not None else 0
            ),
            "total_explained_variance": total_explained_variance,
            "explained_variance_per_component": explained_variance_ratio_.tolist(),
            "transformed_data_sample": (
                X_transformed[:5].tolist() if X_transformed is not None else []
            ),
        }
