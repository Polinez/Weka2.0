from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, silhouette_score, davies_bouldin_score, f1_score, mean_absolute_error
import json

class BaseMLModel(ABC):
    def __init__(self, common_parameters: dict, model_parameters: dict, target_column: str):
        self.model = None
        self.common_parameters = common_parameters
        self.model_parameters = model_parameters
        self.target_column = target_column
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X = None # for unsupervised models

    def prepare_data(self, df):
        test_size_str = self.common_parameters.get("test_size")
        random_state_str = self.common_parameters.get("random_state")

        try:
            test_size = float(test_size_str)
            random_state = int(random_state_str)
        except (ValueError, TypeError):
            test_size = 0.2
            random_state = 42
            print("Invalid common parameters for data splitting. Using default values.")

        # split data into train and test sets based on supervised and unsupervised
        if self.target_column:
            # Classification and Regression
            X = df.drop(columns=[self.target_column])
            y = df[self.target_column]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state
            )
        else:
            # Unsupervised models (e.g., Clustering, PCA)
            self.X = df
            self.X_train = df
            self.y_train = None


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

    def run(self, df):
        self.prepare_data(df)
        self.create_model()
        self.train_model()
        result = self.process_data()
        evaluation = self.evaluate_model(result)
        return evaluation



class BaseClassificationModel(BaseMLModel):
    """Base class for supervised classification models."""

    def train_model(self):
        """Standard training: fit model to X_train and y_train."""
        self.model.fit(self.X_train, self.y_train)

    def process_data(self):
        """Standard prediction: predict labels for X_test."""
        return self.model.predict(self.X_test)

    def evaluate_model(self, y_pred):
        """Evaluation specific for classification: accuracy and report."""
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
        report = classification_report(self.y_test, y_pred, output_dict=True, zero_division=0)
        return {
            "model_type": "Classification",
            "accuracy": accuracy,
            "f1": f1,
            "classification_report": report,
            "y_test": self.y_test.tolist(),
            "y_pred": y_pred.tolist()
        }


class BaseRegressionModel(BaseMLModel):
    """Base class for supervised regression models."""

    def train_model(self):
        """Standard training: fit model to X_train and y_train."""
        self.model.fit(self.X_train, self.y_train)

    def process_data(self):
        """Standard prediction: predict continuous values for X_test."""
        return self.model.predict(self.X_test)

    def evaluate_model(self, y_pred):
        """Evaluation specific for regression: MSE and R2 score."""
        mse = mean_squared_error(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        return {
            "model_type": "Regression",
            "mean_absolute_error": mae,
            "mean_squared_error": mse,
            "r2_score": r2,
            "y_test": self.y_test.tolist(),
            "y_pred": y_pred.tolist()
        }


class BaseUnsupervisedModel(BaseMLModel):
    """Base class for clustering models (e.g., K-Means, DBSCAN)."""

    def __init__(self, common_parameters: dict, model_parameters: dict, target_column: str = None):
        super().__init__(common_parameters, model_parameters, target_column=None)

    def prepare_data(self, df):
        self.X = df
        self.X_train = df
        self.y_train = None

    def train_model(self):
        """Clustering models often perform fitting within the prediction step."""
        pass

    def process_data(self):
        """Performs clustering (fit_predict) and returns labels for all samples X."""
        return self.model.fit_predict(self.X)

    def evaluate_model(self, labels):
        """Evaluation for clustering: uses internal metrics like Silhouette and DB Index."""
        try:
            unique_labels = np.unique(labels)
            # Metrics require at least 2 unique clusters
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
            "number_of_clusters": len(unique_labels) if labels is not None else 0,
            "silhouette_score": silhouette,
            "davies_bouldin_score": db_score
        }


class BaseDimensionalityReduction(BaseMLModel):
    """Base class for dimensionality reduction (e.g., PCA)."""

    def __init__(self, common_parameters: dict, model_parameters: dict, target_column: str = None):
        super().__init__(common_parameters, model_parameters, target_column=None)

    def prepare_data(self, df):
        """Splits data X for fitting (training) and transforming (testing)."""
        test_size = float(self.common_parameters.get("test_size", 0.2))
        random_state = int(self.common_parameters.get("random_state", 42))

        # Split X into train/test, as is standard practice before transformation
        self.X_train, self.X_test = train_test_split(df, test_size=test_size, random_state=random_state)
        self.X = df

    def train_model(self):
        """Trains the model by fitting it to the training data (X_train)."""
        self.model.fit(self.X_train)

    def process_data(self):
        """Returns the transformed (dimensionally reduced) test data."""
        return self.model.transform(self.X_test)

    def evaluate_model(self, X_transformed):
        """Evaluation for dimensionality reduction: variance explained and size comparison."""


        return {
            "model_type": "Dimensionality Reduction",
            "original_features": self.X_train.shape[1] if self.X_train is not None else 0,
            "reduced_features": X_transformed.shape[1] if X_transformed is not None else 0,
            # Return a small sample of the transformed data
            "transformed_data_sample": X_transformed[:5].tolist() if X_transformed is not None else []
        }