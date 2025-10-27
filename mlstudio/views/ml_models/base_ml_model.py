from abc import ABC, abstractmethod
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
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

    def prepare_data(self, df):
        target_column = self.target_column

        test_size_str = self.common_parameters.get("train_set")
        random_state_str = self.common_parameters.get("random_state")

        try:
            test_size = float(test_size_str)
            random_state = int(random_state_str)
        except (ValueError, TypeError):
            test_size = 0.2
            random_state = 42
            print("Invalid common parameters for data splitting. Using default values.")


        X = df.drop(columns=[target_column])
        y = df[target_column]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def predict_model(self):
        pass

    @abstractmethod
    def evaluate_model(self, y_pred):
        pass

    def run(self, df):
        self.prepare_data(df)
        self.create_model()
        self.train_model()
        y_pred = self.predict_model()
        evaluation = self.evaluate_model(y_pred)
        return evaluation



class BaseClassificationModel(BaseMLModel):
    """
    Base class for classification.
    Implements train predict evaluate for classification.
    """
    def train_model(self):
        """Standard trening for most models."""
        self.model.fit(self.X_train, self.y_train)

    def predict_model(self):
        """Standard predykcja for most models."""
        return self.model.predict(self.X_test)

    def evaluate_model(self, y_pred):
        """Evaluation specific for classification."""
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred, output_dict=True, zero_division=0)
        return {
            "model_type": "Classification",
            "accuracy": accuracy,
            "classification_report": report
        }


class BaseRegressionModel(BaseMLModel):
    """
    Base class for regression.
    Implements train predict evaluate for regression.
    """
    def train_model(self):
        """Standard trening for most models."""
        self.model.fit(self.X_train, self.y_train)

    def predict_model(self):
        """Standard predict to most models."""
        return self.model.predict(self.X_test)

    def evaluate_model(self, y_pred):
        """Evaluate for regression."""
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        return {
            "model_type": "Regression",
            "mean_squared_error": mse,
            "r2_score": r2
        }