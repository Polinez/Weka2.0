from .base_ml_model import BaseClassificationModel, BaseRegressionModel

# models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB



class LinearRegressionModel(BaseRegressionModel):
    def create_model(self):
        self.model = LinearRegression(**self.model_parameters)

class DecisionTreeRegressorModel(BaseRegressionModel):
    def create_model(self):
        self.model = DecisionTreeRegressor(**self.model_parameters)

class DecisionTreeClassificationModel(BaseClassificationModel):
    def create_model(self):
        self.model = DecisionTreeClassifier(**self.model_parameters)

class KNNClassifierModel(BaseClassificationModel):
    def create_model(self):
        self.model = KNeighborsClassifier(**self.model_parameters)

class SVMModel(BaseClassificationModel):
    def create_model(self):
        self.model = SVC(**self.model_parameters)

class NaiveBayesModel(BaseClassificationModel):
    def create_model(self):
        self.model = GaussianNB(**self.model_parameters)

class RandomForestClassifierModel(BaseClassificationModel):
    def create_model(self):
        self.model = RandomForestClassifier(**self.model_parameters)

class LogisticRegressionModel(BaseClassificationModel):
    def create_model(self):
        self.model = LogisticRegression(**self.model_parameters)