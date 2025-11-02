from .base_ml_model import BaseClassificationModel, BaseRegressionModel, BaseUnsupervisedModel,BaseDimensionalityReduction

# models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA


# ===============================================
# Classification Models (BaseClassificationModel)
# ===============================================

class LogisticRegressionModel(BaseClassificationModel):
    def create_model(self):
        # Setting max_iter high to ensure convergence, as it might not be in the simplified UI parameters.
        self.model_parameters.setdefault('max_iter', 1000)
        if 'class_weight' in self.model_parameters:
            value = self.model_parameters['class_weight']
            if value == 'None' or value == '':
                self.model_parameters['class_weight'] = None
        self.model = LogisticRegression(**self.model_parameters)


class DecisionTreeClassificationModel(BaseClassificationModel):
    def create_model(self):
        # Convert max_depth=0 from UI (intended as unlimited) to None for scikit-learn
        if self.model_parameters.get('max_depth') == 0:
            self.model_parameters['max_depth'] = None
        # Convert "None" to None for class_weight
        if 'class_weight' in self.model_parameters:
            value = self.model_parameters['class_weight']
            if value == 'None' or value == '':
                self.model_parameters['class_weight'] = None
        self.model = DecisionTreeClassifier(**self.model_parameters)


class KNNClassifierModel(BaseClassificationModel):
    def create_model(self):
        self.model = KNeighborsClassifier(**self.model_parameters)


class SVCModel(BaseClassificationModel):
    def create_model(self):
        if 'class_weight' in self.model_parameters:
            value = self.model_parameters['class_weight']
            if value == 'None' or value == '':
                self.model_parameters['class_weight'] = None
        self.model = SVC(**self.model_parameters)


class NaiveBayesModel(BaseClassificationModel):
    def create_model(self):
        self.model = GaussianNB(**self.model_parameters)


class RandomForestClassifierModel(BaseClassificationModel):
    def create_model(self):
        # Convert max_depth=0 from UI (intended as unlimited) to None for scikit-learn
        if self.model_parameters.get('max_depth') == 0:
            self.model_parameters['max_depth'] = None
        # Convert "None" to None for class_weight
        if 'class_weight' in self.model_parameters:
            value = self.model_parameters['class_weight']
            if value == 'None' or value == '':
                self.model_parameters['class_weight'] = None
        self.model = RandomForestClassifier(**self.model_parameters)


# ===============================================
# Regression Models (BaseRegressionModel)
# ===============================================

class LinearRegressionModel(BaseRegressionModel):
    def create_model(self):
        self.model = LinearRegression(**self.model_parameters)


class DecisionTreeRegressorModel(BaseRegressionModel):
    def create_model(self):
        # Convert max_depth=0 from UI (intended as unlimited) to None for scikit-learn
        if self.model_parameters.get('max_depth') == 0:
            self.model_parameters['max_depth'] = None
        self.model = DecisionTreeRegressor(**self.model_parameters)


class RandomForestRegressorModel(BaseRegressionModel):
    def create_model(self):
        # Convert max_depth=0 from UI (intended as unlimited) to None for scikit-learn
        if self.model_parameters.get('max_depth') == 0:
            self.model_parameters['max_depth'] = None
        self.model = RandomForestRegressor(**self.model_parameters)


class SVRModel(BaseRegressionModel):
    def create_model(self):
        self.model = SVR(**self.model_parameters)


# ===============================================
# Clustering Models (BaseUnsupervisedModel)
# ===============================================

class KMeansModel(BaseUnsupervisedModel):
    def create_model(self):
        self.model = KMeans(**self.model_parameters)


class DBSCANModel(BaseUnsupervisedModel):
    def create_model(self):
        self.model = DBSCAN(**self.model_parameters)


# ===============================================
# Dimensions Reduction (BaseDimensionalityReduction)
# ===============================================

class PCAModel(BaseDimensionalityReduction):
    def create_model(self):
        n_comp_value = self.model_parameters.get('n_components')

        # Handle the logic for n_components based on the UI's design choices.
        # in case 0 or None are passed to mean 'no reduction'.
        if  n_comp_value <= 0:
            self.model_parameters['n_components'] = None

        self.model = PCA(**self.model_parameters)