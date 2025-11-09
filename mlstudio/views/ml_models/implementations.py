from .base_ml_model import BaseClassificationModel, BaseRegressionModel, BaseUnsupervisedModel,BaseDimensionalityReduction

# draw plots
from ..utils import plot_to_base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.base import clone
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder
import pandas as pd

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

    def evaluate_model(self, y_pred):
        evaluation = super().evaluate_model(y_pred)

        try:
            fig, ax = plt.subplots(figsize=(25, 15))

            feature_names = self.X_train.columns.tolist()
            class_names = [str(c) for c in self.model.classes_]

            tree.plot_tree(
                self.model,
                feature_names=feature_names,
                class_names=class_names,
                filled=True,
                rounded=True,
                ax=ax,
                fontsize=10
            )

            evaluation['plot_tree_base64'] = plot_to_base64(fig)

        except Exception as e:
            print(f"Błąd podczas rysowania drzewa decyzyjnego: {e}")
            evaluation['plot_tree_base64'] = None

        return evaluation


class KNNClassifierModel(BaseClassificationModel):
    def create_model(self):
        self.model = KNeighborsClassifier(**self.model_parameters)

    def evaluate_model(self, y_pred):
        """
        It evaluates the model and generates a decision boundary graph.
        - If the data has 2 features, it draws the graph directly.
        - If the data has > 2 features, it uses PCA to reduce it to 2D and then draws it.
        """
        evaluation = super().evaluate_model(y_pred)

        try:
            n_features = self.X_train.shape[1]
            plot_title = f"Granice Decyzyjne KNN (k = {self.model_parameters.get('n_neighbors', 'auto')})"

            if n_features >= 2:

                X_plot = self.X_train.values
                y_plot = self.y_train
                x_label = self.X_train.columns[0]
                y_label = self.X_train.columns[1]

                if n_features > 2:
                    plot_title += " (Wizualizacja PCA)"
                    x_label = "Główna Składowa 1 (PC1)"
                    y_label = "Główna Składowa 2 (PC2)"


                    pca = PCA(n_components=2)
                    X_plot = pca.fit_transform(self.X_train.values)

                le = LabelEncoder()
                y_numeric = le.fit_transform(y_plot)

                plot_model = clone(self.model)
                plot_model.fit(X_plot, y_numeric)

                h = .02
                x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
                y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                     np.arange(y_min, y_max, h))

                Z = plot_model.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)

                n_classes = len(np.unique(y_numeric))
                palette_light = sns.color_palette("pastel", n_classes)
                palette_bold = sns.color_palette("bright", n_classes)
                cmap_light = ListedColormap(palette_light)

                fig, ax = plt.subplots(figsize=(10, 8))
                ax.contourf(xx, yy, Z, cmap=cmap_light)
                sns.scatterplot(x=X_plot[:, 0], y=X_plot[:, 1], hue=y_plot,
                                palette=palette_bold, ax=ax, edgecolor="k")

                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                ax.set_title(plot_title)

                evaluation['plot_knn_decision_boundary'] = plot_to_base64(fig)

            else:
                print(f"Wizualizacja granic KNN pominięta: wymaga co najmniej 2 cech, znaleziono {n_features}")
                evaluation['plot_knn_decision_boundary'] = None

        except Exception as e:
            print(f"Błąd podczas rysowania granic decyzyjnych KNN: {e}")
            evaluation['plot_knn_decision_boundary'] = None

        return evaluation


class SVCModel(BaseClassificationModel):
    def create_model(self):
        if 'class_weight' in self.model_parameters:
            value = self.model_parameters['class_weight']
            if value == 'None' or value == '':
                self.model_parameters['class_weight'] = None
        self.model = SVC(**self.model_parameters)

    def evaluate_model(self, y_pred):
        """
        Evaluates the model and generates a decision boundary plot for SVC.
        - Uses PCA for data > 2D.
        - Always scales the data (critical for SVC).
        """
        evaluation = super().evaluate_model(y_pred)

        try:
            n_features = self.X_train.shape[1]

            kernel = self.model_parameters.get('kernel', 'rbf')
            C = self.model_parameters.get('C', 1.0)
            plot_title = f"Granice Decyzyjne SVC (kernel={kernel}, C={C})"

            if n_features >= 2:

                y_plot = self.y_train

                if n_features > 2:
                    plot_title += " (Wizualizacja PCA)"
                    x_label = "Główna Składowa 1 (PC1)"
                    y_label = "Główna Składowa 2 (PC2)"

                    pca = PCA(n_components=2)
                    X_plot = pca.fit_transform(self.X_train.values)
                else:
                    X_plot = self.X_train.values
                    x_label = self.X_train.columns[0]
                    y_label = self.X_train.columns[1]

                le = LabelEncoder()
                y_numeric = le.fit_transform(y_plot)

                plot_model = clone(self.model)
                plot_model.fit(X_plot, y_numeric)

                h = .02
                x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
                y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                     np.arange(y_min, y_max, h))

                Z = plot_model.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)

                n_classes = len(np.unique(y_numeric))
                palette_light = sns.color_palette("pastel", n_classes)
                palette_bold = sns.color_palette("bright", n_classes)
                cmap_light = ListedColormap(palette_light)

                fig, ax = plt.subplots(figsize=(10, 8))
                ax.contourf(xx, yy, Z, cmap=cmap_light)
                sns.scatterplot(x=X_plot[:, 0], y=X_plot[:, 1], hue=y_plot,
                                palette=palette_bold, ax=ax, edgecolor="k")

                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                ax.set_title(plot_title)

                evaluation['plot_svc_decision_boundary'] = plot_to_base64(fig)

            else:
                print(f"Wizualizacja granic SVC pominięta: wymaga co najmniej 2 cech, znaleziono {n_features}")
                evaluation['plot_svc_decision_boundary'] = None

        except Exception as e:
            print(f"Błąd podczas rysowania granic decyzyjnych SVC: {e}")
            evaluation['plot_svc_decision_boundary'] = None

        return evaluation


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

    def evaluate_model(self, y_pred):
        evaluation = super().evaluate_model(y_pred)

        try:
            fig, ax = plt.subplots(figsize=(25, 15))
            feature_names = self.X_train.columns.tolist()

            tree.plot_tree(
                self.model,
                feature_names=feature_names,
                filled=True,
                rounded=True,
                ax=ax,
                fontsize=10
            )

            evaluation['plot_tree_base64'] = plot_to_base64(fig)

        except Exception as e:
            print(f"Błąd podczas rysowania drzewa regresji: {e}")
            evaluation['plot_tree_base64'] = None

        return evaluation


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

    def evaluate_model(self, labels):
        """
        Evaluates the model and generates a visualization of 2D clusters.
        - If the data has 2 features, it draws them directly.
        - If the data has > 2 features, it uses PCA to reduce it to 2D.
        """
        evaluation = super().evaluate_model(labels)

        try:
            n_features = self.X.shape[1]
            X_plot = None
            pca = None

            if n_features == 2:
                X_plot = self.X.values
                x_label = self.X.columns[0]
                y_label = self.X.columns[1]
                plot_title = "Wizualizacja Klastrów K-Means"

            elif n_features > 2:
                pca = PCA(n_components=2)
                X_plot = pca.fit_transform(self.X.values)
                x_label = "Główna Składowa 1 (PC1)"
                y_label = "Główna Składowa 2 (PC2)"
                plot_title = "Wizualizacja Klastrów K-Means (PCA)"

            if X_plot is not None:
                fig, ax = plt.subplots(figsize=(10, 8))

                plot_df = pd.DataFrame(X_plot, columns=[x_label, y_label])
                plot_df['Klaster'] = pd.Categorical(labels)

                sns.scatterplot(
                    data=plot_df,
                    x=x_label,
                    y=y_label,
                    hue='Klaster',
                    palette='deep',
                    ax=ax,
                    legend='full'
                )

                centroids_original_space = self.model.cluster_centers_

                if pca:
                    centroids_plot_space = pca.transform(centroids_original_space)
                else:
                    centroids_plot_space = centroids_original_space

                ax.scatter(
                    centroids_plot_space[:, 0],
                    centroids_plot_space[:, 1],
                    marker='X',
                    s=250,
                    c='red',
                    edgecolor='black',
                    label='Centroidy'
                )

                ax.set_title(plot_title)
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                ax.legend()

                evaluation['plot_kmeans_clusters'] = plot_to_base64(fig)

        except Exception as e:
            print(f"Błąd podczas rysowania wizualizacji K-Means: {e}")
            evaluation['plot_kmeans_clusters'] = None

        return evaluation


class DBSCANModel(BaseUnsupervisedModel):
    def create_model(self):
        self.model = DBSCAN(**self.model_parameters)

    def evaluate_model(self, labels):
        """
        Evaluates the model and generates a 2D cluster visualization for DBSCAN.
        - If the data has 2 features, it plots them directly.
        - If the data has > 2 features, it uses PCA to reduce to 2D.
        - Noise points (label -1) are plotted separately.
        """
        evaluation = super().evaluate_model(labels)

        try:
            n_features = self.X.shape[1]
            X_plot = None


            if n_features == 2:
                X_plot = self.X.values
                x_label = self.X.columns[0] + " (Przeskalowane)"
                y_label = self.X.columns[1] + " (Przeskalowane)"
                plot_title = "Wizualizacja Klastrów DBSCAN"

            elif n_features > 2:
                pca = PCA(n_components=2)
                X_plot = pca.fit_transform(self.X.values)
                x_label = "Główna Składowa 1 (PC1)"
                y_label = "Główna Składowa 2 (PC2)"
                plot_title = "Wizualizacja Klastrów DBSCAN (PCA)"

            if X_plot is not None:
                fig, ax = plt.subplots(figsize=(10, 8))

                plot_df = pd.DataFrame(X_plot, columns=[x_label, y_label])
                plot_df['Klaster'] = pd.Categorical(labels)


                noise_df = plot_df[plot_df['Klaster'] == -1]
                clustered_df = plot_df[plot_df['Klaster'] != -1]

                # 1. Narysuj punkty w klastrach
                if not clustered_df.empty:
                    sns.scatterplot(
                        data=clustered_df,
                        x=x_label,
                        y=y_label,
                        hue='Klaster',
                        palette='deep',
                        ax=ax,
                        legend='full'
                    )

                # 2. Narysuj punkty szumu
                if not noise_df.empty:
                    ax.scatter(
                        noise_df[x_label],
                        noise_df[y_label],
                        marker='x',
                        s=20,
                        c='black',
                        label='Szum (Klaster -1)'
                    )

                ax.set_title(plot_title)
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                ax.legend()

                evaluation['plot_dbscan_clusters'] = plot_to_base64(fig)

        except Exception as e:
            print(f"Błąd podczas rysowania wizualizacji DBSCAN: {e}")
            evaluation['plot_dbscan_clusters'] = None

        return evaluation

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

    def evaluate_model(self, X_transformed):
        """
        Generates visualizations for PCA:
        1. Scree Plot (Explained Variance by Component).
        2. Biplot (if n_components = 2).
        """
        evaluation = super().evaluate_model(X_transformed)

        try:

            full_pca = PCA(n_components=None)
            full_pca.fit(self.X_train)

            # 1. Scree Plot
            fig_scree, ax_scree = plt.subplots(figsize=(10, 6))
            num_components = len(full_pca.explained_variance_ratio_)
            components = np.arange(1, num_components + 1)

            ax_scree.plot(components, full_pca.explained_variance_ratio_, marker='o', linestyle='--')
            ax_scree.set_xlabel('Główna Składowa')
            ax_scree.set_ylabel('Wyjaśniona Wariancja')
            ax_scree.set_title('Wykres Spadków Wariancji (Scree Plot)')
            ax_scree.grid(True)
            evaluation['plot_pca_scree'] = plot_to_base64(fig_scree)

            # 2. Biplot (if n_components = 2)
            actual_n_components = self.model.n_components_ if hasattr(self.model, 'n_components_') else None

            if (actual_n_components == 2) or (self.model_parameters.get('n_components') is None and self.X_train.shape[1] == 2):

                fig_biplot, ax_biplot = plt.subplots(figsize=(12, 10))

                X_reduced_2d = self.model.fit_transform(self.X_train)

                sns.scatterplot(
                    x=X_reduced_2d[:, 0],
                    y=X_reduced_2d[:, 1],
                    ax=ax_biplot,
                    alpha=0.6,
                    label='Zredukowane Dane'
                )

                feature_vectors = self.model.components_.T * np.sqrt(self.model.explained_variance_)

                for i, feature in enumerate(self.X_train.columns):
                    ax_biplot.arrow(
                        0, 0,
                        feature_vectors[i, 0],
                        feature_vectors[i, 1],
                        head_width=0.05,
                        head_length=0.05,
                        fc='red',
                        ec='red'
                    )
                    ax_biplot.text(
                        feature_vectors[i, 0] * 1.1,
                        feature_vectors[i, 1] * 1.1,
                        feature,
                        color='green',
                        ha='center',
                        va='center'
                    )

                ax_biplot.set_xlabel("Główna Składowa 1")
                ax_biplot.set_ylabel("Główna Składowa 2")
                ax_biplot.set_title("Biplot PCA (Punkty i Wektory Cech)")
                ax_biplot.grid(True)
                ax_biplot.axhline(0, color='gray', linewidth=0.5)
                ax_biplot.axvline(0, color='gray', linewidth=0.5)
                ax_biplot.legend()

                evaluation['plot_pca_biplot'] = plot_to_base64(fig_biplot)

        except Exception as e:
            print(f"Błąd podczas rysowania wizualizacji PCA: {e}")
            evaluation['plot_pca_scree'] = None
            evaluation['plot_pca_biplot'] = None

        return evaluation