from .base_ml_model import (
    BaseClassificationModel,
    BaseRegressionModel,
    BaseUnsupervisedModel,
    BaseDimensionalityReduction,
)

from ml.services.plot_service import plot_to_base64
import matplotlib
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.base import clone
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder
import pandas as pd

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

matplotlib.use("Agg")


# ===============================================
# Helper functions (DRY Helpers)
# ===============================================


def sanitize_parameters(params):
    """
    Cleans and casts input parameters from the UI to the appropriate types according to the definition.
    Uses the 'if key in cleaned' check to avoid adding arguments unsupported by a given model.
    """
    cleaned = params.copy()

    # Cast to int type
    int_params = [
        "max_iter",
        "n_neighbors",
        "p",
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "degree",
        "n_clusters",
        "min_samples",
    ]
    for key in int_params:
        if key in cleaned and cleaned[key] not in [None, "", "None", "auto"]:
            try:
                cleaned[key] = int(float(cleaned[key]))
            except ValueError:
                pass

    # Cast to float type
    float_params = ["C", "var_smoothing", "epsilon", "eps"]
    for key in float_params:
        if key in cleaned and cleaned[key] not in [None, "", "None", "auto"]:
            try:
                cleaned[key] = float(cleaned[key])
            except ValueError:
                pass

    # Cast to bool type
    bool_params = ["bootstrap", "fit_intercept", "positive", "whiten"]
    for key in bool_params:
        if key in cleaned:
            val = str(cleaned[key]).lower()
            if val in ["true", "1", "yes"]:
                cleaned[key] = True
            elif val in ["false", "0", "no"]:
                cleaned[key] = False

    # Special rules catching empty entries (changing string to None object)
    for key in ["class_weight", "max_features", "penalty"]:
        if key in cleaned and cleaned[key] in ["None", "", None]:
            cleaned[key] = None

    if "max_depth" in cleaned and cleaned["max_depth"] == 0:
        cleaned["max_depth"] = None

    if "n_components" in cleaned:
        try:
            n_comp = int(cleaned["n_components"])
            if n_comp <= 0:
                cleaned["n_components"] = None
            else:
                cleaned["n_components"] = n_comp
        except (ValueError, TypeError):
            cleaned["n_components"] = None

    if "max_features" in cleaned and str(cleaned["max_features"]) == "1.0":
        cleaned["max_features"] = 1.0

    return cleaned


def plot_feature_importance_helper(
    model, X_train, title="Ważność cech", max_features=None
):
    try:
        feature_importance = model.feature_importances_
        feature_names = X_train.columns.tolist()

        indices = np.argsort(feature_importance)[::-1]
        sorted_features = [feature_names[i] for i in indices]
        sorted_importance = feature_importance[indices]

        if max_features and len(sorted_features) > max_features:
            sorted_features = sorted_features[:max_features]
            sorted_importance = sorted_importance[:max_features]

        fig, ax = plt.subplots(figsize=(10, max(6, len(sorted_features) * 0.3)))
        sns.barplot(
            x=sorted_importance,
            y=sorted_features,
            hue=sorted_features,
            ax=ax,
            palette="viridis",
            legend=False,
        )
        ax.set_xlabel("Ważność cechy")
        ax.set_ylabel("Cecha")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)
        return plot_to_base64(fig)
    except Exception as e:
        print(f"Błąd podczas rysowania ważności cech: {e}")
        return None


def plot_decision_tree_helper(model, X_train):
    try:
        fig, ax = plt.subplots(figsize=(25, 15))
        feature_names = X_train.columns.tolist()
        class_names = (
            [str(c) for c in model.classes_] if hasattr(model, "classes_") else None
        )

        tree.plot_tree(
            model,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            ax=ax,
            fontsize=10,
        )
        return plot_to_base64(fig)
    except Exception as e:
        print(f"Błąd podczas rysowania drzewa: {e}")
        return None


def plot_decision_boundary_helper(model_instance, X_train, y_train, plot_title):
    try:
        n_features = X_train.shape[1]

        if n_features >= 2:
            X_plot = X_train.values
            y_plot = y_train  # Keep pd.Series (required for Seaborn legends and colors)
            x_label = X_train.columns[0]
            y_label = X_train.columns[1]

            if n_features > 2:
                plot_title += " (Wizualizacja PCA)"
                x_label = "Główna Składowa 1 (PC1)"
                y_label = "Główna Składowa 2 (PC2)"

                pca = PCA(n_components=2)
                X_plot = pca.fit_transform(X_train.values)

            le = LabelEncoder()
            y_numeric = le.fit_transform(y_plot)

            plot_model = clone(model_instance.model)
            plot_model.fit(X_plot, y_numeric)

            PLOT_SAMPLE_SIZE = 1000

            if len(y_numeric) > PLOT_SAMPLE_SIZE:
                try:
                    from sklearn.model_selection import train_test_split

                    X_plot, _, y_numeric, _ = train_test_split(
                        X_plot,
                        y_numeric,
                        train_size=PLOT_SAMPLE_SIZE,
                        stratify=y_numeric,
                    )
                    y_plot = y_numeric
                except ValueError:
                    indices = np.random.choice(
                        len(y_numeric), PLOT_SAMPLE_SIZE, replace=False
                    )
                    X_plot = X_plot[indices]
                    y_numeric = y_numeric[indices]
                    y_plot = y_numeric

            h = 0.15
            x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
            y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

            Z = plot_model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            n_classes = len(np.unique(y_numeric))
            palette_light = sns.color_palette("pastel", n_classes)
            palette_bold = sns.color_palette("bright", n_classes)
            cmap_light = ListedColormap(palette_light)

            fig, ax = plt.subplots(figsize=(10, 8))
            ax.contourf(xx, yy, Z, cmap=cmap_light)
            sns.scatterplot(
                x=X_plot[:, 0],
                y=X_plot[:, 1],
                hue=y_plot,
                palette=palette_bold,
                ax=ax,
                edgecolor="k",
            )

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(plot_title)

            return plot_to_base64(fig)
        else:
            print(
                f"Wizualizacja granic pominięta: wymaga co najmniej 2 cech, znaleziono {n_features}"
            )
            return None

    except Exception as e:
        print(f"Błąd podczas rysowania granic decyzyjnych: {e}")
        return None


def plot_clusters_2d_helper(
    X_df, labels, title="Wizualizacja Klastrów", centroids=None, is_dbscan=False
):
    try:
        n_features = X_df.shape[1]
        X_plot = None
        pca = None

        if n_features == 2:
            X_plot = X_df.values
            x_label = X_df.columns[0]
            y_label = X_df.columns[1]
            if is_dbscan:
                x_label += " (Przeskalowane)"
                y_label += " (Przeskalowane)"

        elif n_features > 2:
            pca = PCA(n_components=2)
            X_plot = pca.fit_transform(X_df.values)
            x_label = "Główna Składowa 1 (PC1)"
            y_label = "Główna Składowa 2 (PC2)"
            title += " (PCA)"

        if X_plot is not None:
            fig, ax = plt.subplots(figsize=(10, 8))

            plot_df = pd.DataFrame(X_plot, columns=[x_label, y_label])
            plot_df["Klaster"] = pd.Categorical(labels)

            if is_dbscan:
                noise_df = plot_df[plot_df["Klaster"] == -1]
                clustered_df = plot_df[plot_df["Klaster"] != -1]

                if not clustered_df.empty:
                    sns.scatterplot(
                        data=clustered_df,
                        x=x_label,
                        y=y_label,
                        hue="Klaster",
                        palette="deep",
                        ax=ax,
                        legend="full",
                    )
                if not noise_df.empty:
                    ax.scatter(
                        noise_df[x_label],
                        noise_df[y_label],
                        marker="x",
                        s=20,
                        c="black",
                        label="Szum (Klaster -1)",
                    )
            else:
                sns.scatterplot(
                    data=plot_df,
                    x=x_label,
                    y=y_label,
                    hue="Klaster",
                    palette="deep",
                    ax=ax,
                    legend="full",
                )

            if centroids is not None:
                if pca:
                    centroids_plot_space = pca.transform(centroids)
                else:
                    centroids_plot_space = centroids
                ax.scatter(
                    centroids_plot_space[:, 0],
                    centroids_plot_space[:, 1],
                    marker="X",
                    s=250,
                    c="red",
                    edgecolor="black",
                    label="Centroidy",
                )

            ax.set_title(title)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.legend()
            return plot_to_base64(fig)
        return None

    except Exception as e:
        print(f"Błąd podczas rysowania wizualizacji klastrów: {e}")
        return None


# ===============================================
# Classification Models (BaseClassificationModel)
# ===============================================


class LogisticRegressionModel(BaseClassificationModel):
    def create_model(self):
        self.model_parameters.setdefault("max_iter", 1000)
        self.model_parameters = sanitize_parameters(self.model_parameters)
        self.model = LogisticRegression(**self.model_parameters)


class DecisionTreeClassificationModel(BaseClassificationModel):
    def create_model(self):
        self.model_parameters = sanitize_parameters(self.model_parameters)
        self.model = DecisionTreeClassifier(**self.model_parameters)

    def evaluate_model(self, y_pred):
        evaluation = super().evaluate_model(y_pred)
        evaluation["plot_tree_base64"] = plot_decision_tree_helper(
            self.model, self.X_train
        )
        evaluation["plot_feature_importance"] = plot_feature_importance_helper(
            self.model,
            self.X_train,
            title="Ważność cech - Drzewo Decyzyjne (Klasyfikacja)",
        )
        return evaluation


class KNNClassifierModel(BaseClassificationModel):
    def create_model(self):
        self.model_parameters = sanitize_parameters(self.model_parameters)
        self.model = KNeighborsClassifier(**self.model_parameters)

    def evaluate_model(self, y_pred):
        evaluation = super().evaluate_model(y_pred)
        k_val = self.model_parameters.get("n_neighbors", "auto")
        title = f"Granice Decyzyjne KNN (k = {k_val})"
        evaluation["plot_knn_decision_boundary"] = plot_decision_boundary_helper(
            self, self.X_train, self.y_train, title
        )
        return evaluation


class SVCModel(BaseClassificationModel):
    def create_model(self):
        self.model_parameters = sanitize_parameters(self.model_parameters)
        self.model = SVC(**self.model_parameters)

    def evaluate_model(self, y_pred):
        evaluation = super().evaluate_model(y_pred)
        kernel = self.model_parameters.get("kernel", "rbf")
        c_val = self.model_parameters.get("C", 1.0)
        title = f"Granice Decyzyjne SVC (kernel={kernel}, C={c_val})"
        evaluation["plot_svc_decision_boundary"] = plot_decision_boundary_helper(
            self, self.X_train, self.y_train, title
        )
        return evaluation


class NaiveBayesModel(BaseClassificationModel):
    def create_model(self):
        self.model_parameters = sanitize_parameters(self.model_parameters)
        self.model = GaussianNB(**self.model_parameters)


class RandomForestClassifierModel(BaseClassificationModel):
    def create_model(self):
        self.model_parameters = sanitize_parameters(self.model_parameters)
        self.model = RandomForestClassifier(**self.model_parameters)

    def evaluate_model(self, y_pred):
        evaluation = super().evaluate_model(y_pred)
        title = f"Ważność cech - Lasy Losowe (Klasyfikacja, n_estimators={self.model.n_estimators})"
        evaluation["plot_feature_importance"] = plot_feature_importance_helper(
            self.model, self.X_train, title=title, max_features=20
        )
        return evaluation


# ===============================================
# Regression Models (BaseRegressionModel)
# ===============================================


class LinearRegressionModel(BaseRegressionModel):
    def create_model(self):
        self.model_parameters = sanitize_parameters(self.model_parameters)
        self.model = LinearRegression(**self.model_parameters)


class DecisionTreeRegressorModel(BaseRegressionModel):
    def create_model(self):
        self.model_parameters = sanitize_parameters(self.model_parameters)
        self.model = DecisionTreeRegressor(**self.model_parameters)

    def evaluate_model(self, y_pred):
        evaluation = super().evaluate_model(y_pred)
        evaluation["plot_tree_base64"] = plot_decision_tree_helper(
            self.model, self.X_train
        )
        evaluation["plot_feature_importance"] = plot_feature_importance_helper(
            self.model, self.X_train, title="Ważność cech - Drzewo Decyzyjne (Regresja)"
        )
        return evaluation


class RandomForestRegressorModel(BaseRegressionModel):
    def create_model(self):
        self.model_parameters = sanitize_parameters(self.model_parameters)
        self.model = RandomForestRegressor(**self.model_parameters)

    def evaluate_model(self, y_pred):
        evaluation = super().evaluate_model(y_pred)
        title = f"Ważność cech - Lasy Losowe (Regresja, n_estimators={self.model.n_estimators})"
        evaluation["plot_feature_importance"] = plot_feature_importance_helper(
            self.model, self.X_train, title=title, max_features=20
        )
        return evaluation


class SVRModel(BaseRegressionModel):
    def create_model(self):
        self.model_parameters = sanitize_parameters(self.model_parameters)
        self.model = SVR(**self.model_parameters)


# ===============================================
# Clustering Models (BaseUnsupervisedModel)
# ===============================================


class KMeansModel(BaseUnsupervisedModel):
    def create_model(self):
        self.model_parameters = sanitize_parameters(self.model_parameters)
        self.model = KMeans(**self.model_parameters)

    def evaluate_model(self, labels):
        evaluation = super().evaluate_model(labels)
        evaluation["plot_kmeans_clusters"] = plot_clusters_2d_helper(
            self.X,
            labels,
            title="Wizualizacja Klastrów K-Means",
            centroids=self.model.cluster_centers_,
        )

        try:
            max_k = min(10, len(self.X))
            k_range = range(1, max_k + 1)
            inertias = []

            for k in k_range:
                kmeans = KMeans(
                    n_clusters=k,
                    random_state=self.model_parameters.get("random_state", 42),
                    n_init=10,
                )
                kmeans.fit(self.X)
                inertias.append(kmeans.inertia_)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(
                k_range, inertias, marker="o", linestyle="--", linewidth=2, markersize=8
            )
            ax.set_xlabel("Liczba klastrów (k)")
            ax.set_ylabel("WCSS (Within-Cluster Sum of Squares)")
            ax.set_title("Elbow Method - Wybór optymalnej liczby klastrów")
            ax.grid(True, alpha=0.3)

            current_k = self.model.n_clusters
            if current_k in k_range:
                current_inertia = inertias[current_k - 1]
                ax.scatter(
                    [current_k],
                    [current_inertia],
                    color="red",
                    s=200,
                    zorder=5,
                    label=f"Obecna wartość k={current_k}",
                )
                ax.legend()

            evaluation["plot_kmeans_elbow"] = plot_to_base64(fig)
        except Exception as e:
            print(f"Błąd podczas rysowania metody łokcia: {e}")
            evaluation["plot_kmeans_elbow"] = None

        return evaluation


class DBSCANModel(BaseUnsupervisedModel):
    def create_model(self):
        self.model_parameters = sanitize_parameters(self.model_parameters)
        self.model = DBSCAN(**self.model_parameters)

    def evaluate_model(self, labels):
        evaluation = super().evaluate_model(labels)
        evaluation["plot_dbscan_clusters"] = plot_clusters_2d_helper(
            self.X, labels, title="Wizualizacja Klastrów DBSCAN", is_dbscan=True
        )
        return evaluation


# ===============================================
# Dimensions Reduction (BaseDimensionalityReduction)
# ===============================================


class PCAModel(BaseDimensionalityReduction):
    def create_model(self):
        self.model_parameters = sanitize_parameters(self.model_parameters)
        self.model = PCA(**self.model_parameters)

    def evaluate_model(self, X_transformed):
        evaluation = super().evaluate_model(X_transformed)

        try:
            n_comp_param = self.model_parameters.get("n_components")
            if n_comp_param is None:
                full_pca_model = self.model
            else:
                full_pca_model = PCA(n_components=None)
                full_pca_model.fit(self.X_train)

            fig_scree, ax_scree = plt.subplots(figsize=(10, 6))
            num_components = len(full_pca_model.explained_variance_ratio_)
            components = np.arange(1, num_components + 1)

            ax_scree.plot(
                components,
                full_pca_model.explained_variance_ratio_,
                marker="o",
                linestyle="--",
            )
            ax_scree.set_xlabel("Główna Składowa")
            ax_scree.set_ylabel("Wyjaśniona Wariancja")
            ax_scree.set_title("Wykres Spadków Wariancji (Scree Plot)")
            ax_scree.grid(True)
            evaluation["plot_pca_scree"] = plot_to_base64(fig_scree)

            actual_n_components = self.model.n_components_
            if actual_n_components == 2:
                fig_biplot, ax_biplot = plt.subplots(figsize=(12, 10))
                X_reduced_2d = self.model.transform(self.X_train)

                sns.scatterplot(
                    x=X_reduced_2d[:, 0],
                    y=X_reduced_2d[:, 1],
                    ax=ax_biplot,
                    alpha=0.6,
                    label="Zredukowane Dane Treningowe",
                )
                feature_vectors = self.model.components_.T * np.sqrt(
                    self.model.explained_variance_
                )

                for i, feature in enumerate(self.X_train.columns):
                    ax_biplot.arrow(
                        0,
                        0,
                        feature_vectors[i, 0],
                        feature_vectors[i, 1],
                        head_width=0.05,
                        head_length=0.05,
                        fc="red",
                        ec="red",
                    )
                    ax_biplot.text(
                        feature_vectors[i, 0] * 1.1,
                        feature_vectors[i, 1] * 1.1,
                        feature,
                        color="green",
                        ha="center",
                        va="center",
                    )

                ax_biplot.set_xlabel("Główna Składowa 1")
                ax_biplot.set_ylabel("Główna Składowa 2")
                ax_biplot.set_title("Biplot PCA (Punkty i Wektory Cech)")
                ax_biplot.grid(True)
                ax_biplot.axhline(0, color="gray", linewidth=0.5)
                ax_biplot.axvline(0, color="gray", linewidth=0.5)
                ax_biplot.legend()

                evaluation["plot_pca_biplot"] = plot_to_base64(fig_biplot)

        except Exception as e:
            print(f"Błąd podczas rysowania wizualizacji PCA: {e}")
            evaluation["plot_pca_scree"] = None
            evaluation["plot_pca_biplot"] = None

        return evaluation
