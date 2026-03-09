"""Plot generation for ML results."""

import io
import base64
import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.preprocessing import LabelEncoder

matplotlib.use("Agg")


def plot_to_base64(fig):
    """Converts Matplotlib figure to Base64 string for HTML."""
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Błąd podczas konwersji wykresu na Base64: {e}")
        return None
    finally:
        if fig:
            plt.close(fig)


def generate_exploration_histogram(df: pd.DataFrame, column: str) -> str | None:
    """
    Generates a histogram for a specific column in the dataframe.
    """
    if not column or column not in df.columns:
        return None

    # Creating figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Checking data logic
    if not pd.api.types.is_numeric_dtype(df[column]):
        ax.text(
            0.5,
            0.5,
            f"Kolumna '{column}' jest nienumeryczna",
            ha="center",
            va="center",
            fontsize=13,
        )
        ax.axis("off")
    else:
        # Drawing histogram
        data_to_plot = df[column].dropna()
        if data_to_plot.empty:
            ax.text(0.5, 0.5, "Brak danych do wyświetlenia", ha="center", va="center")
        else:
            ax.hist(
                data_to_plot,
                bins=30,
                alpha=0.6,
                label=column,
                edgecolor="black",
                linewidth=1,
            )
            ax.set_title("Rozkład cechy")
            ax.set_xlabel(column)
            ax.set_ylabel("Liczba wystąpień")
            ax.legend()
            ax.grid(alpha=0.3)

    # Conversion to Base64 using helper
    return plot_to_base64(fig)


def generate_classification_plots(result_data, df, target_column):
    """Generates visualizations for CLASSIFICATION."""
    plots = []
    y_test = result_data.get("y_test")
    y_pred = result_data.get("y_pred")
    y_pred_proba = result_data.get("y_pred_proba")
    accuracy = result_data.get("accuracy")
    f1 = result_data.get("f1")

    if accuracy is not None and f1 is not None:
        try:
            fig, ax = plt.subplots()
            sns.barplot(x=["Accuracy", "F1 Score"], y=[accuracy, f1], ax=ax)
            ax.set_xlabel("Metryka")
            ax.set_ylabel("Wartość")
            ax.set_title("Porównanie Metryk: Accuracy vs F1 Score")
            ax.set_ylim(0, 1.1)
            for p in ax.patches:
                ax.annotate(
                    f"{p.get_height():.3f}",
                    (p.get_x() + p.get_width() / 2.0, p.get_height()),
                    ha="center",
                    va="center",
                    xytext=(0, 9),
                    textcoords="offset points",
                )
            plots.append(plot_to_base64(fig))
        except Exception as e:
            print(f"Błąd rysowania wykresu metryk: {e}")

    if y_test and y_pred_proba is not None:
        try:
            y_pred_proba = np.array(y_pred_proba)
            unique_classes = sorted(list(set(y_test)))
            if len(unique_classes) == 2:
                y_scores = (
                    y_pred_proba[:, 1]
                    if y_pred_proba.shape[1] == 2
                    else y_pred_proba[:, 0]
                )
                le = LabelEncoder()
                y_test_binary = le.fit_transform(y_test)
                fpr, tpr, _ = roc_curve(y_test_binary, y_scores)
                auc_score = roc_auc_score(y_test_binary, y_scores)
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(
                    fpr,
                    tpr,
                    color="darkorange",
                    lw=2,
                    label=f"ROC curve (AUC = {auc_score:.3f})",
                )
                ax.plot(
                    [0, 1],
                    [0, 1],
                    color="navy",
                    lw=2,
                    linestyle="--",
                    label="Random (AUC = 0.500)",
                )
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("Krzywa ROC")
                ax.legend(loc="lower right")
                ax.grid(alpha=0.3)
                plots.append(plot_to_base64(fig))
        except Exception as e:
            print(f"Błąd rysowania krzywej ROC: {e}")

    if y_test and y_pred:
        try:
            cm = confusion_matrix(y_test, y_pred)
            labels = sorted(list(set(y_test) | set(y_pred)))
            fig, ax = plt.subplots()
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                xticklabels=labels,
                yticklabels=labels,
                cmap="Blues",
                ax=ax,
            )
            ax.set_xlabel("Przewidziane")
            ax.set_ylabel("Rzeczywiste")
            ax.set_title("Macierz Pomyłek")
            plots.append(plot_to_base64(fig))
        except Exception as e:
            print(f"Błąd rysowania macierzy pomyłek: {e}")

        try:
            pred_counts = pd.Series(y_pred).value_counts()
            fig, ax = plt.subplots()
            sns.barplot(
                x=pred_counts.index,
                y=pred_counts.values,
                ax=ax,
                order=pred_counts.index,
            )
            ax.set_xlabel("Przewidziana Klasa")
            ax.set_ylabel("Liczba Obserwacji")
            ax.set_title("Liczba Obserwacji Przewidzianych dla Każdej Klasy")
            for p in ax.patches:
                ax.annotate(
                    f"{int(p.get_height())}",
                    (p.get_x() + p.get_width() / 2.0, p.get_height()),
                    ha="center",
                    va="center",
                    xytext=(0, 9),
                    textcoords="offset points",
                )
            ax.set_ylim(top=ax.get_ylim()[1] * 1.1)
            if len(pred_counts) > 5:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            plots.append(plot_to_base64(fig))
        except Exception as e:
            print(f"Błąd rysowania wykresu słupkowego: {e}")

    return plots


def generate_regression_plots(result_data, df, target_column):
    """Generates visualizations for REGRESSION."""
    plots = []
    y_test = result_data.get("y_test")
    y_pred = result_data.get("y_pred")
    mae = result_data.get("mean_absolute_error")
    mse = result_data.get("mean_squared_error")
    r2 = result_data.get("r2_score")

    if mae is not None and mse is not None:
        try:
            fig, ax = plt.subplots()
            sns.barplot(x=["MAE", "MSE"], y=[mae, mse], ax=ax)
            ax.set_title("Porównanie Błędów Modelu")
            ax.set_ylabel("Wartość błędu")
            for p in ax.patches:
                ax.annotate(
                    f"{p.get_height():.3f}",
                    (p.get_x() + p.get_width() / 2.0, p.get_height()),
                    ha="center",
                    va="center",
                    xytext=(0, 9),
                    textcoords="offset points",
                )
            ax.set_ylim(top=ax.get_ylim()[1] * 1.1)
            plots.append(plot_to_base64(fig))
        except Exception as e:
            print(f"Błąd rysowania wykresu błędów: {e}")

    if r2 is not None:
        try:
            fig, ax = plt.subplots()
            sns.barplot(x=["R2"], y=[r2], ax=ax)
            ax.set_title("Współczynnik Determinacji R2")
            ax.set_ylabel("Wartość R2")
            for p in ax.patches:
                ax.annotate(
                    f"{p.get_height():.3f}",
                    (p.get_x() + p.get_width() / 2.0, p.get_height()),
                    ha="center",
                    va="center",
                    xytext=(0, 9),
                    textcoords="offset points",
                )
            plots.append(plot_to_base64(fig))
        except Exception as e:
            print(f"Błąd rysowania wykresu R2: {e}")

    if y_test and y_pred:
        try:
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.5)
            lims = [min(min(y_test), min(y_pred)), max(max(y_test), max(y_pred))]
            ax.plot(lims, lims, "r-", alpha=0.75, zorder=0)
            ax.set_xlabel("Rzeczywiste")
            ax.set_ylabel("Przewidziane")
            ax.set_title("Rzeczywiste vs Przewidziane")
            plots.append(plot_to_base64(fig))
        except Exception as e:
            print(f"Błąd rysowania wykresu regresji: {e}")

    return plots


def generate_clustering_plots(result_data, df, target_column):
    """Generates visualizations for CLUSTERING."""
    plots = []
    labels = result_data.get("labels")
    if labels and len(labels) == len(df):
        try:
            df_plot = df.copy()
            df_plot["cluster"] = labels
            numeric_cols = df_plot.select_dtypes(include=np.number).columns
            if len(numeric_cols) >= 2:
                fig, ax = plt.subplots()
                sns.scatterplot(
                    data=df_plot,
                    x=numeric_cols[0],
                    y=numeric_cols[1],
                    hue="cluster",
                    palette="deep",
                    ax=ax,
                )
                ax.set_title("Wizualizacja klastrów")
                plots.append(plot_to_base64(fig))
            fig_count, ax_count = plt.subplots()
            sns.countplot(x=df_plot["cluster"], ax=ax_count)
            ax_count.set_title("Liczność klastrów")
            plots.append(plot_to_base64(fig_count))
        except Exception as e:
            print(f"Błąd rysowania wykresów klastrowania: {e}")
    return plots


def generate_dim_reduction_plots(result_data, df, target_column):
    """Generates visualizations for DIMENSIONALITY_REDUCTION."""
    return []
