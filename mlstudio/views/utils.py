import io

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import base64
from io import BytesIO
import pandas as pd
from django.contrib import messages
from django.shortcuts import get_object_or_404, redirect
from loadData.models import Dataset

def load_data_from_session(request):
    dataset_id = request.session.get('dataset_id')  # Retrieve dataset_id from session

    if not dataset_id:
        messages.error(request, "Nie wybrano datasetu. Wybierz dataset lub załaduj nowy.")
        return redirect("loadData:load_data")

    dataset = get_object_or_404(Dataset, id=dataset_id, user=request.user)

    # check if dataset not in session
    working_data = request.session.get('working_data')
    session_dataset_id = request.session.get('dataset_id_for_data')

    if working_data and session_dataset_id == dataset.id:
        dataset.data = working_data

    return dataset

def get_graph():
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph

def get_plot(dataFrame, columns: list = None):

    plt.figure(figsize=(10, 6))
    df = dataFrame.copy()

    # check if columns are provided
    if columns is None or len(columns) == 0:
        plt.text(0.5, 0.5, "Wybierz kolumny do wizualizacji", ha='center', va='center', fontsize=14)
        plt.axis('off')
        graph = get_graph()
        plt.close()
        return graph

    # check if selected columns exist in DataFrame
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        plt.text(0.5, 0.5, f"Nie znaleziono kolumn: {', '.join(missing_cols)}", ha='center', va='center', fontsize=12)
        plt.axis('off')
        graph = get_graph()
        plt.close()
        return graph

    # Check if selected columns are numeric
    non_numeric = [col for col in columns if not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric:
        msg = "Kolumna " + (", ".join(non_numeric)) + " jest nienumeryczna"
        plt.text(0.5, 0.5, msg, ha='center', va='center', fontsize=13)
        plt.axis('off')
        graph = get_graph()
        plt.close()
        return graph

    # Plot histograms for each selected numeric column
    for col in columns:
        plt.hist(df[col].dropna(), bins=30, alpha=0.5, label=col)

    plt.title("Rozkład wybranych cech numerycznych")
    plt.xlabel("Wartość cechy")
    plt.ylabel("Liczba wystąpień")
    plt.legend()
    plt.grid(alpha=0.3)

    graph = get_graph()
    plt.close()
    return graph

# plots for Visualizations in ML Model Results
def plot_to_base64(fig):
    """Converts figure Matplotlib to string Base64 for HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode('utf-8')
    return data


def generate_classification_plots(result_data, df, target_column):
    """
    Generates visualizations for CLASSIFICATION.
    Returns a list of Base64 strings.
    """
    plots = []
    y_test = result_data.get('y_test')
    y_pred = result_data.get('y_pred')

    # 1. Confusion Matrix
    if y_test and y_pred:
        try:
            cm = confusion_matrix(y_test, y_pred)
            labels = sorted(list(set(y_test) | set(y_pred)))
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues', ax=ax)
            ax.set_xlabel('Przewidziane')
            ax.set_ylabel('Rzeczywiste')
            ax.set_title('Macierz Pomyłek')
            plots.append(plot_to_base64(fig))
        except Exception as e:
            print(f"Błąd rysowania macierzy pomyłek: {e}")

    return plots


def generate_regression_plots(result_data, df, target_column):
    """
    Generates visualizations for REGRESSION.
    Returns a list of Base64 strings.
    """
    plots = []
    y_test = result_data.get('y_test')
    y_pred = result_data.get('y_pred')

    # 1. Plot of Actual vs Predicted
    if y_test and y_pred:
        try:
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.5)

            lims = [min(min(y_test), min(y_pred)), max(max(y_test), max(y_pred))]
            ax.plot(lims, lims, 'r-', alpha=0.75, zorder=0)
            ax.set_xlabel('Rzeczywiste')
            ax.set_ylabel('Przewidziane')
            ax.set_title('Rzeczywiste vs Przewidziane')
            plots.append(plot_to_base64(fig))
        except Exception as e:
            print(f"Błąd rysowania wykresu regresji: {e}")

    return plots


def generate_clustering_plots(result_data, df, target_column):
    """
    Generates visualizations for CLUSTERING.
    Returns a list of Base64 strings.
    """
    plots = []
    labels = result_data.get('labels')

    if labels and len(labels) == len(df):
        try:
            # Adds cluster labels to DataFrame for plotting
            df_plot = df.copy()
            df_plot['cluster'] = labels

            # Try to plot using first two numeric columns
            numeric_cols = df_plot.select_dtypes(include=np.number).columns

            if len(numeric_cols) >= 2:
                x_axis = numeric_cols[0]
                y_axis = numeric_cols[1]

                fig, ax = plt.subplots()
                sns.scatterplot(data=df_plot, x=x_axis, y=y_axis, hue='cluster', palette='deep', ax=ax)
                ax.set_title(f'Wizualizacja klastrów ({y_axis} vs {x_axis})')
                plots.append(plot_to_base64(fig))

            # 2. Plot of cluster counts
            fig_count, ax_count = plt.subplots()
            sns.countplot(x=df_plot['cluster'], ax=ax_count)
            ax_count.set_title('Liczność klastrów')
            plots.append(plot_to_base64(fig_count))

        except Exception as e:
            print(f"Błąd rysowania wykresów klastrowania: {e}")

    return plots


def generate_dim_reduction_plots(result_data, df, target_column):
    """
    Generates visualizations for DIMENSIONALITY REDUCTION.
    Returns a list of Base64 strings
    """
    plots = []
    variance_ratio = result_data.get('explained_variance_ratio')

    if variance_ratio:
        try:
            # 1. Plot of explained variance ratio
            fig, ax = plt.subplots()
            ax.bar(range(1, len(variance_ratio) + 1), variance_ratio, alpha=0.5, align='center', label='Indywidualna wariancja')
            ax.step(range(1, len(variance_ratio) + 1), np.cumsum(variance_ratio), where='mid', label='Skumulowana wariancja')
            ax.set_xlabel('Główne składowe')
            ax.set_ylabel('Współczynnik wyjaśnionej wariancji')
            ax.set_title('Wykres wariancji (PCA)')
            ax.legend(loc='best')
            plots.append(plot_to_base64(fig))
        except Exception as e:
            print(f"Błąd rysowania wykresu PCA: {e}")

    return plots