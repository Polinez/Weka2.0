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
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

def load_data_from_session(request):
    dataset_id = request.session.get('dataset_id')  # Retrieve dataset_id from session
    if not dataset_id:
        messages.error(request, "Nie wybrano datasetu. Wybierz dataset lub załaduj nowy.")
        return redirect("loadData:load_data")

    dataset = get_object_or_404(Dataset, id=dataset_id, user=request.user)

    train_data = request.session.get('train_data')
    session_dataset_id = request.session.get('dataset_id_for_data')

    # check if dataset not in session
    if train_data and session_dataset_id == dataset.id:
        dataset.data = train_data

    elif not train_data and session_dataset_id == dataset.id:
        messages.warning(request, "Brak podzielonych danych w sesji. Proszę ponownie skonfigurować zadanie.")
        return redirect("loadData:set_target", dataset_id=dataset.id)

    elif session_dataset_id != dataset.id:
        messages.info(request, "Zmieniono aktywny dataset. Proszę skonfigurować zadanie, aby dokonać podziału danych.")
        return redirect("loadData:set_target", dataset_id=dataset.id)


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
        plt.hist(df[col].dropna(), bins=30, alpha=0.6, label=col, edgecolor='black', linewidth=1)

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
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)

        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        return img_str

    except Exception as e:
        print(f"Błąd podczas konwersji wykresu na Base64: {e}")
        return None

    finally:
        if fig:
            plt.close(fig)


def generate_classification_plots(result_data, df, target_column):
    """
    Generates visualizations for CLASSIFICATION.
    Returns a list of Base64 strings.
    """
    plots = []
    y_test = result_data.get('y_test')
    y_pred = result_data.get('y_pred')
    y_pred_proba = result_data.get('y_pred_proba')

    accuracy = result_data.get('accuracy')
    f1 = result_data.get('f1')

    # 1. Accuracy vs F1 Score Bar Plot
    if accuracy is not None and f1 is not None:
        try:
            metric_names = ['Accuracy', 'F1 Score']
            metric_values = [accuracy, f1]

            fig, ax = plt.subplots()

            sns.barplot(x=metric_names, y=metric_values, ax=ax)

            ax.set_xlabel('Metryka')
            ax.set_ylabel('Wartość')
            ax.set_title('Porównanie Metryk: Accuracy vs F1 Score')

            ax.set_ylim(0, 1.1)

            for p in ax.patches:
                ax.annotate(f'{p.get_height():.3f}',
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center',
                            xytext=(0, 9),
                            textcoords='offset points')

            plots.append(plot_to_base64(fig))

        except Exception as e:
            print(f"Błąd rysowania wykresu metryk (Accuracy/F1): {e}")

    # 2. ROC Curve (only for binary classification)
    if y_test and y_pred_proba is not None:
        try:
            # Check if binary classification (2 classes)
            unique_classes = sorted(list(set(y_test)))
            if len(unique_classes) == 2:
                # Get probabilities for positive class (second column or class with higher index)
                if y_pred_proba.shape[1] == 2:
                    y_scores = y_pred_proba[:, 1]  # Probability of positive class
                else:
                    y_scores = y_pred_proba[:, 0]
                
                # Convert y_test to binary (0/1) if needed
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y_test_binary = le.fit_transform(y_test)
                
                # Calculate ROC curve
                fpr, tpr, thresholds = roc_curve(y_test_binary, y_scores)
                auc_score = roc_auc_score(y_test_binary, y_scores)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random (AUC = 0.500)')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('Krzywa ROC (Receiver Operating Characteristic)')
                ax.legend(loc="lower right")
                ax.grid(alpha=0.3)
                
                plots.append(plot_to_base64(fig))
        except Exception as e:
            print(f"Błąd rysowania krzywej ROC: {e}")

    # 3. Confusion Matrix
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

        # 4. Bar Plot of Predicted Class Counts
        try:
            pred_counts = pd.Series(y_pred).value_counts()

            fig, ax = plt.subplots()
            sns.barplot(x=pred_counts.index, y=pred_counts.values, ax=ax, order=pred_counts.index)

            ax.set_xlabel('Przewidziana Klasa')
            ax.set_ylabel('Liczba Obserwacji')
            ax.set_title('Liczba Obserwacji Przewidzianych dla Każdej Klasy')

            # Annotate bars with counts
            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}',
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center',
                            xytext=(0, 9),
                            textcoords='offset points')

            # Adjust y-axis limit for better visibility of annotations
            ax.set_ylim(top=ax.get_ylim()[1] * 1.1)

            if len(pred_counts) > 5:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

            plots.append(plot_to_base64(fig))
        except Exception as e:
            print(f"Błąd rysowania wykresu słupkowego predykcji: {e}")

    return plots


def generate_regression_plots(result_data, df, target_column):
    """
    Generates visualizations for REGRESSION.
    Returns a list of Base64 strings.
    """
    plots = []
    y_test = result_data.get('y_test')
    y_pred = result_data.get('y_pred')

    mae = result_data.get('mean_absolute_error')
    mse = result_data.get('mean_squared_error')
    r2 = result_data.get('r2_score')


    # 1. MAE and MSE Bar Plot
    if mae is not None and mse is not None:
        try:
            metric_names = ['MAE (Śr. Błąd Abs.)', 'MSE (Błąd Kwadrat.)']
            metric_values = [mae, mse]

            fig, ax = plt.subplots()
            sns.barplot(x=metric_names, y=metric_values, ax=ax)
            ax.set_title('Porównanie Błędów Modelu (Im niżej, tym lepiej)')
            ax.set_ylabel('Wartość błędu')

            # Dodaj adnotacje (wartości) nad słupkami
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.3f}',
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center',
                            xytext=(0, 9),
                            textcoords='offset points')

            # Zwiększ limit osi Y, aby etykiety się zmieściły
            ax.set_ylim(top=ax.get_ylim()[1] * 1.1)
            plots.append(plot_to_base64(fig))

        except Exception as e:
            print(f"Błąd rysowania wykresu błędów (MAE/MSE): {e}")

    # 2. R-squared (R2) Bar Plot
    if r2 is not None:
        try:
            fig, ax = plt.subplots()
            sns.barplot(x=['R-squared (R2)'], y=[r2], ax=ax)
            ax.set_title('Współczynnik Determinacji (Im wyżej, tym lepiej)')
            ax.set_ylabel('Wartość R2')

            # Ustaw limity osi Y (R2 jest zwykle w [0, 1], ale może być ujemny)
            if r2 > 0:
                ax.set_ylim(0, max(1.1, r2 * 1.1))
            else:
                ax.set_ylim(min(-0.1, r2 * 1.2), 0.1)  # Zakres dla ujemnego R2

            # Dodaj adnotacje
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.3f}',
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center',
                            xytext=(0, 9),
                            textcoords='offset points')

            plots.append(plot_to_base64(fig))

        except Exception as e:
            print(f"Błąd rysowania wykresu R2: {e}")

    # 3. Plot of Actual vs Predicted
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
    plots = []
    return plots