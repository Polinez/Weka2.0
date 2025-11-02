import matplotlib.pyplot as plt
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
        return redirect("loadData:loadData")

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
    plt.switch_backend('AGG')

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