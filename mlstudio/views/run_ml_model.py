import io
import pandas as pd
import sklearn

def run_ml_model(dataset, model, common_parameters, model_parameters):
    """
    Funkcja uruchamiająca ML na podstawie modelu i parametrów.
    Zwraca wynik (np. słownik z metrykami lub predykcjami).
    """
    # Wczytanie danych
    df = pd.read_csv(io.StringIO(dataset.data))

    # Tutaj implementujesz swój algorytm ML
    # Możesz dopasować model, zrobić predykcje, policzyć metryki itp.
    # Poniżej przykład zwracający fikcyjny wynik
    result = {
        "accuracy": 0.95,
        "predictions": df.head(10).to_dict(orient="records")
    }

    return result