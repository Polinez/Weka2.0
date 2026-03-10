"""Seed preprocessing types and ML models with parameters."""

from django.core.management.base import BaseCommand
from preprocessing.models import PreprocessingType
from ml.models import MLModel, ModelParameterDef


class Command(BaseCommand):
    help = "Seed PreprocessingType and ML models with parameter definitions"

    def handle(self, *args, **options):
        self._seed_preprocessing_types()
        self._seed_ml_models()
        self.stdout.write(self.style.SUCCESS("Seed data updated successfully."))

    def _seed_preprocessing_types(self):
        types = [
            ("Imputation", "Uzupełnianie brakujących wartości"),
            ("Encoding", "Kodowanie zmiennych kategorycznych"),
            ("Scaling", "Skalowanie cech numerycznych"),
            ("DropColumn", "Usuwanie kolumny"),
        ]
        for name, desc in types:
            PreprocessingType.objects.update_or_create(
                name=name, defaults={"description": desc}
            )
        self.stdout.write("PreprocessingType seeded.")

    def _seed_ml_models(self):
        models_data = [
            # Classification
            (
                "Regresja Logistyczna",
                "Classification",
                "Regresja logistyczna to algorytm do klasyfikacji, który przewiduje prawdopodobieństwo wystąpienia zdarzenia (np. tak/nie, 0/1). Model wykorzystuje funkcję logistyczną (sigmoidę), aby przekształcić liniową kombinację predyktorów w wartość z zakresu od 0 do 1. Dzięki temu, regresja logistyczna jest efektywnym modelem do zadań klasyfikacji binarnej i wieloklasowej.",
                [
                    (
                        "max_iter",
                        "1000",
                        "int",
                        "Maksymalna liczba iteracji algorytmu optymalizacji. Zwiększ, jeśli model nie osiąga zbieżności.",
                    ),
                    (
                        "C",
                        "1.0",
                        "float",
                        "Siła regularyzacji (ważność danych). Mniejsze C upraszcza model i zmniejsza ryzyko przeuczenia, większe C pozwala dokładniej dopasować dane.",
                    ),
                    (
                        "solver",
                        "lbfgs",
                        "str",
                        "Algorytm optymalizacji używany do uczenia modelu (np. lbfgs, liblinear, saga).",
                    ),
                    (
                        "class_weight",
                        "None",
                        "str",
                        'Korekta nierównowagi klas. Ustaw na "balanced", jeśli jedna kategoria jest znacznie rzadsza.',
                    ),
                ],
            ),
            (
                "Drzewo decyzyjne (Klasyfikacja)",
                "Classification",
                "Drzewo decyzyjne do klasyfikacji działa jak diagram blokowy z serią pytań TAK/NIE. W każdym węźle wybierana jest cecha, która najlepiej dzieli dane na czystsze podgrupy (klasy). Proces ten jest powtarzany aż do osiągnięcia liścia, który przypisuje ostateczną klasę. Kluczowym wyzwaniem jest zapobieganie przeuczeniu, gdzie drzewo staje się zbyt skomplikowane.",
                [
                    (
                        "max_depth",
                        "0",
                        "int",
                        "Maksymalna głębokość drzewa. Mniejsza wartość zapobiega przeuczeniu. 0 oznacza brak limitu.",
                    ),
                    (
                        "min_samples_split",
                        "2",
                        "int",
                        "Minimalna liczba próbek w węźle, aby można było wykonać podział.",
                    ),
                    (
                        "class_weight",
                        "None",
                        "str",
                        'Ważenie klas. Ustaw "balanced", jeśli jedna kategoria jest znacznie rzadsza.',
                    ),
                ],
            ),
            (
                "K-najblizszych sasiadów (KNN)",
                "Classification",
                "KNN to algorytm klasyfikacyjny, który nie uczy się modelu jawnie, lecz zapamiętuje dane treningowe. Klasa nowego punktu jest wyznaczana na podstawie głosowania K najbliższych sąsiadów. Skuteczność zależy od doboru K oraz metryki odległości.",
                [
                    (
                        "n_neighbors",
                        "5",
                        "int",
                        "Liczba głosujących sąsiadów (K). Małe K jest wrażliwe na szum, duże K prowadzi do nadmiernego uśredniania.",
                    ),
                    (
                        "weights",
                        "uniform",
                        "str",
                        'Ważenie głosów sąsiadów. "uniform" daje równe wagi, "distance" wzmacnia wpływ bliższych punktów.',
                    ),
                ],
            ),
            (
                "Naiwny Klasyfikator Bayesa",
                "Classification",
                "Naiwny Klasyfikator Bayesa to szybki algorytm oparty na Twierdzeniu Bayesa i założeniu niezależności cech. Model GaussianNB zakłada rozkład normalny cech numerycznych dla każdej klasy.",
                [
                    (
                        "var_smoothing",
                        "0.000000001",
                        "float",
                        "Stabilizacja wariancji. Mała wartość dodawana do wariancji cech, aby uniknąć problemów numerycznych.",
                    ),
                ],
            ),
            (
                "Lasy Losowe (Klasyfikacja)",
                "Classification",
                "Lasy Losowe to algorytmy zespołowe budujące wiele drzew decyzyjnych na losowych podzbiorach danych i cech. Wynik jest ustalany przez głosowanie większościowe, co poprawia stabilność i ogranicza przeuczenie.",
                [
                    (
                        "n_estimators",
                        "100",
                        "int",
                        "Liczba drzew w lesie. Większa wartość zwykle poprawia dokładność kosztem czasu obliczeń.",
                    ),
                    (
                        "max_depth",
                        "0",
                        "int",
                        "Maksymalna głębokość pojedynczego drzewa. Ograniczenie może poprawić uogólnienie modelu.",
                    ),
                    (
                        "class_weight",
                        "None",
                        "str",
                        'Ważenie klas. Ustaw "balanced", jeśli klasy są nierówne liczebnie.',
                    ),
                ],
            ),
            (
                "Maszyna wektorów nosnych (SVM) (Klasyfikacja)",
                "Classification",
                "SVC to algorytm szukający hiperpłaszczyzny maksymalizującej margines między klasami. Dzięki funkcjom jądra potrafi modelować złożone, nieliniowe granice decyzyjne.",
                [
                    (
                        "C",
                        "1.0",
                        "float",
                        "Twardość marginesu. Niskie C daje miękki margines i toleruje błędy, wysokie C wymusza dokładne dopasowanie.",
                    ),
                    (
                        "kernel",
                        "rbf",
                        "str",
                        'Kształt granicy decyzyjnej. "rbf" dla nieliniowych granic, "linear" dla liniowych zależności.',
                    ),
                    (
                        "class_weight",
                        "None",
                        "str",
                        'Ważenie klas. Opcja "balanced" zapobiega ignorowaniu rzadkich klas.',
                    ),
                ],
            ),
            # Regression
            (
                "Regresja Liniowa",
                "Regression",
                "Regresja liniowa modeluje zależność między zmienną zależną a predyktorami poprzez dopasowanie prostej minimalizującej błąd średniokwadratowy.",
                [
                    (
                        "fit_intercept",
                        "True",
                        "bool",
                        "Czy obliczać wyraz wolny (b). False oznacza, że linia regresji przechodzi przez punkt (0,0).",
                    ),
                    (
                        "positive",
                        "False",
                        "bool",
                        "Wymusza dodatnie współczynniki regresji. Stosowane, gdy cechy mogą mieć tylko dodatni wpływ.",
                    ),
                ],
            ),
            (
                "Drzewo decyzyjne (Regresja)",
                "Regression",
                "Drzewo decyzyjne do regresji przewiduje wartości ciągłe poprzez uśrednianie wartości w liściach. Podziały minimalizują błąd regresji.",
                [
                    (
                        "max_depth",
                        "0",
                        "int",
                        "Maksymalna głębokość drzewa. Ogranicza złożoność i zapobiega przeuczeniu.",
                    ),
                    (
                        "min_samples_split",
                        "2",
                        "int",
                        "Minimalna liczba próbek w węźle, aby można było wykonać podział.",
                    ),
                ],
            ),
            (
                "Lasy Losowe (Regresja)",
                "Regression",
                "Lasy Losowe w regresji uśredniają predykcje wielu drzew decyzyjnych, co poprawia stabilność i dokładność prognoz.",
                [
                    (
                        "n_estimators",
                        "100",
                        "int",
                        "Liczba drzew w lesie. Więcej drzew daje stabilniejsze uśrednianie wyników.",
                    ),
                    (
                        "max_depth",
                        "0",
                        "int",
                        "Maksymalna głębokość drzew. Ograniczenie poprawia uogólnienie modelu.",
                    ),
                ],
            ),
            (
                "Maszyna wektorów nosnych (SVM) (Regresja)",
                "Regression",
                "SVR przewiduje wartości ciągłe, dopasowując funkcję mieszczącą większość punktów w strefie tolerancji epsilon.",
                [
                    (
                        "C",
                        "1.0",
                        "float",
                        "Siła dopasowania. Wysokie C mocniej karze błędy poza strefą tolerancji, niskie C daje łagodniejszy model.",
                    ),
                    (
                        "kernel",
                        "rbf",
                        "str",
                        'Kształt funkcji regresji. "rbf" pozwala modelować nieliniowe zależności.',
                    ),
                ],
            ),
            # Clustering
            (
                "K-Średnich (K-Means)",
                "Clustering",
                "K-Means to algorytm grupujący dane w K klastrów poprzez minimalizację sumy kwadratów odległości punktów od centroidów.",
                [
                    (
                        "n_clusters",
                        "3",
                        "int",
                        "Liczba klastrów (K), które algorytm ma znaleźć. Optymalną wartość można dobrać np. metodą łokcia.",
                    ),
                    (
                        "random_state",
                        "42",
                        "int",
                        "Ziarno losowości zapewniające powtarzalność wyników.",
                    ),
                ],
            ),
            (
                "DBSCAN",
                "Clustering",
                "DBSCAN grupuje punkty na podstawie gęstości, wykrywając klastry o dowolnym kształcie oraz identyfikując szum.",
                [
                    (
                        "eps",
                        "0.5",
                        "float",
                        "Promień sąsiedztwa. Maksymalna odległość, przy której punkty uznawane są za sąsiadów.",
                    ),
                    (
                        "min_samples",
                        "5",
                        "int",
                        "Minimalna liczba punktów w promieniu eps, aby punkt został uznany za rdzeń klastra.",
                    ),
                ],
            ),
            # Dimensionality Reduction
            (
                "Redukcja Wymiarowości (PCA)",
                "Dimensionality_Reduction",
                "PCA redukuje liczbę wymiarów danych, zachowując jak największą część wariancji poprzez projekcję na główne składowe.",
                [
                    (
                        "n_components",
                        "2",
                        "int",
                        "Liczba składowych do zachowania. Określa, ile głównych składowych PCA zostanie użytych. 0 oznacza automatyczny dobór.",
                    ),
                ],
            ),
        ]

        for name, ptype, desc, params in models_data:
            model, _ = MLModel.objects.update_or_create(
                name=name,
                defaults={
                    "type": ptype,
                    "description": desc,
                    "library": "scikit-learn",
                },
            )
            for pname, pval, ptype_str, pdesc in params:
                ModelParameterDef.objects.update_or_create(
                    model=model,
                    name=pname,
                    defaults={
                        "default_value": pval,
                        "data_type": ptype_str,
                        "description": pdesc,
                    },
                )
        self.stdout.write("ML models and parameters seeded.")
