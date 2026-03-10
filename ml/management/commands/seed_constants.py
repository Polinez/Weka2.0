"""Seed data constants for ML models and preprocessing types."""

PREPROCESSING_TYPES = [
    ("Imputation", "Uzupełnianie brakujących wartości"),
    ("Encoding", "Kodowanie zmiennych kategorycznych"),
    ("Scaling", "Skalowanie cech numerycznych"),
    ("DropColumn", "Usuwanie kolumny"),
]

# Format per param: (name, default, dtype, options, description)
# options = [] means free-text input, options = [...] means dropdown
ML_MODELS = [
    # ── Classification ──────────────────────────────────────────────────────────
    (
        "Regresja Logistyczna",
        "Classification",
        "Regresja logistyczna to algorytm do klasyfikacji, który przewiduje prawdopodobieństwo "
        "wystąpienia zdarzenia (np. tak/nie, 0/1). Model wykorzystuje funkcję logistyczną (sigmoidę), "
        "aby przekształcić liniową kombinację predyktorów w wartość z zakresu od 0 do 1. Dzięki temu, "
        "regresja logistyczna jest efektywnym modelem do zadań klasyfikacji binarnej i wieloklasowej.",
        [
            (
                "penalty",
                "l2",
                "str",
                ["l2", "l1", "elasticnet", "None"],
                "Rodzaj regularyzacji (kary). Zapobiega przeuczeniu modelu. Uwaga: nie każdy solver obsługuje każdą karę (np. domyślny 'lbfgs' obsługuje tylko 'l2' oraz 'None').",
            ),
            (
                "max_iter",
                "100",
                "int",
                [],
                "Maksymalna liczba iteracji algorytmu optymalizacji. Zwiększ, jeśli model nie osiąga zbieżności.",
            ),
            (
                "C",
                "1.0",
                "float",
                [],
                "Siła regularyzacji (odwrotność kary). Mniejsze C upraszcza model i zmniejsza ryzyko przeuczenia, większe C pozwala dokładniej dopasować dane.",
            ),
            (
                "solver",
                "lbfgs",
                "str",
                ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
                "Algorytm optymalizacji używany do uczenia modelu.",
            ),
            (
                "class_weight",
                "None",
                "str",
                ["None", "balanced"],
                'Korekta nierównowagi klas. Ustaw na "balanced", jeśli jedna kategoria jest znacznie rzadsza.',
            ),
        ],
    ),
    (
        "Drzewo decyzyjne (Klasyfikacja)",
        "Classification",
        "Drzewo decyzyjne do klasyfikacji działa jak diagram blokowy z serią pytań TAK/NIE. W każdym węźle "
        "wybierana jest cecha, która najlepiej dzieli dane na czystsze podgrupy (klasy). Proces ten jest "
        "powtarzany aż do osiągnięcia liścia, który przypisuje ostateczną klasę. Kluczowym wyzwaniem jest "
        "zapobieganie przeuczeniu, gdzie drzewo staje się zbyt skomplikowane.",
        [
            (
                "criterion",
                "gini",
                "str",
                ["gini", "entropy", "log_loss"],
                "Funkcja oceniająca jakość podziału węzła. 'gini' jest często szybsze w obliczeniach, z kolei 'entropy' minimalizuje chaos informacyjny.",
            ),
            (
                "splitter",
                "best",
                "str",
                ["best", "random"],
                "Strategia wyboru podziału. 'best' szuka optymalnego punktu na wszystkich cechach. 'random' przyspiesza działanie i zapobiega przeuczeniu dodając czynnik losowy.",
            ),
            (
                "max_depth",
                "0",
                "int",
                [],
                "Maksymalna głębokość drzewa. Ograniczenie tego parametru to najprostszy sposób na zapobieganie przeuczeniu. 0 oznacza brak limitu.",
            ),
            (
                "min_samples_split",
                "2",
                "int",
                [],
                "Minimalna liczba próbek w węźle wymagana do wykonania jego podziału na dwie gałęzie.",
            ),
            (
                "min_samples_leaf",
                "1",
                "int",
                [],
                "Minimalna liczba próbek, która musi znaleźć się w liściu końcowym. Zwiększenie tej wartości wymusza gładsze, mniej rozdrobnione granice decyzyjne.",
            ),
            (
                "max_features",
                "None",
                "str",
                ["None", "sqrt", "log2"],
                "Liczba cech brana pod uwagę przy szukaniu najlepszego podziału. 'None' oznacza użycie wszystkich cech dostępnych w zbiorze.",
            ),
            (
                "class_weight",
                "None",
                "str",
                ["None", "balanced"],
                'Ważenie klas. Ustaw na "balanced", jeśli występują duże różnice w liczebności klas w Twoich danych.',
            ),
        ],
    ),
    (
        "K-najblizszych sasiadów (KNN)",
        "Classification",
        "KNN to algorytm klasyfikacyjny, który nie uczy się modelu jawnie, lecz zapamiętuje dane treningowe. "
        "Klasa nowego punktu jest wyznaczana na podstawie głosowania K najbliższych sąsiadów. Skuteczność "
        "zależy od doboru K oraz metryki odległości.",
        [
            (
                "n_neighbors",
                "5",
                "int",
                [],
                "Liczba głosujących sąsiadów (K). Małe K jest wrażliwe na szum i anomalie, duże K uodparnia na szum, ale prowadzi do nadmiernego rozmycia granic decyzyjnych.",
            ),
            (
                "weights",
                "uniform",
                "str",
                ["uniform", "distance"],
                '"uniform" daje równe wagi, "distance" wzmacnia wpływ punktów leżących bliżej (bardzo przydatne przy nierównomiernym rozkładzie danych).',
            ),
            (
                "metric",
                "minkowski",
                "str",
                ["minkowski", "euclidean", "manhattan", "chebyshev"],
                "Sposób obliczania odległości między punktami. Różne metryki potrafią drastycznie zmienić wynik algorytmu.",
            ),
            (
                "p",
                "2",
                "int",
                [],
                'Parametr potęgowy używany tylko gdy metric="minkowski". Wartość p=1 oznacza odległość miejską (Manhattan), a p=2 standardową euklidesową.',
            ),
            (
                "algorithm",
                "auto",
                "str",
                ["auto", "ball_tree", "kd_tree", "brute"],
                "Algorytm indeksowania przestrzeni. Opcja 'auto' sama dobierze najszybszą strukturę dla Twoich danych.",
            ),
        ],
    ),
    (
        "Naiwny Klasyfikator Bayesa",
        "Classification",
        "Naiwny Klasyfikator Bayesa to szybki algorytm oparty na Twierdzeniu Bayesa i założeniu niezależności "
        "cech. Model GaussianNB zakłada rozkład normalny cech numerycznych dla każdej klasy.",
        [
            (
                "var_smoothing",
                "0.000000001",
                "float",
                [],
                "Stabilizacja wariancji. Niewielka wartość dodawana do wariancji cech, aby zapobiec dzieleniu przez zero podczas obliczeń numerycznych.",
            ),
        ],
    ),
    (
        "Lasy Losowe (Klasyfikacja)",
        "Classification",
        "Lasy Losowe to algorytmy zespołowe budujące wiele drzew decyzyjnych na losowych podzbiorach danych "
        "i cech. Wynik jest ustalany przez głosowanie większościowe, co poprawia stabilność i ogranicza przeuczenie.",
        [
            (
                "n_estimators",
                "100",
                "int",
                [],
                "Liczba drzew w lesie. Większa wartość zwykle poprawia dokładność kosztem czasu obliczeń.",
            ),
            (
                "criterion",
                "gini",
                "str",
                ["gini", "entropy", "log_loss"],
                "Funkcja oceniająca jakość podziału węzła. Wpływa na sposób budowania poszczególnych drzew.",
            ),
            (
                "max_depth",
                "0",
                "int",
                [],
                "Maksymalna głębokość pojedynczego drzewa (0 = brak limitu). Ograniczenie tego parametru to najlepszy sposób na zapobieganie przeuczeniu (overfitting).",
            ),
            (
                "min_samples_split",
                "2",
                "int",
                [],
                "Minimalna liczba próbek wymagana do wykonania podziału węzła. Wyższe wartości zapobiegają zbytniemu rozdrobnieniu drzewa.",
            ),
            (
                "min_samples_leaf",
                "1",
                "int",
                [],
                "Minimalna liczba próbek, która musi znaleźć się w liściu końcowym. Zwiększenie tej wartości wygładza model.",
            ),
            (
                "max_features",
                "sqrt",
                "str",
                ["sqrt", "log2", "None"],
                "Rozmiar losowego podzbioru cech rozpatrywanego przy każdym podziale. 'sqrt' to standard w klasyfikacji.",
            ),
            (
                "bootstrap",
                "True",
                "bool",
                [],
                "Czy używać losowania próbek ze zwracaniem (bootstrap) do budowy poszczególnych drzew. Odznaczenie oznacza użycie całego zbioru dla każdego drzewa.",
            ),
            (
                "class_weight",
                "None",
                "str",
                ["None", "balanced", "balanced_subsample"],
                'Ważenie klas. Ustaw "balanced", jeśli klasy są mocno nierówne liczebnie.',
            ),
        ],
    ),
    (
        "Maszyna wektorów nosnych (SVM) (Klasyfikacja)",
        "Classification",
        "SVC to algorytm szukający hiperpłaszczyzny maksymalizującej margines między klasami. "
        "Dzięki funkcjom jądra potrafi modelować złożone, nieliniowe granice decyzyjne.",
        [
            (
                "C",
                "1.0",
                "float",
                [],
                "Twardość marginesu. Niskie C daje miękki margines i toleruje błędy (chroni przed przeuczeniem), wysokie C wymusza dokładne dopasowanie.",
            ),
            (
                "kernel",
                "rbf",
                "str",
                ["linear", "poly", "rbf", "sigmoid"],
                '"rbf" dla nieliniowych granic, "linear" dla liniowych zależności.',
            ),
            (
                "degree",
                "3",
                "int",
                [],
                "Stopień wielomianu. Parametr używany tylko wtedy, gdy wybrano kernel 'poly'.",
            ),
            (
                "gamma",
                "scale",
                "str",
                ["scale", "auto"],
                "'scale' dostosowuje się do wariancji danych, co zazwyczaj daje optymalne rezultaty.",
            ),
            (
                "class_weight",
                "None",
                "str",
                ["None", "balanced"],
                '"balanced" zapobiega ignorowaniu rzadkich klas i przesuwa granicę decyzyjną na ich korzyść.',
            ),
        ],
    ),
    # ── Regression ──────────────────────────────────────────────────────────────
    (
        "Regresja Liniowa",
        "Regression",
        "Regresja liniowa modeluje zależność między zmienną zależną a predyktorami poprzez dopasowanie "
        "prostej minimalizującej błąd średniokwadratowy.",
        [
            (
                "fit_intercept",
                "True",
                "bool",
                [],
                "Czy obliczać wyraz wolny (przecięcie z osią Y). Odznacz (False), jeśli dla zerowych cech wynik musi wynosić dokładnie zero.",
            ),
            (
                "positive",
                "False",
                "bool",
                [],
                "Wymusza dodatnie współczynniki regresji. Stosowane, gdy cechy mogą mieć wyłącznie pozytywny wpływ na wynik (np. metraż na cenę mieszkania).",
            ),
        ],
    ),
    (
        "Drzewo decyzyjne (Regresja)",
        "Regression",
        "Drzewo decyzyjne do regresji przewiduje wartości ciągłe poprzez uśrednianie wartości w liściach. "
        "Podziały minimalizują błąd regresji. Model jest podatny na przeuczenie bez nałożenia limitów głębokości.",
        [
            (
                "criterion",
                "squared_error",
                "str",
                ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                "'squared_error' minimalizuje MSE, 'absolute_error' minimalizuje MAE (odporniejsze na wartości odstające).",
            ),
            (
                "splitter",
                "best",
                "str",
                ["best", "random"],
                "'best' wybiera najlepszy podział, 'random' redukuje wariancję modelu.",
            ),
            (
                "max_depth",
                "0",
                "int",
                [],
                "Maksymalna głębokość drzewa. Ogranicza złożoność i skutecznie zapobiega przeuczeniu (0 = brak limitu).",
            ),
            (
                "min_samples_split",
                "2",
                "int",
                [],
                "Minimalna liczba próbek wymagana do podziału węzła wewnętrznego.",
            ),
            (
                "min_samples_leaf",
                "1",
                "int",
                [],
                "Minimalna liczba próbek wymagana w liściu końcowym. Zwiększenie jej 'wygładza' prognozy modelu.",
            ),
            (
                "max_features",
                "None",
                "str",
                ["None", "sqrt", "log2"],
                "'None' oznacza wykorzystanie wszystkich dostępnych zmiennych.",
            ),
        ],
    ),
    (
        "Lasy Losowe (Regresja)",
        "Regression",
        "Lasy Losowe w regresji uśredniają predykcje wielu drzew decyzyjnych, co znacznie poprawia "
        "stabilność i dokładność prognoz w porównaniu do pojedynczego drzewa.",
        [
            (
                "n_estimators",
                "100",
                "int",
                [],
                "Liczba drzew w lesie. Więcej drzew daje stabilniejsze uśrednianie wyników, ale wydłuża czas treningu.",
            ),
            (
                "criterion",
                "squared_error",
                "str",
                ["squared_error", "absolute_error", "friedman_mse", "poisson"],
                "'squared_error' to klasyczny błąd MSE, 'absolute_error' (MAE) jest wolniejszy, ale odporniejszy na szum.",
            ),
            (
                "max_depth",
                "0",
                "int",
                [],
                "Maksymalna głębokość pojedynczego drzewa. 0 oznacza brak limitu.",
            ),
            (
                "min_samples_split",
                "2",
                "int",
                [],
                "Minimalna liczba próbek wymagana do podziału wewnętrznego węzła drzewa.",
            ),
            (
                "min_samples_leaf",
                "1",
                "int",
                [],
                "Minimalna liczba próbek wymagana w liściu końcowym.",
            ),
            (
                "max_features",
                "1.0",
                "str",
                ["1.0", "sqrt", "log2", "None"],
                "'1.0' oznacza użycie wszystkich cech (brak losowania), 'sqrt' losuje część z nich dla większej różnorodności.",
            ),
            (
                "bootstrap",
                "True",
                "bool",
                [],
                "Czy losować podzbiory danych ze zwracaniem (bootstrap) dla każdego drzewa. Rekomendowane: 'True'.",
            ),
        ],
    ),
    (
        "Maszyna wektorów nosnych (SVM) (Regresja)",
        "Regression",
        "SVR przewiduje wartości ciągłe, dopasowując funkcję mieszczącą jak najwięcej punktów w strefie "
        "tolerancji epsilon. Skutecznie radzi sobie ze skomplikowanymi, nieliniowymi zależnościami.",
        [
            (
                "C",
                "1.0",
                "float",
                [],
                "Siła dopasowania. Wysokie C mocniej karze błędy poza strefą tolerancji (ryzyko przeuczenia), niskie C daje łagodniejszy model.",
            ),
            (
                "epsilon",
                "0.1",
                "float",
                [],
                "Szerokość strefy tolerancji. Błędy mniejsze niż epsilon są ignorowane. Większy epsilon daje prostszy model odporny na szum.",
            ),
            (
                "kernel",
                "rbf",
                "str",
                ["linear", "poly", "rbf", "sigmoid"],
                '"rbf" pozwala modelować skomplikowane zależności nieliniowe, "linear" wymusza prostą linię.',
            ),
            (
                "degree",
                "3",
                "int",
                [],
                "Stopień wielomianu. Parametr brany pod uwagę tylko wtedy, gdy wybrano kernel 'poly'.",
            ),
            (
                "gamma",
                "scale",
                "str",
                ["scale", "auto"],
                "Wartość 'scale' automatycznie dostosowuje się do wariancji cech w Twoich danych.",
            ),
        ],
    ),
    # ── Clustering ──────────────────────────────────────────────────────────────
    (
        "K-Średnich (K-Means)",
        "Clustering",
        "K-Means to algorytm grupujący dane w K klastrów poprzez minimalizację sumy kwadratów odległości "
        "punktów od centroidów. Wymaga z góry określonej liczby klastrów.",
        [
            (
                "n_clusters",
                "8",
                "int",
                [],
                "Liczba klastrów (K). Optymalną wartość można dobrać korzystając z wykresu metody łokcia.",
            ),
            (
                "init",
                "k-means++",
                "str",
                ["k-means++", "random"],
                "'k-means++' to inteligentny dobór przyspieszający zbieżność, 'random' to całkowita losowość.",
            ),
            (
                "max_iter",
                "300",
                "int",
                [],
                "Maksymalna liczba iteracji algorytmu dla pojedynczego uruchomienia.",
            ),
            (
                "algorithm",
                "lloyd",
                "str",
                ["lloyd", "elkan"],
                "'lloyd' to klasyczny standard EM, 'elkan' potrafi być szybszy na dużych zbiorach, ale zużywa więcej pamięci RAM.",
            ),
        ],
    ),
    (
        "DBSCAN",
        "Clustering",
        "DBSCAN to algorytm grupowania oparty na gęstości. W przeciwieństwie do K-Means, nie wymaga podania "
        "liczby klastrów z góry, potrafi wykrywać klastry o dowolnych kształtach oraz identyfikuje szum.",
        [
            (
                "eps",
                "0.5",
                "float",
                [],
                "Maksymalna odległość między punktami, aby uznać je za sąsiadów. Zbyt mały = wszystko szumem, zbyt duży = jeden klaster.",
            ),
            (
                "min_samples",
                "5",
                "int",
                [],
                "Minimalna liczba punktów w promieniu eps, aby punkt był uznany za rdzeń klastra.",
            ),
            (
                "metric",
                "euclidean",
                "str",
                ["euclidean", "manhattan", "chebyshev", "minkowski"],
                "Metryka używana do obliczania odległości między punktami.",
            ),
            (
                "algorithm",
                "auto",
                "str",
                ["auto", "ball_tree", "kd_tree", "brute"],
                "'auto' spróbuje dobrać najbardziej odpowiednią metodę na podstawie Twoich danych.",
            ),
        ],
    ),
    # ── Dimensionality Reduction ─────────────────────────────────────────────────
    (
        "Redukcja Wymiarowości (PCA)",
        "Dimensionality_Reduction",
        "PCA (Analiza Głównych Składowych) redukuje liczbę wymiarów danych, tworząc nowy zestaw "
        "nieskorelowanych zmiennych (głównych składowych). Pozwala to na kompresję danych przy zachowaniu "
        "jak największej części oryginalnej wariancji.",
        [
            (
                "n_components",
                "2",
                "int",
                [],
                "Liczba głównych składowych do zachowania. Dla wizualizacji ustaw 2 lub 3. Wartość 0 = zachowanie wszystkich składowych.",
            ),
            (
                "whiten",
                "False",
                "bool",
                [],
                "Whitening przekształca dane tak, aby składowe miały jednostkową wariancję. Może poprawić wyniki modeli takich jak SVM czy KNN.",
            ),
            (
                "svd_solver",
                "auto",
                "str",
                ["auto", "full", "arpack", "randomized", "covariance_eigh"],
                "'auto' dobiera metodę automatycznie na podstawie rozmiaru danych i liczby składowych.",
            ),
        ],
    ),
]
