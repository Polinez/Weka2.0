# 🧠 Weka 2.0 - ML Studio

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Django](https://img.shields.io/badge/django-5.1.4-green.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)

**Weka 2.0** is a comprehensive, web-based Machine Learning platform built on the Django framework. The application allows users to upload custom datasets, build preprocessing pipelines, train a wide range of ML models, and visualize results—all directly from the browser, without writing a single line of code.

## 🚀 Key Features

The application offers a complete workflow for data analysis:

### 1. Data Management (`data`)
* Upload **CSV** files (e.g., `HousingPrices.csv`, `Titanic-Dataset.csv`).
* Auto-detection of column types.
* Dataset archiving and management.

### 2. Advanced Preprocessing (`preprocessing`)
Build a custom pipeline to prepare your data before training:
* **Imputation:** Handle missing values (Mean, Median, Most Frequent).
* **Encoding:** Convert categorical data (Label Encoder, One-Hot Encoder).
* **Scaling:** Normalize numerical data (StandardScaler, MinMaxScaler).
* **Drop Column:** Remove unnecessary features (e.g., ID columns).

### 3. Machine Learning Models (`ml`)
Access to popular algorithms from the **Scikit-learn** library:

* **Classification:**
    * Logistic Regression
    * Decision Tree Classifier (with tree visualization)
    * Random Forest Classifier
    * K-Nearest Neighbors (KNN)
    * Support Vector Classifier (SVC)
    * Naive Bayes (GaussianNB)

* **Regression:**
    * Linear Regression
    * Decision Tree Regressor
    * Random Forest Regressor
    * Support Vector Regressor (SVR)

* **Unsupervised Learning:**
    * **Clustering:** K-Means, DBSCAN.
    * **Dimensionality Reduction:** PCA (Principal Component Analysis).

### 4. Visualization & Evaluation
* **Interactive Plots:** Confusion Matrix, ROC Curve, Precision-Recall, Scatter Plots, Residuals.
* **Metrics:** Accuracy, F1-Score, MSE, R2, Silhouette Score.
* **Model Artifacts:** Download trained models (`.joblib`) and processed datasets.

---

## 🛠️ Tech Stack

The project is built using modern technologies:

* **Backend:** Python 3.10+, Django 5.1.4
* **ML & Data Science:** Scikit-learn, Pandas, NumPy, SciPy
* **Visualization:** Matplotlib, Seaborn
* **Frontend:** HTML5, CSS3 (Bootstrap 5), Django Templates

---

## 📂 File Structure

Overview of the project modules:

```text
Weka2.0/
├── data/                 # Dataset management (Upload, List, Delete)
│   ├── models.py         # Dataset & DatasetColumn models
│   └── views.py          # File handling logic
├── preprocessing/        # Data cleaning pipeline
│   ├── models.py         # Pipeline, Step, PreprocessingType models
│   └── services.py       # Logic for applying transformations
├── ml/                   # Machine Learning engine
│   ├── models.py         # MLModel, MLRun, Parameter definitions
│   ├── services/         # Training logic and Model Registry
│   └── views/            # Views for model selection and execution
├── register/             # User authentication (Login/Signup)
├── Weka2_0/              # Main project configuration
├── templates/            # Global HTML templates
├── manage.py             # Django management script
└── requirements.txt      # Python dependencies
```

## ⚙️ Installation and Setup

Follow these steps to run the project locally.

### Prerequisites
* Python **3.10** or higher
* Git
* Git Bash (recommended for Windows users to run `.sh` scripts)

### Step 1: Clone the repository
```bash
git clone https://github.com/Polinez/Weka2.0.git
cd Weka2.0
```

### Step 2: Environment & Database Setup
You can easily set up the whole project using the provided `ctl` script.

To see all comands:
```bash
bash ctl
```

Initialize virtual environment and install requirements:
```bash
bash ctl env_init
```

Initialize the database, run migrations, and seed initial data (Crucial step to populate ML models and preprocessing types lists):
```bash
bash ctl db_init
```

*(Alternatively, you can manually create a venv, install requirements, run `makemigrations`, `migrate`, and `seed_data` using `manage.py`)*

### Step 3: Run the server
Start the local development server:

```bash
bash ctl run
```

Open your browser and navigate to: `http://127.0.0.1:8000/`

---

## 📖 Workflow

1.  **Load Data (`/data/load/`):** Upload your CSV dataset. The system will automatically detect column types.
2.  **Configuration (`/data/target/<id>/`):** Select the target variable (column to predict) and the problem type (Classification or Regression).
3.  **Preprocessing (`/preprocessing/list/<id>/`):** (Optional) Clean your data by adding steps like:
    * **Imputation:** Handle missing values.
    * **Encoding:** Convert text categories to numbers.
    * **Scaling:** Standardize numerical features.
4.  **Model Selection (`/ml/models/`):** Choose an algorithm suitable for your problem type and tune its hyperparameters (e.g., `n_estimators` for Random Forest).
5.  **Train & Evaluate:** Execute the training process. The system will split the data (Train/Test), train the model, and display detailed metrics (Accuracy, MSE, R2) along with interactive visualizations (Confusion Matrix, ROC Curve, etc.).
