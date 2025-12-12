# 🧠 Weka 2.0 - ML Studio

**Weka 2.0** is a web-based Machine Learning platform built on the Django framework. The application allows users to upload their own datasets, perform preprocessing, train a wide range of ML models, and visualize results. All directly from the browser, without writing a single line of code.

## 🚀 Key Features

The application offers a comprehensive environment for data analysis:

### 1. Data Management (`loadData`)
* Upload **CSV** files (e.g., `HousingPrices.csv`, `Titanic-Dataset.csv`).
* Browse and manage available datasets.

### 2. Machine Learning Models (`mlstudio`)
Access to popular algorithms from the **Scikit-learn** library:

* **Classification:**
    * Logistic Regression
    * Decision Tree Classifier (with tree visualization)
    * Random Forest Classifier
    * K-Nearest Neighbors (KNN) (with decision boundary visualization)
    * Support Vector Classifier (SVC)
    * Naive Bayes (GaussianNB)

* **Regression:**
    * Linear Regression
    * Decision Tree Regressor
    * Random Forest Regressor
    * Support Vector Regressor (SVR)

* **Unsupervised Learning (Clustering):**
    * K-Means
    * DBSCAN (with noise detection)

* **Dimensionality Reduction:**
    * PCA (Principal Component Analysis) – includes Scree Plot and Biplot.

### 3. Visualization & Evaluation
* Automatic plot generation using Matplotlib and Seaborn.
* Visualization of decision boundaries for 2D models and PCA reduction for multi-dimensional data.
* Interactive model training reports.

### 4. User System (`register`)
* User registration and login.
* Secure access to data.

---

## 🛠️ Tech Stack

The project is built using modern technologies:

* **Backend:** Python 3.10, Django 5.2.5
* **ML & Data Science:** Scikit-learn, Pandas, NumPy, SciPy
* **Visualization:** Matplotlib, Seaborn
* **Frontend:** HTML, CSS (Bootstrap/Crispy Forms)

---

## 📂 File Structure

Here is a simplified schema of the project structure:

```text
Weka2.0/
├── loadData/                 # App for uploading and managing datasets
│   ├── templates/            # HTML templates for data
│   ├── models.py             # Database models for CSV files
│   └── views.py              # File handling logic
├── mlstudio/                 # Core Machine Learning engine
│   ├── views/                # Logic for preprocessing, models, and visualization
│   │   └── ml_models/        # Algorithm implementations (implementations.py)
│   └── templates/            # ML Studio interface templates
├── register/                 # User account management (Login/Signup)
├── Weka2_0/                  # Main project settings (settings.py, urls.py)
├── templates/                # Global templates (base.html, menu)
├── manage.py                 # Django management script
├── requirements.txt          # Python dependencies list
└── howToRun                  # Run instructions
```

## ⚙️ Installation and Setup

Follow these steps to run the project locally.

### Prerequisites
* Python **3.10** (recommended)
* Git

### Step 1: Clone the repository
Clone the repository to your local machine:
```bash
git clone https://github.com/Polinez/Weka2.0.git
cd Weka2.0
```
*(Ensure you are in the folder containing `manage.py`)*

### Step 2: Install dependencies
It is recommended to create a virtual environment (optional), then install the libraries:

```bash
pip install -r requirements.txt
```

### Step 3: Database Migration
Before running the application for the first time, create the database structure:

```bash
python manage.py makemigrations
python manage.py migrate
```

### Step 4: Run the server
To start the development server:

```bash
python manage.py runserver
```
The application will be available at: `http://127.0.0.1:8000/`

---

## 📖 How to use the app?

1.  **Register/Login:** Create an account or log in.
2.  **Load Data:** Go to the data loading section and upload your CSV file (e.g., `HousingPrices.csv`).
3.  **Select Dataset:** Choose the uploaded file for analysis.
4.  **ML Studio:**
    * Select the target column (Target).
    * Choose the problem type (Classification/Regression).
    * Configure model parameters (e.g., tree depth, number of neighbors).
5.  **Run:** Click "Train" to build the model.
6.  **Results:** View performance metrics and generated plots.

---

## 📄 License

Project created for educational/development purposes.
