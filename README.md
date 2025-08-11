# Weka2.0

## 📁 Project Structure
```
Weka2.0/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # punkt wejścia FastAPI
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── endpoints.py     # endpointy REST API
│   │   ├── models/              # ML modele + serializacja
│   │   │   ├── __init__.py
│   │   │   ├── trainer.py
│   │   │   ├── predictor.py
│   │   │   ├── my_model.pkl     # wytrenowany model (ew. w S3)
│   │   ├── services/            # logika biznesowa, helpery
│   │   │   ├── __init__.py
│   │   │   ├── data_utils.py
│   │   │   ├── metrics.py
│   │   └── config.py           # ustawienia, np. dotenv
│   ├── tests/
│   │   ├── test_api.py
│   │   └── test_models.py
│   └── Dockerfile
│
├── frontend/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── components/
│   │   │   ├── FileUpload.jsx
│   │   │   ├── Results.jsx
│   │   ├── pages/
│   │   │   ├── Home.jsx
│   │   ├── App.jsx
│   │   ├── index.js
│   ├── package.json
│   ├── vite.config.js / webpack.config.js
│   └── Dockerfile
│
├── .gitignore
├── docker-compose.yml
├── README.md

```
