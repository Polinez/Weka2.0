from django.urls import path
from .views import dataset_views
from .views import models_views
from .views import preprocess_views
from .views import run_views
from .views import visualize_views

app_name = "ml"
urlpatterns = [
    path("", dataset_views.studio, name="studio"),
    path(
        "select/<uuid:dataset_id>/", dataset_views.select_dataset, name="select_dataset"
    ),
    path("preprocess/", preprocess_views.preprocess, name="preprocess"),
    path(
        "preprocess/apply/", preprocess_views.preprocess_apply, name="preprocess_apply"
    ),
    path(
        "preprocess/reset/", preprocess_views.preprocess_reset, name="preprocess_reset"
    ),
    path("models/", models_views.models, name="models"),
    path("run/", run_views.run_model, name="run_model"),
    path("run/delete/", run_views.delete_run, name="delete_run"),
    path(
        "run/download/<uuid:run_id>/", run_views.download_model, name="download_model"
    ),
    path("visualize/", visualize_views.visualize, name="visualize"),
]
