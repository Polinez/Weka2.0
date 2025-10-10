from django.urls import path
from . import views

app_name = 'mlstudio'
urlpatterns = [
    path("", views.studio, name="studio"),
    path("select/<int:dataset_id>/", views.select_dataset, name="select_dataset"),
    path("preprocess/", views.preprocess, name="preprocess"),
    path("models/", views.models, name="models"),
    path("run/", views.run_model, name="run_model"),
    path("visualize/", views.visualize, name="visualize"),
]