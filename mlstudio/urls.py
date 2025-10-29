from django.urls import path
from . import views
from .views import preprocess_views

app_name = 'mlstudio'
urlpatterns = [
    path("", views.studio, name="studio"),
    path("select/<int:dataset_id>/", views.select_dataset, name="select_dataset"),
    path("preprocess/", preprocess_views.preprocess, name="preprocess"),
    path("preprocess/apply/", preprocess_views.preprocess_apply, name="preprocess_apply"),
    path("preprocess/reset/", preprocess_views.preprocess_reset, name="preprocess_reset"),
    path("models/", views.models, name="models"),
    path("run/", views.run_model, name="run_model"),
    path("visualize/", views.visualize, name="visualize"),
]