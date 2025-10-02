from django.urls import path
from . import views

app_name = 'mlstudio'
urlpatterns = [
    path("<int:dataset_id>/", views.studio, name="studio"),
]