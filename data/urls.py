from django.urls import path
from . import views

app_name = "data"
urlpatterns = [
    path("", views.load_data, name="load_data"),
    path("delete/<uuid:dataset_id>/", views.delete_dataset, name="delete_dataset"),
    path("restore/<uuid:dataset_id>/", views.restore_dataset, name="restore_dataset"),
    path("set_target/<uuid:dataset_id>/", views.set_target, name="set_target"),
    path("contact/", views.contact, name="contact"),
    path("about/", views.about, name="about"),
]
