from django.urls import path
from loadData import views

urlpatterns = [
    path('', views.load_data, name='load_data'),
    path('delete/<int:dataset_id>', views.delete_dataset, name='delete_dataset'),
]