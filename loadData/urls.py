from django.urls import path
from loadData import views

app_name = 'loadData'
urlpatterns = [
    path('', views.load_data, name='load_data'),
    path('delete/<int:dataset_id>', views.delete_dataset, name='delete_dataset'),
    path('set_target/<int:dataset_id>/', views.set_target, name='set_target'),
]