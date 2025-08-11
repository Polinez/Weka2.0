from django.shortcuts import render
from .models import Dataset

# Create your views here.


def load_data(request):
    """
    View function to load datasets and render the index page.
    """
    try:
        datasets = Dataset.objects.all()
        error = None
    except Exception as e:
        # Handle any exceptions that occur during dataset retrieval
        datasets = []
        error = str(e)
    return render(request, 'index.html', {"error": error, "datasets": datasets})