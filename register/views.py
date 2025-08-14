from django.shortcuts import render,redirect
from .forms import RegisterForm
# Create your views here.

def signup(request):
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            form.save()

        return redirect('/login')
    else:
        form = RegisterForm()

    return render(request, 'signup/signup.html', {'form': form})
