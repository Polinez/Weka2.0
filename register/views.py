from django.shortcuts import render, redirect
from .forms import RegisterForm
from django.contrib.auth import login


def signup(request):
    if request.method == "POST":
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            # automatically log in the user after successful registration
            login(request, user)
            return redirect("/")
    else:
        form = RegisterForm()

    return render(request, "signup/signup.html", {"form": form})
