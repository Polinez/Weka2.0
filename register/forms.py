from django.contrib.auth import login, authenticate
from django.contrib.auth.forms import UserCreationForm



class RegisterForm(UserCreationForm):
    class Meta:
        model = UserCreationForm.Meta.model
        fields = ('username', 'email', 'password1', 'password2')

