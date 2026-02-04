import os 
from .settings import *
from .settings import BASE_DIR

SECRET=os.environ['SECRET']
ALLOWED_HOSTS = [os.environ["WEBSITE_HOSTNAME"]]
CSRF_TRUSTED_ORIGINS = [f"https://{os.environ['WEBSITE_HOSTNAME']}"]
DEBUG = False

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

connection_string = os.environ.get("AZURE_POSTGRESQL_CONNECTIONSTRING")
if not connection_string:
    raise ValueError("AZURE_POSTGRESQL_CONNECTION_STRING environment variable is not set or empty.")

parameters = dict(pair.split('=') for pair in connection_string.split(' '))

required_keys = ['NAME', 'USER', 'PASSWORD', 'HOST', 'PORT']
missing_keys = [key for key in required_keys if key not in parameters]
if missing_keys:
    raise ValueError(f"Missing required database parameters: {', '.join(missing_keys)}")

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': parameters['NAME'],
        'USER': parameters['USER'],
        'PASSWORD': parameters['PASSWORD'],
        'HOST': parameters['HOST'],
        'PORT': parameters['PORT'],
    }
}

# email
EMAIL_HOST_USER = os.environ.get('EMAIL_USER')
EMAIL_HOST_PASSWORD = os.environ.get('EMAIL_PASSWORD')



