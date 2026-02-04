import os
from .base import *
from decouple import config


DEBUG = False

# 1. SECRET KEY
try:
    SECRET_KEY = os.environ['SECRET']
except KeyError:
    # Fallback, if in azure set as SECRET_KEY instead of SECRET
    SECRET_KEY = os.environ.get('SECRET_KEY')

if not SECRET_KEY:
    raise ValueError("Not found SECRET or SECRET_KEY environment variable. Please set it in Azure App Settings.")

# 2. HOSTS
allowed_hosts_env = os.environ.get("WEBSITE_HOSTNAME")
if allowed_hosts_env:
    ALLOWED_HOSTS = [allowed_hosts_env]
    CSRF_TRUSTED_ORIGINS = [f"https://{allowed_hosts_env}"]
else:
    ALLOWED_HOSTS = []

# 3.DATABASE
connection_string = os.environ.get("AZURE_POSTGRESQL_CONNECTIONSTRING")

if not connection_string:
    raise ValueError("AZURE_POSTGRESQL_CONNECTIONSTRING environment variable is not set or empty.")

try:
    # pars string: "key=value key2=value2"
    parameters = dict(pair.split('=') for pair in connection_string.split(' '))
except ValueError:
    raise ValueError(f"Invalid connection string format: {connection_string}")

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': parameters.get('dbname'),
        'USER': parameters.get('user'),
        'PASSWORD': parameters.get('password'),
        'HOST': parameters.get('host'),
        'PORT': parameters.get('port'),
        'OPTIONS': {
            'sslmode': 'require',
        },
    }
}

# 4. STATIC FILES
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

# 5. EMAIL
EMAIL_HOST_USER = os.environ.get('EMAIL_USER')
EMAIL_HOST_PASSWORD = config('EMAIL_PASSWORD', default='')