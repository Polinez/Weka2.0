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
    raise ValueError("AZURE_POSTGRESQL_CONNECTIONSTRING environment variable is not set or empty.")

try:
    parameters = dict(pair.split('=') for pair in connection_string.split(' '))
except ValueError:
    raise ValueError(f"Invalid connection string format: {connection_string}")

required_azure_keys = ['dbname', 'user', 'password', 'host', 'port']
missing_keys = [key for key in required_azure_keys if key not in parameters]

if missing_keys:
    raise ValueError(f"Missing required database parameters from Azure: {', '.join(missing_keys)}")

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': parameters['dbname'],
        'USER': parameters['user'],
        'PASSWORD': parameters['password'],
        'HOST': parameters['host'],
        'PORT': parameters['port'],
        'OPTIONS': {
            'sslmode': 'require',         
        },
    }
}

# email
EMAIL_HOST_USER = os.environ.get('EMAIL_USER')
EMAIL_HOST_PASSWORD = os.environ.get('EMAIL_PASSWORD')



