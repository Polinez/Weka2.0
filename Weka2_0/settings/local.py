from .base import *
from decouple import config, Csv
import dj_database_url

DEBUG = True

# SECRET from .env
SECRET_KEY = config("SECRET_KEY")

# Hosts from .env
ALLOWED_HOSTS = config("ALLOWED_HOSTS", default="127.0.0.1,localhost", cast=Csv())
CSRF_TRUSTED_ORIGINS = config(
    "CSRF_TRUSTED_ORIGINS", default="http://127.0.0.1,http://localhost", cast=Csv()
)

# Database sqlite
DATABASES = {
    "default": dj_database_url.config(
        default=f"sqlite:///{BASE_DIR / 'db.sqlite3'}", conn_max_age=600
    )
}

# Email from .env
EMAIL_HOST_USER = config("EMAIL_USER", default="")
EMAIL_HOST_PASSWORD = config("EMAIL_PASSWORD", default="")
