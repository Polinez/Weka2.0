from django.apps import AppConfig


class PreprocessingConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'preprocessing'
    verbose_name = 'Preprocessing'
    default = True

    def ready(self):
        import preprocessing.signals