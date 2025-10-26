from django.contrib import admin
from .models import MLModel, ModelParameter, CommonParameter, DatasetModelState, MLRun

# Register your models here.
#to add parameters while adding models in admin panel
class ModelParameterInline(admin.TabularInline):
    model = ModelParameter
    extra = 1


@admin.register(MLModel)
class MLModelAdmin(admin.ModelAdmin):
    list_display = ("name", "description")
    inlines = [ModelParameterInline]


@admin.register(ModelParameter)
class ModelParameterAdmin(admin.ModelAdmin):
    list_display = ("model", "name", "value")
    list_filter = ("model",)

@admin.register(CommonParameter)
class CommonParameterAdmin(admin.ModelAdmin):
    list_display = ("name", "value")

@admin.register(DatasetModelState)
class DatasetModelStateAdmin(admin.ModelAdmin):
    list_display = ("user", "dataset", "model")
    list_filter = ("user", "dataset", "model")
    search_fields = ("user__username", "dataset__name", "model__name")

@admin.register(MLRun)
class MLRunAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "dataset", "model", "created_at")
    list_filter = ("user", "dataset", "model", "created_at")
    search_fields = ("user__username", "dataset__name", "model__name")
    readonly_fields = ("created_at", "result")