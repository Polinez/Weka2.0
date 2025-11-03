from django.contrib import admin
from .models import MLModel, ModelParameter, CommonParameter, DatasetModelState, MLRun

# Register your models here.
#to add parameters while adding models in admin panel
class ModelParameterInline(admin.TabularInline):
    model = ModelParameter
    extra = 1


@admin.register(MLModel)
class MLModelAdmin(admin.ModelAdmin):
    list_display = ("name","model_type" ,"description" )
    list_filter = ("model_type",)
    search_fields = ("name", "description")
    inlines = [ModelParameterInline]


@admin.register(ModelParameter)
class ModelParameterAdmin(admin.ModelAdmin):
    list_display = ("model", "name", "value", "data_type", "description")
    list_filter = ("model", "data_type")
    search_fields = ("name", "model__name")

@admin.register(CommonParameter)
class CommonParameterAdmin(admin.ModelAdmin):
    list_display = ("name", "value", "data_type", "description")
    list_filter = ("data_type",)
    search_fields = ("name",)

@admin.register(DatasetModelState)
class DatasetModelStateAdmin(admin.ModelAdmin):
    list_display = ("user", "dataset", "model")
    list_filter = ("user", "dataset", "model")
    readonly_fields = ("default_parameters", "parameters")

@admin.register(MLRun)
class MLRunAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "dataset", "model", "created_at")
    list_filter = ("user", "dataset", "model", "created_at")
    search_fields = ("user__username", "dataset__name", "model__name")
    readonly_fields = ("created_at", "result", "common_parameters", "model_parameters")