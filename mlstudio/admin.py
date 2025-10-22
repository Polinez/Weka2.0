from django.contrib import admin
from django.contrib import admin
from .models import  MLModel, ModelParameter

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
    list_display = ("model", "name", "value", "is_common")
    list_filter = ("model", "is_common")