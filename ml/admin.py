from django.contrib import admin
from .models import MLModel, ModelParameterDef, MLRun


class ModelParameterDefInline(admin.TabularInline):
    model = ModelParameterDef
    extra = 1


@admin.register(MLModel)
class MLModelAdmin(admin.ModelAdmin):
    list_display = ("name", "type", "library", "description")
    list_filter = ("type",)
    search_fields = ("name", "description")
    inlines = [ModelParameterDefInline]


@admin.register(MLRun)
class MLRunAdmin(admin.ModelAdmin):
    list_display = ("run_id", "user", "dataset", "model", "status", "created_at")
    list_filter = ("status", "model", "user")
    search_fields = ("user__username", "dataset__name", "model__name")
    readonly_fields = ("created_at", "metrics", "used_parameters", "split_config")
