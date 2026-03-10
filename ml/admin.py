from django.contrib import admin
from .models import MLModel, ModelParameterDef, MLRun


class ModelParameterDefInline(admin.TabularInline):
    model = ModelParameterDef
    extra = 1


@admin.register(MLModel)
class MLModelAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "type", "library", "description")
    list_filter = ("type", "library")
    search_fields = ("name", "description")
    inlines = [ModelParameterDefInline]


@admin.register(MLRun)
class MLRunAdmin(admin.ModelAdmin):
    list_display = (
        "run_id",
        "user",
        "pipeline",
        "model",
        "status",
        "execution_time_ms",
        "created_at",
        "model_binary_path",
    )
    list_filter = ("status", "model", "user")
    search_fields = (
        "run_id",
        "user__username",
        "pipeline__dataset__name",
        "model__name",
    )

    readonly_fields = (
        "run_id",
        "created_at",
        "metrics",
        "used_parameters",
        "plots_paths",
    )
